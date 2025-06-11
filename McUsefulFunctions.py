import os
import numpy as np
from ROOT import *
from multiprocessing import Pool
from array import array
from scipy.stats import gennorm
import UsefulFuncs

EnergyPerPair = 3.62 # eV
FanoFactor = 0.13


_cfg = {} ### global config
_noiseEneFrame = None ### global noiseEneFrame
h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc, h1_ChargeCollectionEfficiency = None, None, None, None, None

def init(cfg):
    global _cfg
    _cfg = cfg

### workflow with parameterizations, i.e., infinite depth sampling
pars_alpha, pars_beta = None, None
gr_AlphaT, gr_BetaT = None, None
def parameterization():
    sensorCode = _cfg['sensorCode']
    zBins = _cfg['zBins']
    biasVoltage = _cfg['hv']
    element = _cfg['element']
    simulationMethod = _cfg['simulationMethod']
    sensorThickness = _cfg['sensorThickness'] # cm
    energy = _cfg['energy'] ### eV
    reultsPath = _cfg['resultsPath']

    print(f'Processing {sensorCode} {biasVoltage}V {element} {simulationMethod} with {zBins} bins')
    xsFiles = [f'{reultsPath}/{sensorCode}_{biasVoltage}V_xs_{element}_{i}_of_{zBins}_{simulationMethod}.npy' for i in range(zBins)]
    ysFiles = [f'{reultsPath}/{sensorCode}_{biasVoltage}V_ys_{element}_{i}_of_{zBins}_{simulationMethod}.npy' for i in range(zBins)]
    ggdParList, ggdParUncertList = [], []

    for i in range(zBins):
        xs = np.load(xsFiles[i])
        ys = np.load(ysFiles[i])
        coords = np.concatenate((xs, ys))
        _rms = np.std(coords)
        _binWidth = _rms / 20
        _nBin = int(100 / _binWidth)
        _h1 = TH1F('_h1', '', _nBin, -_rms*5, _rms*5)
        _h1.FillN(len(coords), array('d', coords), array('d', [1]*len(coords)))
        def ggd(x, par):
            beta = par[0]
            alpha = par[1]
            coef = beta / (2 * alpha * TMath.Gamma(1 / beta))
            return coef * TMath.Exp(-TMath.Power((abs(x[0] - 0) / alpha), beta))
        f1 = TF1('f1', ggd, -5 * _rms, 5 * _rms, 2) ### par[0]: beta, par[1]: alpha
        f1.SetParLimits(0, 2, 5)
        f1.SetParLimits(1, .5, 30)
        f1.SetParameters(2., _rms*1.5)
        _h1.Scale(_h1.GetNbinsX()/(10*_rms) /_h1.Integral())
        _h1.Fit(f1, 'Q')
        ggdParList.append((f1.GetParameter(0), f1.GetParameter(1)))
        ggdParUncertList.append((f1.GetParError(0), f1.GetParError(1)))
        del _h1

    z0List = np.linspace(0, sensorThickness, zBins+1)
    z0List = (z0List[:-1] + z0List[1:])/2

    arr_beta = np.array([ggdParList[i][0] for i in range(len(ggdParList))])
    arr_betaUncert = np.array([ggdParUncertList[i][0] for i in range(len(ggdParList))])
    arr_alpha = np.array([ggdParList[i][1] for i in range(len(ggdParList))])
    arr_alphaUncert = np.array([ggdParUncertList[i][1] for i in range(len(ggdParList))])
    arr_z0 = np.array([z0List[i] for i in range(len(ggdParList))])
    arr_t = np.array([getDriftTime_allpix2_8p23(z0) for z0 in z0List])

    c = TCanvas()
    c.SetCanvasSize(1600, 800)
    c.Divide(2, 1)
    # c.SetCanvasSize(800, 1375//2)
    c.SetTopMargin(0.05)
    c.SetRightMargin(0.05)

    global gr_AlphaT, gr_BetaT
    pad2 = c.cd(1)
    gr_AlphaT = TGraphErrors(len(arr_z0), arr_t, arr_alpha, array('d', np.zeros(len(arr_z0))), arr_alphaUncert)
    gr_AlphaT.SetTitle(';Approximated Drift Time [ns];#alpha')
    func_alpha = TF1('func_alpha', '[0] + [1] * sqrt((x)) + [2] * x + [3] * x^2')
    func_alpha.SetLineColor(kRed+1)
    gr_AlphaT.Fit(func_alpha, '')
    # set fit line color
    gr_AlphaT.Draw()
    print(f'Alpha fitting chi2/NDF = {func_alpha.GetChisquare()/func_alpha.GetNDF()}')

    c.cd(2)
    gr_BetaT = TGraphErrors(len(arr_z0), arr_t, arr_beta, array('d', np.zeros(len(arr_z0))), arr_betaUncert)
    func_betaT = TF1('func_betaT', '[0]*(x-[1])^[2]+ [3]*exp([4]*x) + 2')
    func_betaT.SetParameters(0.8, -0.2, -7. -1.2, -0.4)
    func_betaT.SetLineColor(kRed+1)
    gr_BetaT.Fit(func_betaT, '')
    gr_BetaT.SetTitle(';Approximated Drift Time [ns];#beta')
    gr_BetaT.Draw()
    print(f'Beta fitting chi2/NDF = {func_betaT.GetChisquare()/func_betaT.GetNDF()}')

    ### save parameters to file
    global pars_alpha, pars_beta
    pars_alpha = [func_alpha.GetParameter(i) for i in range(func_alpha.GetNpar())]
    pars_beta = [func_betaT.GetParameter(i) for i in range(func_betaT.GetNpar())]
    np.save(f'ggdPar_{sensorCode}_{biasVoltage}V_{element}.npy', pars_alpha + pars_beta)

    return c.Clone()

def getDriftTime_allpix2_8p23(z0):
    ### unit: V, cm, ns, K
    sensorThickness = _cfg['sensorThickness'] # cm
    depletionVoltage = _cfg['depletionVoltage']
    T = _cfg['T']
    hv = _cfg['hv']
    def getEz(z):
        return (hv - depletionVoltage) /  sensorThickness + 2 * depletionVoltage / sensorThickness**2 * z
    v_m = 1.62e8 * T**(-0.52) # cm/s
    E_c = 1.24 * T**1.68 # V/cm
    u0 = v_m / E_c # cm^2/V/s
    k = depletionVoltage * 2 / (sensorThickness * 1e-4)**2 ### um to cm
    t = 1 / u0 * ((sensorThickness - z0)*1e-4/E_c + np.log(getEz(sensorThickness)/getEz(z0)) /k)
    t *= 1e9 # convert to ns
    return t
def singleProcess(thredIdx):
    energy = _cfg['energy'] ### eV
    selectionRange = _cfg['selectionRange']
    Roi = _cfg['Roi'] ### [x1, x2, y1, y2], for noise sampling from measured noise map
    sensorThickness = _cfg['sensorThickness'] # cm
    pixelSize = _cfg['pixelSize'] ### um
    clusterWidth = _cfg['clusterWidth'] ### um
    attenuationLength = _cfg['attenuationLength'] # cm
    global pars_alpha, pars_beta
    func_betaT = TF1('func_betaT', '[0]*(x-[1])^[2]+ [3]*exp([4]*x) + 2')
    func_betaT.SetParameters(pars_beta[0], pars_beta[1], pars_beta[2], pars_beta[3], pars_beta[4])
    func_alphaT = TF1('func_alpha', '[0] + [1] * sqrt((x)) + [2] * x + [3] * x^2')
    func_alphaT.SetParameters(pars_alpha[0], pars_alpha[1], pars_alpha[2], pars_alpha[3])

    _frameWidth = 9
    h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc, h1_ChargeCollectionEfficiency = UsefulFuncs.bookHistograms(energy/1000, suffix = f'_{thredIdx}', isMC= True)
    nEvents = _cfg['nTotalIncident'] // _cfg['nThread']
    ### set random seed
    np.random.seed(thredIdx)
    for i in range(nEvents):
        ### z0 from 0 to sensorThickness
        while True:
            z0 = np.random.exponential(attenuationLength)
            if 0 < z0 < sensorThickness:
                break
        ### drift time, alpha, beta
        t = getDriftTime_allpix2_8p23(z0)
        alpha, beta = func_alphaT.Eval(t), func_betaT.Eval(t)
        ### nChargeCarrierPairs
        nExpectedPair = round(energy / EnergyPerPair)
        nPair = np.random.normal(nExpectedPair, np.sqrt(energy * FanoFactor / EnergyPerPair))
        nPair = round(nPair)
        ### sample the incident position
        x_center = np.random.uniform(pixelSize*_frameWidth//2, pixelSize*_frameWidth//2 + pixelSize)
        y_center = np.random.uniform(pixelSize*_frameWidth//2, pixelSize*_frameWidth//2 + pixelSize)
        ### sample pair positions
        try:
            x_pairs = gennorm.rvs(beta = beta, loc = x_center, scale = alpha, size = nPair)
            y_pairs = gennorm.rvs(beta = beta, loc = y_center, scale = alpha, size = nPair)
        except Exception as e:
            print(f't = {t:.3f} beta = {beta:.3f}, alpha = {alpha:.3f}')
            continue
        x_pairs = x_pairs//pixelSize
        y_pairs = y_pairs//pixelSize
        ### put the pairs into the frame
        _carrierArray = np.zeros((_frameWidth, _frameWidth))
        np.add.at(_carrierArray, (y_pairs.astype(np.int32), x_pairs.astype(np.int32)), 1)
        _energyArray = _carrierArray * EnergyPerPair
        ### spread noise
        x_pedetal = np.random.randint(Roi[0], Roi[1] - _frameWidth)
        y_pedetal = np.random.randint(Roi[2], Roi[3] - _frameWidth)
        _pedestalNoiseFrame = _cfg['noiseEneFrame'][y_pedetal:y_pedetal + _frameWidth, x_pedetal:x_pedetal + _frameWidth]

        shotNoiseFactor = _cfg['shotNoiseFactor']
        _shotNoiseFrame = np.sqrt(_energyArray * shotNoiseFactor)
        _calibrationNoiseFrame = np.ones_like(_energyArray) * _cfg['calibrationNoise']
        _noiseFrame = np.random.normal(0, np.sqrt(_pedestalNoiseFrame**2 + _shotNoiseFrame**2 + _calibrationNoiseFrame**2), size = _energyArray.shape)
        _energyArray += _noiseFrame

        ### truncate to desired cluster width
        ### odd width: highest pixel in the center
        ### even width: highest pixel in the center 2x2 is selected to maximize the cluster energy
        highestPixel = (np.argmax(_energyArray)//_frameWidth, np.argmax(_energyArray)%_frameWidth)
        if clusterWidth%2 == 1:
            pixelArray = _energyArray[highestPixel[0]-clusterWidth//2:highestPixel[0]+clusterWidth//2+1, highestPixel[1]-clusterWidth//2:highestPixel[1]+clusterWidth//2+1]
            carrierArray = _carrierArray[highestPixel[0]-clusterWidth//2:highestPixel[0]+clusterWidth//2+1, highestPixel[1]-clusterWidth//2:highestPixel[1]+clusterWidth//2+1]
        else:
            clusterEnergy = 0 
            for i in range(2):
                for j in range(2):
                    _pixelArray = _energyArray[highestPixel[0]-clusterWidth//2+i:highestPixel[0]+clusterWidth//2+i, highestPixel[1]-clusterWidth//2+j:highestPixel[1]+clusterWidth//2+j]
                    _carrierArray = _carrierArray[highestPixel[0]-clusterWidth//2+i:highestPixel[0]+clusterWidth//2+i, highestPixel[1]-clusterWidth//2+j:highestPixel[1]+clusterWidth//2+j]
                    if np.sum(_pixelArray) > clusterEnergy:
                        clusterEnergy = np.sum(_pixelArray)
                        pixelArray = _pixelArray
                        carrierArray = _carrierArray
        ### fill histograms
        clusterEnergy = np.sum(pixelArray)
        h1_ClusterEnergy.Fill(clusterEnergy/1000)
        if energy - selectionRange*1000 < clusterEnergy < energy + selectionRange*1000:
            h1_PixelEnergy.FillN(len(pixelArray.flatten()), array('d', pixelArray.flatten()/1000), array('d', np.ones(len(pixelArray.flatten()))))
            try:
                x_energyCenter = (np.average(np.arange(clusterWidth), weights = pixelArray.sum(axis=0)) + 0.5)%1
                y_energyCenter = (np.average(np.arange(clusterWidth), weights = pixelArray.sum(axis=1)) + 0.5)%1
                h2_Qc.Fill(x_energyCenter, y_energyCenter)
            except:
                print(f'Error in filling h2_Qc: sum(pixelArray) = {np.sum(pixelArray)}')
                pass
            h1_ChargeCollectionEfficiency.Fill(np.sum(carrierArray) / nPair)
            h1_CenterPixelEnergy.Fill(_energyArray[highestPixel[0], highestPixel[1]]/1000)
    return h1_ClusterEnergy.Clone(), h1_PixelEnergy.Clone(), h1_CenterPixelEnergy.Clone(), h2_Qc.Clone(), h1_ChargeCollectionEfficiency.Clone()

# def process_fromParameters(_pars_alpha, _pars_beta):
def process_fromParameters():
    global pars_alpha, pars_beta
    
    with Pool(processes = _cfg['nThread']) as pool:
        results = pool.map(singleProcess, range(_cfg['nThread']))
    global h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc, h1_ChargeCollectionEfficiency
    h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc, h1_ChargeCollectionEfficiency = UsefulFuncs.bookHistograms(_cfg['energy']/1000, suffix = '_MC', isMC= True)
    for _h1_ClusterEnergy, _h1_PixelEnergy, _h1_CenterPixelEnergy, _h2_Qc, _h1_ChargeCollectionEfficiency in results:
        h1_ClusterEnergy.Add(_h1_ClusterEnergy)
        h1_PixelEnergy.Add(_h1_PixelEnergy)
        h1_CenterPixelEnergy.Add(_h1_CenterPixelEnergy)
        h2_Qc.Add(_h2_Qc)
        h1_ChargeCollectionEfficiency.Add(_h1_ChargeCollectionEfficiency)

    c = TCanvas("c", "c", 1600, 1600)
    c.Divide(2, 2)
    pad1 = c.cd(1)
    h1_ClusterEnergy.SetTitle(';Cluster Energy [keV];Counts')
    h1_ClusterEnergy.Draw()
    pad2 = c.cd(2)
    h1_PixelEnergy.SetTitle(';Pixel Energy [keV];Counts')
    h1_PixelEnergy.Draw()
    pad2.SetLogy()
    pad3 = c.cd(3)
    h1_CenterPixelEnergy.SetTitle(';Center Pixel Energy [keV];Counts')
    h1_CenterPixelEnergy.Draw()
    pad3.SetLogy()
    pad4 = c.cd(4)
    h1_ChargeCollectionEfficiency.SetTitle(';Charge Collection Efficiency;Counts')
    h1_ChargeCollectionEfficiency.Draw()
    clusterWidth = _cfg['clusterWidth']
    print(f'{clusterWidth}x{clusterWidth} cluster, CCE = {h1_ChargeCollectionEfficiency.GetMean()}')
    pad4.SetLogy()
    return c.Clone()

### workflow with ggdDict, i.e., finite depth sampling
def processOneDepth(idxDepth):
    energy = _cfg['energy'] ### eV
    selectionRange = _cfg['selectionRange']
    Roi = _cfg['Roi'] ### [x1, x2, y1, y2], for noise sampling from measured noise map

    pdf = _cfg['pdfList'][idxDepth]
    ggdParams = _cfg['ggdParList'][idxDepth]
    nTotalIncident = _cfg['nTotalIncident']
    pixelSize = _cfg['pixelSize'] ### um
    clusterWidth = _cfg['clusterWidth'] ### um

    _frameWidth = 9
    h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc, h1_ChargeCollectionEfficiency = UsefulFuncs.bookHistograms(energy/1000, suffix = f'_{idxDepth}', isMC= True)
    
    for _ in range(round(nTotalIncident * pdf)):
        nExpectedPair = round(energy / EnergyPerPair)
        nPair = np.random.normal(nExpectedPair, np.sqrt(energy * FanoFactor / EnergyPerPair))
        nPair = round(nPair)
        ### sample the incident position
        x_center = np.random.uniform(pixelSize*_frameWidth//2, pixelSize*_frameWidth//2 + pixelSize)
        y_center = np.random.uniform(pixelSize*_frameWidth//2, pixelSize*_frameWidth//2 + pixelSize)
        ### sample pair positions
        x_pairs = gennorm.rvs(beta = ggdParams[0], loc = x_center, scale = ggdParams[1], size = nPair)
        y_pairs = gennorm.rvs(beta = ggdParams[0], loc = y_center, scale = ggdParams[1], size = nPair)
        x_pairs = x_pairs//pixelSize
        y_pairs = y_pairs//pixelSize
        ### put the pairs into the frame
        _carrierArray = np.zeros((_frameWidth, _frameWidth))
        np.add.at(_carrierArray, (y_pairs.astype(np.int32), x_pairs.astype(np.int32)), 1)
        _energyArray = _carrierArray * EnergyPerPair

        ### spread noise
        x_pedetal = np.random.randint(Roi[0], Roi[1] - _frameWidth)
        y_pedetal = np.random.randint(Roi[2], Roi[3] - _frameWidth)
        _pedestalNoiseFrame = _cfg['noiseEneFrame'][y_pedetal:y_pedetal + _frameWidth, x_pedetal:x_pedetal + _frameWidth]

        shotNoiseFactor = _cfg['shotNoiseFactor']
        _shotNoiseFrame = np.sqrt(_energyArray * shotNoiseFactor)
        _calibrationNoiseFrame = np.ones_like(_energyArray) * _cfg['calibrationNoise']
        _noiseFrame = np.random.normal(0, np.sqrt(_pedestalNoiseFrame**2 + _shotNoiseFrame**2 + _calibrationNoiseFrame**2), size = _energyArray.shape)
        _energyArray += _noiseFrame

        ### truncate to desired cluster width
        ### odd width: highest pixel in the center
        ### even width: highest pixel in the center 2x2 is selected to maximize the cluster energy
        highestPixel = (np.argmax(_energyArray)//_frameWidth, np.argmax(_energyArray)%_frameWidth)
        if clusterWidth%2 == 1:
            pixelArray = _energyArray[highestPixel[0]-clusterWidth//2:highestPixel[0]+clusterWidth//2+1, highestPixel[1]-clusterWidth//2:highestPixel[1]+clusterWidth//2+1]
            carrierArray = _carrierArray[highestPixel[0]-clusterWidth//2:highestPixel[0]+clusterWidth//2+1, highestPixel[1]-clusterWidth//2:highestPixel[1]+clusterWidth//2+1]
        else:
            clusterEnergy = 0 
            for i in range(2):
                for j in range(2):
                    _pixelArray = _energyArray[highestPixel[0]-clusterWidth//2+i:highestPixel[0]+clusterWidth//2+i, highestPixel[1]-clusterWidth//2+j:highestPixel[1]+clusterWidth//2+j]
                    _carrierArray = _carrierArray[highestPixel[0]-clusterWidth//2+i:highestPixel[0]+clusterWidth//2+i, highestPixel[1]-clusterWidth//2+j:highestPixel[1]+clusterWidth//2+j]
                    if np.sum(_pixelArray) > clusterEnergy:
                        clusterEnergy = np.sum(_pixelArray)
                        pixelArray = _pixelArray
                        carrierArray = _carrierArray
        ### fill histograms
        clusterEnergy = np.sum(pixelArray)
        h1_ClusterEnergy.Fill(clusterEnergy/1000)
        if energy - selectionRange*1000 < clusterEnergy < energy + selectionRange*1000:
            h1_PixelEnergy.FillN(len(pixelArray.flatten()), array('d', pixelArray.flatten()/1000), array('d', np.ones(len(pixelArray.flatten()))))
            try:
                x_energyCenter = (np.average(np.arange(clusterWidth), weights = pixelArray.sum(axis=0)) + 0.5)%1
                y_energyCenter = (np.average(np.arange(clusterWidth), weights = pixelArray.sum(axis=1)) + 0.5)%1
                h2_Qc.Fill(x_energyCenter, y_energyCenter)
            except:
                print('Error in filling h2_Qc')
                pass
            h1_ChargeCollectionEfficiency.Fill(np.sum(carrierArray) / nPair)
            h1_CenterPixelEnergy.Fill(_energyArray[highestPixel[0], highestPixel[1]]/1000)
    return h1_ClusterEnergy.Clone(), h1_PixelEnergy.Clone(), h1_CenterPixelEnergy.Clone(), h2_Qc.Clone(), h1_ChargeCollectionEfficiency.Clone()

def process():
    with Pool(processes = _cfg['nThread']) as pool:
        results = pool.map(processOneDepth, range(len(_cfg['pdfList'])))

    global h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc, h1_ChargeCollectionEfficiency
    h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc, h1_ChargeCollectionEfficiency = UsefulFuncs.bookHistograms(_cfg['energy']/1000, suffix = '_MC', isMC= True)
    for _h1_ClusterEnergy, _h1_PixelEnergy, _h1_CenterPixelEnergy, _h2_Qc, _h1_ChargeCollectionEfficiency in results:
        h1_ClusterEnergy.Add(_h1_ClusterEnergy)
        h1_PixelEnergy.Add(_h1_PixelEnergy)
        h1_CenterPixelEnergy.Add(_h1_CenterPixelEnergy)
        h2_Qc.Add(_h2_Qc)
        h1_ChargeCollectionEfficiency.Add(_h1_ChargeCollectionEfficiency)

    c = TCanvas("c", "c", 1600, 1600)
    c.Divide(2, 2)
    pad1 = c.cd(1)
    h1_ClusterEnergy.SetTitle(';Cluster Energy [keV];Counts')
    h1_ClusterEnergy.Draw()

    pad2 = c.cd(2)
    h1_PixelEnergy.SetTitle(';Pixel Energy [keV];Counts')
    h1_PixelEnergy.Draw()
    pad2.SetLogy()

    pad3 = c.cd(3)
    h1_CenterPixelEnergy.SetTitle(';Center Pixel Energy [keV];Counts')
    h1_CenterPixelEnergy.Draw()
    pad3.SetLogy()

    pad4 = c.cd(4)
    h1_ChargeCollectionEfficiency.SetTitle(';Charge Collection Efficiency;Counts')
    h1_ChargeCollectionEfficiency.Draw()
    clusterWidth = _cfg['clusterWidth']
    print(f'{clusterWidth}x{clusterWidth} cluster, CCE = {h1_ChargeCollectionEfficiency.GetMean()}')
    pad4.SetLogy()

    return c.Clone()