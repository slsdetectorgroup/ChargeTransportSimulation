
import os
import numpy as np
from ROOT import *
from multiprocessing import Pool
from array import array

_cfg = {} ### global config
_aduToKev3DMap = None ### global caliFile
_pedestalAduFrame = None ### global pedestalAduFrame
_noiseEneFrame = None ### global noiseEneFrame
h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc = None, None, None, None

def init(cfg):
    global _cfg
    global _aduToKev3DMap
    _cfg = cfg
    _aduToKev3DMap = cfg['caliFile']

def convertAduFrameToEnergyFrame(aduFrame):
    NX, NY = _cfg['NX'], _cfg['NY']
    if _aduToKev3DMap.shape[0] == 1640:
        idxFrame = (np.abs(aduFrame//10)).astype(np.int32)
        idxFrame[idxFrame > _aduToKev3DMap.shape[0]-2] = _aduToKev3DMap.shape[0]-2
        NY, NX = _aduToKev3DMap.shape[1:3]
        _energyFrame0 = _aduToKev3DMap[idxFrame, np.arange(NY).reshape(-1, 1), np.arange(NX)]
        _energyFrame1 = _aduToKev3DMap[idxFrame+1, np.arange(NY).reshape(-1, 1), np.arange(NX)]
        energyFrame = _energyFrame0 + (_energyFrame1 - _energyFrame0)/10 * (np.abs(aduFrame) - idxFrame*10)
        energyFrame *= np.sign(aduFrame)
        if 'energyScalingFactor' in _cfg:
            energyFrame *= _cfg['energyScalingFactor']
    elif _aduToKev3DMap.shape[0] == 1740: ### -1000 ADU to 0 in the first 100 bins
        idxFrame = (aduFrame//10).astype(np.int32) + 100 ### -100:0: the extra negative part
        idxFrame[idxFrame > _aduToKev3DMap.shape[0]-2] = _aduToKev3DMap.shape[0]-2
        NY, NX = _aduToKev3DMap.shape[1:3]
        _energyFrame0 = _aduToKev3DMap[idxFrame, np.arange(NY).reshape(-1, 1), np.arange(NX)]
        _energyFrame1 = _aduToKev3DMap[idxFrame+1, np.arange(NY).reshape(-1, 1), np.arange(NX)]
        energyFrame = _energyFrame0 + (_energyFrame1 - _energyFrame0)/10 * (aduFrame - (idxFrame-100)*10)

    return energyFrame

def bookHistograms(energy, suffix = '', energyBinWidth = 0.1):
    histMaxEnergy = energy * 1.3
    h1_ClusterEnergy = TH1D(f'h1_ClusterEnergy{suffix}', f'h1_ClusterEnergy{suffix}', int(histMaxEnergy//energyBinWidth), -1, histMaxEnergy)
    h1_ClusterEnergy.SetTitle(f'Cluster Energy;Energy [keV];Counts')
    h1_PixelEnergy = TH1D(f'h1_PixelEnergy{suffix}', f'h1_PixelEnergy{suffix}', int(histMaxEnergy//energyBinWidth), -1, histMaxEnergy)
    h1_PixelEnergy.SetTitle(f'Pixel Energy;Energy [keV];Counts')
    h1_CenterPixelEnergy = TH1D(f'h1_CenterPixelEnergy{suffix}', f'h1_CenterPixelEnergy{suffix}', int(histMaxEnergy//energyBinWidth), -1, histMaxEnergy)
    h1_CenterPixelEnergy.SetTitle(f'Center Pixel Energy;Energy [keV];Counts')
    h2_Qc = TH2D(f'h2_Qc{suffix}', f'h2_Qc{suffix}', 101, 0, 101, 0, 101)
    h2_Qc.SetTitle(f'Charge weighted center;#eta_x;#eta_y;Counts')

    return h1_ClusterEnergy.Clone(), h1_PixelEnergy.Clone(), h1_CenterPixelEnergy.Clone(), h2_Qc.Clone()

def _processOneFrame(idxFrame):
    if idxFrame % 10000 == 0:
        print(f'Processing frame {idxFrame}...')
    Energy = _cfg['energy']
    selectionRange = _cfg['selectionRange']
    signalFileNames = _cfg['signalFileNames']
    NX = _cfg['NX']
    NY = _cfg['NY']
    headerSize = _cfg['headerSize']
    Roi = _cfg['Roi']
    nFramePerFile = os.path.getsize(f'{signalFileNames[0]}') // (NX * NY + 56) // 2

    _h1_PixelEnergy, _h1_ClusterEnergy, _h1_CenterPixelEnergy, _h2_Qc = bookHistograms(Energy, f'_{idxFrame}')
    idxFile, idxFrame = divmod(idxFrame, nFramePerFile)
    
    offset = (headerSize + idxFrame * (NX * NY + headerSize)) * 2
    try:
        signalAduFrame = np.fromfile(f'{signalFileNames[idxFile]}', dtype=np.uint16, offset=offset, count=NX * NY).astype(np.int32).reshape(NX, NY)
    except Exception as e:
        return _h1_ClusterEnergy.Clone(), _h1_PixelEnergy.Clone(), _h1_CenterPixelEnergy.Clone(), _h2_Qc.Clone()

    signalAduFrame = signalAduFrame - _pedestalAduFrame
    signalEneFrame = convertAduFrameToEnergyFrame(signalAduFrame)

    for x in range(Roi[0], Roi[1]):
        for y in range(Roi[2], Roi[3]):
            ### 5 sigma noise cut
            if signalEneFrame[y, x] < 5 * _noiseEneFrame[y, x]:
                continue
            ### local maximum
            if signalEneFrame[y, x] != np.max(signalEneFrame[y-1:y+2, x-1:x+2]):
                continue
            ### pile-up cut, 3*simga criterion for the 7*7 cluster excluding the central 3*3 cluster
            r = 3 ### 7*7 cluster
            over3Sigma = signalEneFrame[y-r:y+r+1, x-r:x+r+1] > 3 * _noiseEneFrame[y-r:y+r+1, x-r:x+r+1]
            over3Sigma[r-1:r+2, r-1:r+2] = False
            if np.sum(over3Sigma) > 0:
                continue
            ### bad pixel cut: if bad pixel is in the 5*5 cluster, skip this cluster
            if np.any(_aduToKev3DMap[-1, y-2:y+3, x-2:x+3] < 0):
                continue

            ### get the cluster
            pixleEnergies = signalEneFrame[y-1:y+2, x-1:x+2].flatten()
            clusterEnergy = np.sum(pixleEnergies)
            _h1_ClusterEnergy.Fill(clusterEnergy)
            
            ### energy window cut
            if Energy - selectionRange < clusterEnergy < Energy + selectionRange:
                _h1_CenterPixelEnergy.Fill(signalEneFrame[y, x])
                _h1_PixelEnergy.FillN(len(pixleEnergies), array('d', pixleEnergies), array('d', np.ones(len(pixleEnergies))))
                x_qc = (np.average(np.arange(3), weights=signalEneFrame[y-1:y+2, x-1:x+2].sum(axis=1)) + 0.5)%1
                y_qc = (np.average(np.arange(3), weights=signalEneFrame[y-1:y+2, x-1:x+2].sum(axis=0)) + 0.5)%1
                _h2_Qc.Fill(x_qc, y_qc)

    return _h1_ClusterEnergy.Clone(), _h1_PixelEnergy.Clone(), _h1_CenterPixelEnergy.Clone(), _h2_Qc.Clone()

def getHists():
    from multiprocessing import Pool
    with Pool(16) as p:
        results = p.map(_processOneFrame, range(_cfg['NFrame']))

    global h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc
    h1_ClusterEnergy, h1_PixelEnergy, h1_CenterPixelEnergy, h2_Qc = bookHistograms(_cfg['energy'])
    for _h1_ClusterEnergy, _h1_PixelEnergy, _h1_CenterPixelEnergy, _h2_Qc in results:
        h1_ClusterEnergy.Add(_h1_ClusterEnergy)
        h1_PixelEnergy.Add(_h1_PixelEnergy)
        h1_CenterPixelEnergy.Add(_h1_CenterPixelEnergy)
        h2_Qc.Add(_h2_Qc)
    
    c = TCanvas("c", "c", 1600, 800)
    c.Divide(2, 1)
    pad1 = c.cd(1)
    h1_ClusterEnergy.Draw()
    ### fit the cluster energy
    center = h1_ClusterEnergy.GetBinCenter(h1_ClusterEnergy.GetMaximumBin())
    h1_ClusterEnergy.Fit('gaus', 'Q', '', center - 0.5, center + 1)
    center = h1_ClusterEnergy.GetFunction('gaus').GetParameter(1)
    sigma = h1_ClusterEnergy.GetFunction('gaus').GetParameter(2)
    h1_ClusterEnergy.Fit('gaus', 'Q', '', center - sigma, center + 2*sigma)
    print(f'cluster energy = {h1_ClusterEnergy.GetFunction("gaus").GetParameter(1):.3f} +- {h1_ClusterEnergy.GetFunction("gaus").GetParError(1):.3f}, width = {h1_ClusterEnergy.GetFunction("gaus").GetParameter(2):.3f} +- {h1_ClusterEnergy.GetFunction("gaus").GetParError(2):.3f}, Chi2/NDF = {h1_ClusterEnergy.GetFunction("gaus").GetChisquare()/h1_ClusterEnergy.GetFunction("gaus").GetNDF():.3f}')
    pad2 = c.cd(2); pad2.SetLogy(1)
    h1_PixelEnergy.Draw()

    ### fit the pixel energy
    center = _cfg['energy']
    h1_PixelEnergy.Fit('gaus', 'Q', '', center - 0.1, center + 1)
    center = h1_PixelEnergy.GetFunction('gaus').GetParameter(1)
    sigma = h1_PixelEnergy.GetFunction('gaus').GetParameter(2)
    h1_PixelEnergy.Fit('gaus', 'Q', '', center - 0.5*sigma, center + 2*sigma)
    print(f'pixel energy = {h1_PixelEnergy.GetFunction("gaus").GetParameter(1):.3f} +- {h1_PixelEnergy.GetFunction("gaus").GetParError(1):.3f}, width = {h1_PixelEnergy.GetFunction("gaus").GetParameter(2):.3f} +- {h1_PixelEnergy.GetFunction("gaus").GetParError(2):.3f}, Chi2/NDF = {h1_PixelEnergy.GetFunction("gaus").GetChisquare()/h1_PixelEnergy.GetFunction("gaus").GetNDF():.3f}')
    return c.Clone()

def getPedestalAndNoise():
    NX, NY = _cfg['NX'], _cfg['NY']
    pedestalFileName = _cfg['pedestalFileName']
    pedestalAduFrames = np.fromfile(f'{pedestalFileName}', dtype=np.uint16).astype(np.int32)
    pedestalAduFrames = pedestalAduFrames.reshape(-1, NX * NY + 56)
    pedestalAduFrames = pedestalAduFrames[:, 56:].reshape(-1, NX, NY)
    pedestalAduFrames = pedestalAduFrames[pedestalAduFrames.shape[0]//10:]  # skip the first 10% frames
    global _pedestalAduFrame, _noiseEneFrame
    _pedestalAduFrame = np.mean(pedestalAduFrames, axis=0)
    noiseAduFrame = np.std(pedestalAduFrames, axis=0, ddof=1)
    with Pool(16) as p:
        pedestalEneFrames = p.map(convertAduFrameToEnergyFrame, pedestalAduFrames)
    _noiseEneFrame = np.std(pedestalEneFrames, axis=0, ddof=1)
    print(f'Average noise = {np.mean(_noiseEneFrame):.3f} keV; {np.mean(noiseAduFrame):.3f} ADU')
    del pedestalAduFrames, pedestalEneFrames

def getPedestalAndNoise_simplified():
    NX, NY = _cfg['NX'], _cfg['NY']
    pedestalFileName = _cfg['pedestalFileName']
    pedestalAduFrames = np.fromfile(f'{pedestalFileName}', dtype=np.uint16).astype(np.int32)
    pedestalAduFrames = pedestalAduFrames.reshape(-1, NX * NY + 56)
    pedestalAduFrames = pedestalAduFrames[:, 56:].reshape(-1, NX, NY)
    pedestalAduFrames = pedestalAduFrames[pedestalAduFrames.shape[0]//10:]  # skip the first 10% frames
    global _pedestalAduFrame, _noiseEneFrame
    _pedestalAduFrame = np.mean(pedestalAduFrames, axis=0)
    noiseAduFrame = np.std(pedestalAduFrames, axis=0, ddof=1)
    _noiseEneFrame = convertAduFrameToEnergyFrame(noiseAduFrame)
    
    print(f'Average noise = {np.mean(_noiseEneFrame):.3f} keV; {np.mean(noiseAduFrame):.3f} ADU')
    del pedestalAduFrames
