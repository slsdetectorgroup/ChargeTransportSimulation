### Monte Carlo simulater
from ROOT import *
from array import array
import numpy as np
import torch

### physics constants
kBolzman = 1.380649e-23 # J/K
T = 293 # K
e = 1.60217662e-19 # electron charge in C
epsilon = 11.68 * 8.85418782e-12 # F/um
pi = np.pi
energyPerPair = 3.62 # eV
eulaConstant = 2.7182818285

class McSimulator:
    def __init__(self, config):
        keywords = ['sensorThickness', 'T', 'depletionVoltage', 'appliedVoltage', 'attenuationLength', 'eIncident', 'repulsionInvolved']
        for keyword in keywords:
            if keyword not in config:
                raise ValueError(f"Missing keyword {keyword} in configs")
                
        self.sensorThickness = config['sensorThickness']
        self.T = config['T']
        self.depletionVoltage = config['depletionVoltage']
        self.appliedVoltage = config['appliedVoltage']
        self.attenuationLength = config['attenuationLength']
        self.eIncident = config['eIncident']
        self.repulsionInvolved = config['repulsionInvolved']
        if 'nRepetetion' in config: ### number of repetitions inside the simulationOnce
            self.nRepetetion = config['nRepetetion']
        else:
            self.nRepetetion = 1
        if 'doping' in config:
            self.doping = config['doping']
        self.n = round(self.eIncident / energyPerPair)
        self.tInterval = 0.01 # ns
        self.nThread = 16
        self.zBinning = 64
        self.electronInvolved = True
    
    def get_u_hole111(self, E): ### E in V/um
        ### Jacoboni-Canali Model; allpix-manual-3.0.pdf p76
        E = E * 1e4 ### convert to V/cm
        v_m = 1.62e8 * self.T**(-0.52)
        E_c = 1.24 * self.T**1.68
        beta_h = 0.46 * self.T**0.17
        u = v_m / E_c / (1 + (E/E_c)**beta_h)**(1/beta_h)
        u = u * 1e-1 ### convert to um^/(V*ns)
        return u
    
    def get_u_ele111(self, E): ### E in V/um
        ### Jacoboni-Canali Model; allpix-manual-3.0.pdf p76 + p77
        E = E * 1e4 ### convert to V/cm
        v_m = 1.43e9 * self.T**(-0.87)
        E_c = 1.01 * self.T**1.55
        beta_e = 2.57e-2 * self.T**0.66
        u = v_m / E_c / (1 + (E/E_c)**beta_e)**(1/beta_e)
        u = u * 1e-1 ### convert to um^/(V*ns)
        return u

    def get_u_hole_100(self, E): ### E in V/um
        ### Julian Becker, PhD thesis, p. 52
        ### Very similar results to the Jacoboni-Canali Model
        E = E * 1e4 ### convert to V/cm
        u_0 = 474 * (self.T/300)**(-2.619)
        v_sat = 0.94e7 * (self.T/300)**(-0.226)
        beta = 1.181 * (self.T/300)**(-0.644)
        u = u_0 / (1 + (u_0*E/v_sat)**beta) ** (1/beta) 
        u = u * 1e-1 ### convert to um^/(V*ns)
        return u

    get_u_hole = get_u_hole111
    get_u_ele = get_u_ele111

    def getEz(self, z): ### E field due to bias voltage
        ### deplet from the pixel side (near end)
        Ez = (self.appliedVoltage - self.depletionVoltage) / self.sensorThickness + self.depletionVoltage * 2 / self.sensorThickness * (z) / self.sensorThickness
        if z > self.sensorThickness:
            Ez = (self.appliedVoltage - self.depletionVoltage) / self.sensorThickness + self.depletionVoltage * 2 / self.sensorThickness
        return Ez
    
    
    def simulateOnce(self, z0): ### z0: the intial position in z in um
        ### only return MC simulation results, no change to class variables

        ret_xs = None
        ret_ys = None
        ret_arr_rms = None
        ret_arr_time = None
        for idx_reptetion in range(self.nRepetetion):
            _z0 = z0
            ### initial distribution
            ### Rele = 0.040/ρ * E**1.75, ρ=2.329 is density in g/cm**3, E is initial energy, Rele in um; T. E. Everhart et al., Determination of kilovolt electron energy dissipation vs penetration distance in solid materials, Journal of Applied Physics 42 (1971)
            ### sigma = 1/sqrt(15) * Rele = 0.2572 * Rele, H.-J. Fitting et al., Electron penetration and energy transfer in solid targets, physica status solidi (a) 43 (1977) 185–190.
            sigmaInitial = 0.00443* (self.eIncident/1000)**1.75 ### um 
            xs = np.random.normal(0, sigmaInitial, self.n)
            ys = np.random.normal(0, sigmaInitial, self.n)
            zs = np.random.normal(0, sigmaInitial, self.n) ### relative to z0

            arr_time = array('d')
            arr_rms = array('d')

            totalTime = 0
            while _z0 < self.sensorThickness:
                Ez = self.getEz(_z0)
                u = self.get_u_hole(Ez) ### without repulsion the mobility is the same for all carriers

                ### repulsion
                if self.repulsionInvolved:
                    rs = np.sqrt(xs**2 + ys**2 + zs**2)
                    arrlinds = np.argsort(rs)
                    xs = xs[arrlinds]
                    ys = ys[arrlinds]
                    zs = zs[arrlinds]
                    rs = rs[arrlinds]
                    ### the CDF of the charge distribution
                    cdf_rs = np.array(range(self.n)) / float(self.n)
                    Qr = cdf_rs * self.eIncident / energyPerPair
                    E_rep = Qr * e / (4*np.pi*epsilon*rs*rs) * 1e6 ### V/um
                    E_rep_x = E_rep * xs / rs
                    E_rep_y = E_rep * ys / rs
                    E_rep_z = E_rep * zs / rs
                    Ez = self.getEz(_z0)
                    # u = self.get_u_hole(np.sqrt((Ez + E_rep_z)**2 + E_rep_x**2 + E_rep_y**2))
                    u = self.get_u_hole(np.sqrt(Ez**2 + E_rep_z**2 + E_rep_x**2 + E_rep_y**2))

                    speed = u * E_rep
                    repulsionStep = speed * self.tInterval
                    repulsionStep_x = repulsionStep * xs / rs
                    repulsionStep_y = repulsionStep * ys / rs
                    repulsionStep_z = repulsionStep * zs / rs

                ### random walk
                diffusion = kBolzman * T / e * u # um^2/ns
                randomWalkStep_1D = np.sqrt(2 * diffusion * self.tInterval)# um

                ### update position
                xs += randomWalkStep_1D * (np.random.randint(0, 2, self.n) * 2 - 1)
                ys += randomWalkStep_1D * (np.random.randint(0, 2, self.n) * 2 - 1)
                zs += randomWalkStep_1D * (np.random.randint(0, 2, self.n) * 2 - 1)

                if self.repulsionInvolved:
                    xs += repulsionStep_x
                    ys += repulsionStep_y
                    zs += repulsionStep_z
                
                _z0 += Ez * np.mean(u) * self.tInterval
                totalTime += self.tInterval


                arr_time.append(totalTime)
                arr_rms.append(np.std(xs))

            if idx_reptetion == 0:
                ret_xs = xs
                ret_ys = ys
                ret_arr_rms = arr_rms
                ret_arr_time = arr_time
            else:
                ret_xs = np.concatenate((ret_xs, xs))
                ret_ys = np.concatenate((ret_ys, ys))
        return ret_arr_rms, ret_arr_time, ret_xs, ret_ys

    def simulateOnce2(self, z0): ### z0: the intial position in z in um
        ### another version of the simulation, with repulsion
        ### consider the repulsion between each pair of carriers
        ### very slow, but accelerated using GPU; more precise for 'high energy' X-ray photons (eIncident > 20 keV)

        ret_xs = None
        ret_ys = None
        ret_arr_rms = None
        ret_arr_time = None
        _sumArrRmsSquare = None
        presion = torch.float32 ###
        for idx_reptetion in range(self.nRepetetion):

            ### using PyTorch to speed up the calculation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device {device}")

            ### initial distribution
            sigmaInitial = 0.00443* (self.eIncident/1000)**1.75 ### um
            finalXs = None
            finalYs = None
            if self.electronInvolved:
                xs = torch.normal(0, sigmaInitial, (self.n*2,), device=device, dtype=presion) ### double the number of carriers for both holes and electrons
                ys = torch.normal(0, sigmaInitial, (self.n*2,), device=device, dtype=presion) 
                zs = torch.normal(z0, sigmaInitial, (self.n*2,), device=device, dtype=presion)
                qs = torch.ones(self.n*2, device=device, dtype=torch.int8)
                qs[self.n:] = -1 ### sign of charge carriers
            else:
                xs = torch.normal(0, sigmaInitial, (self.n,), device=device, dtype=presion)
                ys = torch.normal(0, sigmaInitial, (self.n,), device=device, dtype=presion)
                zs = torch.normal(z0, sigmaInitial, (self.n,), device=device, dtype=presion)
                qs = torch.ones(self.n, device=device, dtype=torch.int8)
            mask_holesActive = qs > 0

            arr_time = array('d')
            arr_rms = array('d')
            # arr_Ez, arr_E_rep_z = array('d'), array('d')

            totalTime = 0
            while len(xs) > 0:
                if self.repulsionInvolved:
                    ### repulsion
                    # transform to 2D matrix 
                    dx2D = xs.reshape(1, -1).t() - xs.reshape(1, -1).repeat(len(xs), 1)
                    dy2D = ys.reshape(1, -1).t() - ys.reshape(1, -1).repeat(len(xs), 1)
                    dz2D = zs.reshape(1, -1).t() - zs.reshape(1, -1).repeat(len(xs), 1)
                    qq2D = qs.reshape(1, -1).t() * qs.reshape(1, -1)
                    # calculate the distance between each pair of carriers
                    r2D = torch.sqrt(dx2D**2 + dy2D**2 + dz2D**2)
                    # create a boolean mask to exclude diagonal elements
                    mask = ~torch.eye(r2D.shape[0], dtype=torch.bool, device=device)
                    # apply the mask and reshape
                    r2D = r2D[mask].reshape(r2D.shape[0], -1)
                    dx2D = dx2D[mask].reshape(dx2D.shape[0], -1)
                    dy2D = dy2D[mask].reshape(dy2D.shape[0], -1)
                    dz2D = dz2D[mask].reshape(dz2D.shape[0], -1)
                    qq2D = qq2D[mask].reshape(qq2D.shape[0], -1)
                    E_rep2D_x = qq2D * e / (4*pi*epsilon*r2D*r2D) * 1e6 * dx2D / r2D
                    E_rep2D_y = qq2D * e / (4*pi*epsilon*r2D*r2D) * 1e6 * dy2D / r2D
                    E_rep2D_z = qq2D * e / (4*pi*epsilon*r2D*r2D) * 1e6 * dz2D / r2D
                    E_rep_x = torch.sum(E_rep2D_x, axis=1) ### 
                    E_rep_y = torch.sum(E_rep2D_y, axis=1)
                    E_rep_z = torch.sum(E_rep2D_z, axis=1)
                
                    Ez = (self.appliedVoltage - self.depletionVoltage) / self.sensorThickness + self.depletionVoltage * 2 / self.sensorThickness * (zs) / self.sensorThickness
                    
                    u = self.get_u_hole(torch.sqrt((Ez + E_rep_z)**2 + E_rep_x**2 + E_rep_y**2))
                    u_ele = self.get_u_ele(torch.sqrt((Ez + E_rep_z)**2 + E_rep_x**2 + E_rep_y**2))
                    u[qs < 0] = u_ele[qs < 0]

                    ### mask for the active hole carriers: zs < self.sensorThickness and qs > 0
                    mask_holesActive = (zs < self.sensorThickness) & (zs > 0) & (qs > 0)
                    ### update position due to repulsion
                    xs[mask_holesActive] += u[mask_holesActive] * E_rep_x[mask_holesActive] * self.tInterval
                    ys[mask_holesActive] += u[mask_holesActive] * E_rep_y[mask_holesActive] * self.tInterval
                    zs[mask_holesActive] += u[mask_holesActive] * (E_rep_z+Ez)[mask_holesActive] * self.tInterval

                    ### update position due to random walk
                    randomWalkStep_1D = torch.sqrt(2 * kBolzman * T / e * u * self.tInterval)
                    _size = len(xs[mask_holesActive])
                    xs[mask_holesActive] += randomWalkStep_1D[mask_holesActive] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)
                    ys[mask_holesActive] += randomWalkStep_1D[mask_holesActive] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)
                    zs[mask_holesActive] += randomWalkStep_1D[mask_holesActive] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)

                    ### mask for the active electron carriers: zs > 0 and qs < 0
                    mask_elesActive = (zs > 0) & (qs < 0) & (zs < self.sensorThickness)
                    mask_elesStopped = (zs <= 0) & (qs < 0)
                    xs[mask_elesActive] += u[mask_elesActive] * E_rep_x[mask_elesActive] * self.tInterval
                    ys[mask_elesActive] += u[mask_elesActive] * E_rep_y[mask_elesActive] * self.tInterval
                    zs[mask_elesActive] += u[mask_elesActive] * (E_rep_z-Ez)[mask_elesActive] * self.tInterval

                    ### update position due to random walk
                    _size = len(xs[mask_elesActive])
                    xs[mask_elesActive] += randomWalkStep_1D[mask_elesActive] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)
                    ys[mask_elesActive] += randomWalkStep_1D[mask_elesActive] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)
                    zs[mask_elesActive] += randomWalkStep_1D[mask_elesActive] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)

                    mask_holesCollected = (zs >= self.sensorThickness) & (qs > 0)

                    totalTime += self.tInterval
                    arr_time.append(totalTime)

                    ### calculate rms of holes both active and collected
                    mask_holes = (qs > 0)
                    if finalXs is not None:
                        sumRmsSquare = torch.sum(xs[mask_holes]**2) + torch.sum(finalXs**2)
                        rms = torch.sqrt(sumRmsSquare / (len(xs[mask_holes]) + len(finalXs)))
                    else:
                        sumRmsSquare = torch.sum(xs[mask_holes]**2)
                        rms = torch.sqrt(sumRmsSquare / len(xs[mask_holes]))
                    arr_rms.append(rms.item())
                    
                else:
                    Ez = (self.appliedVoltage - self.depletionVoltage) / self.sensorThickness + self.depletionVoltage * 2 / self.sensorThickness * (zs) / self.sensorThickness
                    u = self.get_u_hole(Ez)
                    u_ele = self.get_u_ele(Ez)
                    u[qs < 0] = u_ele[qs < 0]
                    randomWalkStep_1D = torch.sqrt(2 * kBolzman * T / e * u * self.tInterval)
                    mask_active = (zs < self.sensorThickness) & (zs > 0)
                    _size = len(xs[mask_active])
                    xs[mask_active] += randomWalkStep_1D[mask_active] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)
                    ys[mask_active] += randomWalkStep_1D[mask_active] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)
                    zs[mask_active] += randomWalkStep_1D[mask_active] * (torch.randint(0, 2, (_size,), device=device) * 2 - 1)
                    zs[mask_active] += u[mask_active] * Ez[mask_active] * self.tInterval * qs[mask_active] ### qs is the sign of the charge carriers
                    mask_holesActive = (zs < self.sensorThickness) & (qs > 0)
                    mask_holesCollected = zs >= self.sensorThickness
                    mask_elesActive = (zs > 0) & (qs < 0)
                    mask_elesStopped = zs <= 0
                    totalTime += self.tInterval

                ### set charge the stopped carriers almost 0, avoid the contribution to the repulsion
                if finalXs is None:
                    finalXs = xs[mask_holesCollected]
                    finalYs = ys[mask_holesCollected]
                else:
                    finalXs = torch.cat((finalXs, xs[mask_holesCollected]), 0)
                    finalYs = torch.cat((finalYs, ys[mask_holesCollected]), 0)
                if len(xs[mask_holesActive]) == 0:
                    break
                
                ### remove the stopped carriers
                mask_active = (zs < self.sensorThickness) & (zs > 0)
                xs = xs[mask_active]
                ys = ys[mask_active]
                zs = zs[mask_active]
                qs = qs[mask_active]
                
            if idx_reptetion == 0:
                ret_xs = finalXs.cpu().numpy()
                ret_ys = finalYs.cpu().numpy()
                _sumArrRmsSquare = np.array(arr_rms)**2
                ret_arr_time = arr_time
            else:
                ret_xs = np.concatenate((ret_xs, finalXs.cpu().numpy()))
                ret_ys = np.concatenate((ret_ys, finalYs.cpu().numpy()))
                _size = min(len(_sumArrRmsSquare), len(arr_rms))
                _sumArrRmsSquare[:_size] += np.array(arr_rms[:_size])**2

        _sumArrRmsSquare /= self.nRepetetion
        ret_arr_rms = np.sqrt(_sumArrRmsSquare)
        ret_arr_rms = array('d', ret_arr_rms)
        ret_arr_time = ret_arr_time[:len(ret_arr_rms)]
        return ret_arr_rms, ret_arr_time, ret_xs, ret_ys

    def simulate2(self):
        ### main function to run the simulation
        self.zList = np.linspace(0, self.sensorThickness, self.zBinning+1)
        self.z0List = (self.zList[:-1] + self.zList[1:])/2
        self.pdfList = (1 - np.exp(-self.zList[1:]/self.attenuationLength) - (1 - np.exp(-self.zList[:-1]/self.attenuationLength)))
        self.pdfList /= np.sum(self.pdfList) ### renormalized

        # from multiprocessing import Pool
        # with Pool(self.nThread) as p:
        #     results = p.map(self.simulateOnce, self.z0List)
        results = []
        for z0 in self.z0List:
            print(f'z0 = {z0:.2f}')
            results.append(self.simulateOnce2(z0))
        
        ggdParList = [] ### (beta, alpha) of the Generalized Gaussian Distribution fitting the distribution
        rmsList = []
        for result in results:
            _rms = np.std(result[2])
            print(f'rms = {_rms:.2f}')
            _binWidth = _rms / 20
            _nBin = int(100 / _binWidth)
            h1 = TH1D('h1', 'h1', _nBin, -5 *_rms, 5 * _rms)
            h1.FillN(len(result[2]), array('d', result[2]), np.ones(len(result[2]))) ### xs
            h1.FillN(len(result[3]), array('d', result[3]), np.ones(len(result[3]))) ### ys, as x and y are symmetric
            def ggd(x, par):
                beta = par[0]
                alpha = par[1]
                coef = beta / (2 * alpha * TMath.Gamma(1 / beta))
                return coef * TMath.Exp(-TMath.Power((abs(x[0] - 0) / alpha), beta))
            f1 = TF1('f1', ggd, -5 * _rms, 5 * _rms, 2) ### par[0]: beta, par[1]: alpha
            f1.SetParLimits(0, 2, 5)
            f1.SetParLimits(1, .5, 30)
            f1.SetParameters(2., _rms*1.5)
            if not self.repulsionInvolved:
                ### no repulsion, fix beta = 2 to Gaussian
                f1.SetParameter(0, 2)
                f1.FixParameter(0, 2)

            h1.Scale(h1.GetNbinsX()/(10*_rms) /h1.Integral())
            h1.Fit(f1, 'Q')
            ggdParList.append((f1.GetParameter(0), f1.GetParameter(1)))
            c = TCanvas()
            c.SetCanvasSize(800, 800)
            h1.SetTitle(';X [#mum];Normalized Counts')
            h1.GetYaxis().SetRangeUser(0, 1.4*h1.GetMaximum())
            h1.Draw()
            f1.SetLineWidth(1)
            l = TLegend(0.5, 0.65, 0.8, 0.85)
            l.AddEntry(f1, 'GGD fit', 'lp')
            l.AddEntry(h1, 'MC simulation', 'lp')
            l.Draw('same')
            c.SaveAs(f'figures/No{len(ggdParList)}.png')
            print(f'No{len(ggdParList)}: beta = {f1.GetParameter(0):.2f}, alpha = {f1.GetParameter(1):.2f}, rms = {result[0][-1]:.2f}, time = {result[1][-1]:.2f}')
            print(f'Chi2/NDF = {f1.GetChisquare() / f1.GetNDF():.2f}')
            del h1
        # print(f'weighted RMS = {np.sqrt(np.sum(np.array(rmsList)**2*self.pdfList))}')
        return rmsList, self.pdfList, ggdParList

    def getDriftTime_allpix2_8p23(self, z0):
        ### unit: V, cm, ns, K
        T = self.T
        v_m = 1.62e8 * T**(-0.52) # cm/s
        E_c = 1.24 * T**1.68 # V/cm
        u0 = v_m / E_c # cm^2/V/s
        k = 21.6 * 2 / (320 * 1e-4)**2 ### um to cm
        t = 1 / u0 * ((self.sensorThickness - z0)*1e-4/E_c + log(self.getEz(self.sensorThickness)/self.getEz(z0)) /k)
        t *= 1e9 # convert to ns
        return t

    def simulate2_fromParameters(self, pars_alpha, pars_beta):
        self.zBinning = 1024
        print(f'Overwrite zBinning = {self.zBinning}')
        self.zList = np.linspace(0, self.sensorThickness, self.zBinning+1)
        self.z0List = (self.zList[:-1] + self.zList[1:])/2
        self.pdfList = (1 - np.exp(-self.zList[1:]/self.attenuationLength) - (1 - np.exp(-self.zList[:-1]/self.attenuationLength)))
        self.pdfList /= np.sum(self.pdfList) ### renormalized

        ggdParList = [] ### (beta, alpha) of the Generalized Gaussian Distribution fitting the distribution
        func_betaT = TF1('func_betaT', '[0]*x^[1]+ [2]*exp([3]*x) + 2')
        func_betaT.SetParameters(pars_beta[0], pars_beta[1], pars_beta[2], pars_beta[3])
        func_alphaT = TF1('func_alpha', '[0] + [1] * sqrt((x)) + [2] * x + [3] * x^2')
        func_alphaT.SetParameters(pars_alpha[0], pars_alpha[1], pars_alpha[2], pars_alpha[3])
        
        for z0 in self.z0List:
            t = getDriftTime_allpix2_8p23(z0)
            beta = func_betaT.Eval(t)
            alpha = func_alphaT.Eval(t)
            ggdParList.append((beta, alpha))
        return self.pdfList, ggdParList

    def simulate2_fromFiles(self, xsList, ysList=None):
        ### main function to run the simulation
        self.zList = np.linspace(0, self.sensorThickness, self.zBinning+1)
        self.z0List = (self.zList[:-1] + self.zList[1:])/2
        self.pdfList = (1 - np.exp(-self.zList[1:]/self.attenuationLength) - (1 - np.exp(-self.zList[:-1]/self.attenuationLength)))
        self.pdfList /= np.sum(self.pdfList) ### renormalized

        ggdParList = [] ### (beta, alpha) of the Generalized Gaussian Distribution fitting the distribution
        rmsList = []
        for idx in range(len(xsList)):
            xs = np.load(xsList[idx])
            _rms = np.std(xs)
            # print(f'rms = {_rms:.2f}, {xsList[idx]}')
            _binWidth = _rms / 20
            _nBin = int(100 / _binWidth)
            h1 = TH1D('h1', 'h1', _nBin, -5 *_rms, 5 * _rms)
            h1.FillN(len(xs), array('d', xs), np.ones(len(xs))) ### xs
            if ysList is not None:
                ys = np.load(ysList[idx])
                h1.FillN(len(ys), array('d', ys), np.ones(len(ys)))### ys, as x and y are symmetric
            def ggd(x, par):
                beta = par[0]
                alpha = par[1]
                coef = beta / (2 * alpha * TMath.Gamma(1 / beta))
                return coef * TMath.Exp(-TMath.Power((abs(x[0] - 0) / alpha), beta))
            f1 = TF1('f1', ggd, -5 * _rms, 5 * _rms, 2) ### par[0]: beta, par[1]: alpha
            f1.SetParLimits(0, 2, 5)
            f1.SetParLimits(1, .5, 30)
            f1.SetParameters(2., _rms*1.5)
            h1.Scale(h1.GetNbinsX()/(10*_rms) /h1.Integral())
            h1.Fit(f1, 'Q')
            ggdParList.append((f1.GetParameter(0), f1.GetParameter(1)))
            c = TCanvas()
            c.SetCanvasSize(800, 800)
            h1.SetTitle(';X [#mum];Normalized Counts')
            h1.GetYaxis().SetRangeUser(0, 1.4*h1.GetMaximum())
            h1.Draw()
            f1.SetLineWidth(1)
            l = TLegend(0.5, 0.65, 0.8, 0.85)
            l.AddEntry(f1, 'GGD fit', 'lp')
            l.AddEntry(h1, 'MC simulation', 'lp')
            l.Draw('same')
            c.SaveAs(f'figures/No{len(ggdParList)}.png')
            print(f'No{len(ggdParList)}: beta = {f1.GetParameter(0):.2f}, alpha = {f1.GetParameter(1):.2f}, rms = {_rms:.2f}')
            print(f'Chi2/NDF = {f1.GetChisquare() / f1.GetNDF():.2f}')
            del h1
        # print(f'weighted RMS = {np.sqrt(np.sum(np.array(rmsList)**2*self.pdfList))}')
        return rmsList, self.pdfList, ggdParList

    def simulate(self):
        ### main function to run the simulation
        self.zList = np.linspace(0, self.sensorThickness, self.zBinning+1)
        self.z0List = (self.zList[:-1] + self.zList[1:])/2
        self.pdfList = (1 - np.exp(-self.zList[1:]/self.attenuationLength) - (1 - np.exp(-self.zList[:-1]/self.attenuationLength)))
        self.pdfList /= np.sum(self.pdfList) ### renormalized

        from multiprocessing import Pool
        with Pool(self.nThread) as p:
            results = p.map(self.simulateOnce, self.z0List)
        
        rmsList = []
        ggdParList = [] ### (beta, alpha) of the Generalized Gaussian Distribution fitting the distribution
        for result in results:
            _rms = result[0][-1]
            rmsList.append(_rms)
            _binWidth = _rms / 20
            _nBin = int(100 / _binWidth)
            h1 = TH1D('h1', 'h1', _nBin, -5 *_rms, 5 * _rms)
            h1.FillN(len(result[2]), result[2], np.ones(len(result[2])))
            def ggd(x, par):
                beta = par[0]
                alpha = par[1]
                coef = beta / (2 * alpha * TMath.Gamma(1 / beta))
                return coef * TMath.Exp(-TMath.Power((abs(x[0] - 0) / alpha), beta))
            f1 = TF1('f1', ggd, -5 * _rms, 5 * _rms, 2) ### par[0]: beta, par[1]: alpha
            f1.SetParLimits(0, 2, 5)
            f1.SetParLimits(1, .5, 30)
            f1.SetParameters(2., _rms*1.5)
            if not self.repulsionInvolved:
                ### no repulsion, fix beta = 2 to Gaussian
                f1.SetParameter(0, 2)
                f1.FixParameter(0, 2)

            h1.Scale(h1.GetNbinsX()/(10*_rms) /h1.Integral())
            h1.Fit(f1, 'Q')
            ggdParList.append((f1.GetParameter(0), f1.GetParameter(1)))
            c = TCanvas()
            c.SetCanvasSize(800, 800)
            h1.SetTitle(';X [#mum];Normalized Counts')
            h1.GetYaxis().SetRangeUser(0, 1.4*h1.GetMaximum())
            h1.Draw()
            f1.SetLineWidth(1)
            l = TLegend(0.5, 0.65, 0.8, 0.85)
            l.AddEntry(f1, 'GGD fit', 'lp')
            l.AddEntry(h1, 'MC simulation', 'lp')
            l.Draw('same')
            c.SaveAs(f'figures/No{len(ggdParList)}.png')
            print(f'No{len(ggdParList)}: beta = {f1.GetParameter(0):.2f}, alpha = {f1.GetParameter(1):.2f}, rms = {result[0][-1]:.2f}, time = {result[1][-1]:.2f}, Chi2/NDF = {f1.GetChisquare() / f1.GetNDF():.2f}')
            del h1

        print(f'weighted RMS = {np.sqrt(np.sum(np.array(rmsList)**2*self.pdfList))}')
        return rmsList, self.pdfList, ggdParList