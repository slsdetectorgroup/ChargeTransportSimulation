from ROOT import *
from array import array
import numpy as np

### physics constants
kBolzman = 1.380649e-23 # J/K
T = 293 # K
e = 1.60217662e-19 # electron charge in C
epsilon = 11.68 * 8.85418782e-12 # F/um

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
        if 'n' in config:
            self.n = config['n']
        else:
            self.n = 100000 ### number of groups of carriers (one group can have less than 1 carrier)
        self.tInterval = 0.01 # ns
        self.nThread = 16
        self.zBinning = 64
    
    def get_u_hole111(self, E): ### E in V/um
        ### Jacoboni-Canali Model; allpix-manual-3.0.pdf p76
        E = E * 1e4 ### convert to V/cm
        v_m = 1.62e8 * self.T**(-0.52)
        E_c = 1.24 * self.T**1.68
        beta_h = 0.46 * self.T**0.17
        u = v_m / E_c / (1 + (E/E_c)**beta_h)**(1/beta_h)
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

    def getEz(self, z): ### E field due to bias voltage
        Ez = (self.appliedVoltage - self.depletionVoltage) / self.sensorThickness + self.depletionVoltage * 2 / self.sensorThickness * (z) / self.sensorThickness
        if z > self.sensorThickness:
            Ez = (self.appliedVoltage - self.depletionVoltage) / self.sensorThickness + self.depletionVoltage * 2 / self.sensorThickness
        return Ez
    
    
    def simulateOnce(self, z0): ### z0: the intial position in z in um
        ### only return MC simulation results, no change to class variables

        ret_xs = None
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
                rs = np.sqrt(xs**2 + ys**2 + zs**2)
                arrlinds = np.argsort(rs)
                xs = xs[arrlinds]
                ys = ys[arrlinds]
                zs = zs[arrlinds]
                rs = rs[arrlinds]
                ### the CDF of the charge distribution
                cdf_rs = np.array(range(self.n)) / float(self.n)

                Ez = self.getEz(_z0)
                u = self.get_u_hole(Ez) ### without repulsion the mobility is the same for all carriers

                ### repulsion
                if self.repulsionInvolved:
                    Qr = cdf_rs * self.eIncident / 3.6
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
                ret_arr_rms = arr_rms
                ret_arr_time = arr_time
            else:
                ret_xs = np.concatenate((ret_xs, xs))
        return ret_arr_rms, ret_arr_time, ret_xs
    
    def simulateOnce2(self, z0): ### z0: the intial position in z in um
        ### another version of the simulation, with repulsion
        ### consider the repulsion between each pair of carriers
        ### very slow; results are similar to simulateOnce

        ### initial distribution
        sigmaInitial = 0.0044* (self.eIncident/1000)**1.75 ### um
        xs = np.random.normal(0, sigmaInitial, self.n)
        ys = np.random.normal(0, sigmaInitial, self.n)
        zs = np.random.normal(z0, sigmaInitial, self.n)

        arr_time = array('d')
        arr_rms = array('d')
        arr_Ez, arr_E_rep_z = array('d'), array('d')

        totalTime = 0
        # while z0 < self.sensorThickness:
        while np.min(zs) < self.sensorThickness:
            print(z0, np.average(zs), self.sensorThickness)

            ### repulsion
            q = self.eIncident / 3.6 / self.n
            xs2D = xs.reshape(1, -1).T - np.repeat(xs.reshape(1, -1), len(xs), axis=0)
            ys2D = ys.reshape(1, -1).T - np.repeat(ys.reshape(1, -1), len(xs), axis=0)
            zs2D = zs.reshape(1, -1).T - np.repeat(zs.reshape(1, -1), len(xs), axis=0)
            rs2D = np.sqrt(xs2D**2 + ys2D**2 + zs2D**2)
            rs2D = rs2D[~np.eye(rs2D.shape[0],dtype=bool)].reshape(rs2D.shape[0],-1)
            xs2D = xs2D[~np.eye(xs2D.shape[0],dtype=bool)].reshape(xs2D.shape[0],-1)
            ys2D = ys2D[~np.eye(ys2D.shape[0],dtype=bool)].reshape(ys2D.shape[0],-1)
            zs2D = zs2D[~np.eye(zs2D.shape[0],dtype=bool)].reshape(zs2D.shape[0],-1)
            E_rep2D_x = q * e / (4*np.pi*epsilon*rs2D*rs2D) * 1e6 * xs2D / rs2D
            E_rep2D_y = q * e / (4*np.pi*epsilon*rs2D*rs2D) * 1e6 * ys2D / rs2D
            E_rep2D_z = q * e / (4*np.pi*epsilon*rs2D*rs2D) * 1e6 * zs2D / rs2D
            E_rep_x = np.sum(E_rep2D_x, axis=1)
            E_rep_y = np.sum(E_rep2D_y, axis=1)
            E_rep_z = np.sum(E_rep2D_z, axis=1)
            
            Ez = (self.appliedVoltage - self.depletionVoltage) / self.sensorThickness + self.depletionVoltage * 2 / self.sensorThickness * (zs) / self.sensorThickness
            
            u = self.get_u_hole(np.sqrt((Ez + E_rep_z)**2 + E_rep_x**2 + E_rep_y**2))
            xs[zs < self.sensorThickness] += u[zs < self.sensorThickness] * E_rep_x[zs < self.sensorThickness] * self.tInterval
            ys[zs < self.sensorThickness] += u[zs < self.sensorThickness] * E_rep_y[zs < self.sensorThickness] * self.tInterval
            zs[zs < self.sensorThickness] += u[zs < self.sensorThickness] * (E_rep_z+Ez)[zs < self.sensorThickness] * self.tInterval

            ### random walk
            diffusion = kBolzman * T / e * u # um^2/ns
            randomWalkStep_1D = np.sqrt(2 * diffusion * self.tInterval)# um
            ### update position
            xs[zs < self.sensorThickness] += randomWalkStep_1D[zs < self.sensorThickness] * (np.random.randint(0, 2, len(xs[zs < self.sensorThickness])) * 2 - 1)
            ys[zs < self.sensorThickness] += randomWalkStep_1D[zs < self.sensorThickness] * (np.random.randint(0, 2, len(ys[zs < self.sensorThickness])) * 2 - 1)
            zs[zs < self.sensorThickness] += randomWalkStep_1D[zs < self.sensorThickness] * (np.random.randint(0, 2, len(zs[zs < self.sensorThickness])) * 2 - 1)
            
            totalTime += self.tInterval
            arr_time.append(totalTime)
            arr_rms.append(np.std(xs))

            if len(xs) == 0:
                break
        return arr_rms, arr_time, xs, arr_Ez, arr_E_rep_z

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
            # for x in result[2]:
            #     h1.Fill(x)
            def ggd(x, par):
                beta = par[0]
                alpha = par[1]
                coef = beta / (2 * alpha * TMath.Gamma(1 / beta))
                return coef * TMath.Exp(-TMath.Power((abs(x[0] - 0) / alpha), beta))
            f1 = TF1('f1', ggd, -5 * _rms, 5 * _rms, 2) ### par[0]: beta, par[1]: alpha
            f1.SetParLimits(0, 2, 5)
            f1.SetParLimits(1, 1.5, 20)
            if len(ggdParList) == 0:
                f1.SetParameters(2, 10)
            else:
                f1.SetParameters(0, ggdParList[-1][0])
                f1.SetParameters(1, ggdParList[-1][1])
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

        print(f'weighted RMS = {np.sqrt(np.sum(np.array(rmsList)**2*self.pdfList))}')
        return rmsList, self.pdfList, ggdParList