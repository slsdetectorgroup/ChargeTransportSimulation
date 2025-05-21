energyDict = { ### https://xdb.lbl.gov/Section1/Table_1-2.pdf
    'Sn':25.271, 'Sn_alpha2': 25.044, 'Sn_beta':28.486,
    'Ag':22.1629, 'Ag_alpha2': 21.9903, 'Ag_beta':24.9424,
    'Mo':17.479, 'Mo_alpha2':17.374, 'Mo_beta':19.608,
    'Zr':15.7751,  'Zr_alpha2':15.6909, 'Zr_beta':17.668,
    'Se':11.222, 'Se_beta':12.496,
    'Ge': 9.886, 'Ge_beta':10.982,
    'Cu': 8.0478, 'Cu_alpha2':  8.0278, 'Cu_beta':8.9053,
    'Fe': 6.404, 'Fe_beta':7.079,
    'Ti': 4.51,
    '5keV': 5, '10keV': 10,'15keV': 15, '20keV':20, '25keV':25,
    '1.8keV': 1.9, '1.9keV': 1.9, 
}
for key in energyDict:
    energyDict[key] *= 1000 ### eV
attenuationLengthDict = { ### https://henke.lbl.gov/optical_constants/atten2.html
    'Fe':36.5417, 'Fe_beta':48.8063,
    'Cu':70.8363, 'Cu_alpha2':70.3350, 'Cu_beta':95.1905, 
    'Ge':129.280, 'Ge_beta':257.804,
    'Se':187.700, 'Se_beta':257.804, 
    'Zr':514.376, 'Zr_alpha2':505.814, 'Zr_beta':719.087,
    'Mo':696.533, 'Mo_alpha2':684.201, 'Mo_beta':978.831,
    'Ag':1404.03, 'Ag_alpha2':1372.19, 'Ag_beta':1978.21, 
    'Sn':2054.03, 'Sn_alpha2':2001.54, 'Sn_beta':2881.53,
    '5keV': 18.0101, '10keV':133.706, '15keV':442.607, '20keV':1037.80, '25keV':1991.45,
    '1.8keV': 13.4951, '1.9keV': 1.36995,
}
kb1ka1RatioDict = {'Fe':0.1209, 'Cu':0.1216, 'Zr':0.1536, 'Mo':0.1586, 'Ag':0.2502, 'Sn':0.1751} ###  https://www.sciencedirect.com/science/article/pii/S0092640X74800197?via%3Dihub
ka2ka1RatioDict = {'Fe':0.511,  'Cu':0.513,  'Zr':0.523, 'Mo':0.525, 'Ag':0.531,  'Sn':0.535}  ###  https://www.sciencedirect.com/science/article/pii/S0092640X74800197?via%3Dihub