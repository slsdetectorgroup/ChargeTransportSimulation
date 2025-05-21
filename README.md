## Charge Carrier Transport Simulator

This project implements two Monte Carlo simulation methods to model charge carrier transport in silicon sensors, incorporating the initial charge distribution, drift, charge diffusion, and especially charge repulsion.
The simulator receives detector configurations and X-ray energies.

### Required libraries

ROOT: for histogramming, fitting, and visualization.

numpy: for numerical operations.

torch: for GPU acceleration.
 
### Example

See the Example.ipynb notebook for details.
The histograms from measurements using MÖNCH detector are stored in the Measurements.root file, and decoded in the Example.ipynb notebook.
Measurement data were taken at METROLOGIE beamline, SOLEIL, France using MÖNCH detectors collecting low-flux and monochromatic X-rays at different energies.
 
### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
