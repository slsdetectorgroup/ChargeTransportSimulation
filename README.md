## Charge Carrier Transport Simulator

This project implements a Monte Carlo simulation to model charge carrier transport in silicon sensors, incorporating  initial charge distribution, drift, charge diffusion, and especially charge repulsion.
The simulator receives detector configurations and X-ray energies.

### Required libraries

ROOT: for histogramming, fitting, and visualization.

numpy: for numerical operations.

torch: for GPU acceleration.
 
### Example

See the Example.ipynb notebook for a demonstration of the simulator.
The histograms from measurements using MÖNCH detector are provided in the Measurements.root file, and decoded in the Example.ipynb notebook.

 
### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
