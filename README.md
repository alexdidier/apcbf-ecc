# Approximate Predictive Control Barrier Functions - A computational cheap and permissive safety filter
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber, Kim P. Wabersich, Melanie N. Zeilinger

https://doi.org/10.3929/ethz-b-000609876

*Accepted at ECC 2023*

Contact:
Robin C. Jacobs (<jacobsr@ethz.ch>)
Alexandre Didier (<adidier@ethz.ch>)

## Data files
Please download the data points used for training under https://doi.org/10.3929/ethz-b-000610171


## Overview


    ├── apcbf                   # Source files (safety filter, neural networks etc.)
    ├── data                    # Generated data points used for traing
    ├── notebooks               # Notebooks for simulation, training and visualization
    ├── models                  # Trained Neural Networks
    ├── params                  # Computed Parameters (terminal CBFs, etc.)
    ├── scripts                 # Scripts for generating data points and time measurements
    ├── LICENSE
    └── README.md

## Getting started

### Requirements
- Tested using Python version 3.8.1 on Ubuntu 20.04 LTS
- For some optimization the MOSEK solver is used which requires a valid license and installation via pip. If unavailable, other solvers can be used.
- To run the jupyter notebooks `jupyter lab` should be installed

### Installation Instructions
Clone the repository to your local machine. 
Install the required python libraries by running the following commands in the root directory of the repository: 

```sh
pip install -r requirements.txt
```

```sh
pip install -e .
```


### Notebooks

In `notebooks` there are multiple jupyter notebooks illustrating our proposed safety filter in simulation. In addition we also provide simulation results for the PCBF-SF algorithm in simulation.

### Generate PCBF data points used for training
The data points for the linear system are generated in the same notebook which is used for training the neural network.

For generating data points for the nonlinear system run

```sh
python -m  scripts.nonlinear_data_gen
```

### Training the neural network

The training of the neural network can be done in the corresponding jupyter notebooks `{Linear, Nonlinear}_Learning_PCBF.ipynb`


### Measure solve times

For the linear system
```sh
 python -m  scripts.time_measurement_apcbf_linear 
 ```

For the nonlinear system
```sh
python -m scripts.time_measurement_apcbf_nonlinear
 ```

To compute the solve time of the PCBF-SF implementation run
```sh
python -m scripts.time_measurement_pcbf_nonlinear
 ```

### Overview of trained NN approximators

| Name                                 | System  | Architecture |
|--------------------------------------|---------|--------------|
|   model_lup_Aplus_06_04_18_46        |linear   | 2-128-64-1   |
|   model_nlup_NNplus_100_02_06_11_35  |nonlinear| 4-64-64-64-1 |
|   model_l1_{1,2,3,4}                 |nonlinear| 4-64-64-64-1 |
|   model_nlup_NNplus_100_04_06_13_54  |nonlinear| 4-64-64-1    |


## License

BSD 2-Clause. See `LICENSE` file 
