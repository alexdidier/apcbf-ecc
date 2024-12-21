""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module runs the geometric random sampler for the nonlinear system using multiple processes"""
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from apcbf.pcbf_geometric_sampler import *
from apcbf.pcbf_nonlinear import *
from apcbf.custom_dataset import *
from apcbf.dynamic import *

# Parameter values to be set
NUM_WORKERS = 8
JOBS_PER_WORKER = 16  # 1024
H_THRESHOLD = 100
NUM_TRAINING_GOAL_PTS = 2 ** 10  # With current stepsize generates roughly 400x more pts
NUM_BUFFER_GOAL_PTS = 2 ** 1
NUM_TEST_GOAL_PTS = 2 ** 1

# System definition
vs = 5
L = 5.0
T_disc = 0.05
T_s = T_disc


def nonlin_xdot(t, x, u):  # System definition
    target_vel = vs
    x_dot = np.zeros_like(x)
    x_dot[0] = x[3] * np.sin(x[1])
    x_dot[1] = ((target_vel + x[3]) / L) * np.tan(x[2])
    x_dot[2] = u[0]
    x_dot[3] = u[1]
    return x_dot


def pcbf_test(x, init_values=None):
    """ Checks whether the h_PB is below H_THRESHOLD for state x"""
    try:
        if init_values is not None:
            pcbf.opti.set_initial(pcbf.gsi, init_values[0])
            pcbf.opti.set_initial(pcbf.gsi_N, init_values[1])
            pcbf.opti.set_initial(pcbf.u, init_values[2])
            pcbf.opti.set_initial(pcbf.x, init_values[3])
        else:
            gsi_temp = 1 * np.ones((pcbf.numb_x_constr, pcbf.N))
            pcbf.opti.set_initial(pcbf.gsi, gsi_temp)

        xi, xi_n, h, u, x = pcbf.solve(np.array(x))
        new_init_values = [xi, xi_n, u, x]
    except:
        return False, 0, None
    h = np.abs(h)
    # print(h)
    if h < H_THRESHOLD:
        return True, h, new_init_values
    return False, h, None





if __name__=="__main__":
    non_lin_cont = NonLinearContinuousDynamics(nonlin_xdot, 4, 2)

    constraint_dict = pickle.load(open("params/non_linear_constraints_params.p", 'rb'))
    X = constraint_dict['X']
    U = constraint_dict['U']

    params_dict = pickle.load(open("params/non_linear_termset_params.p", "rb"))
    delta_i = lambda i: i * 0.004  # as done in example
    pcbf = SlackOpt(non_lin_cont, X, U, delta_i, params_dict, N=50, verbose=False)

    print('Test if pcbf works...')
    x0 = np.array([1.1, 1, 0, 0])
    gsi, gsi_N, opt_value, _, _ = pcbf.solve(x0)
    print('ok')

    # Initalize the PCBF optimization procedure used for the test function
    pcbf = SlackOpt(non_lin_cont, X, U, delta_i, params_dict, N=50, use_warm_start=True, verbose=False)




    # Initialize geometric random walk sampler
    datagen = DataSetGeneratorParallelized(Xconstraint=X, Npoints_tr=NUM_TRAINING_GOAL_PTS, Npoints_bf=NUM_BUFFER_GOAL_PTS,
                                        Npoints_ts=NUM_TEST_GOAL_PTS, set_test_function=pcbf_test,
                                        jobs_per_worker=JOBS_PER_WORKER, num_workers=NUM_WORKERS)

    # Start sampling of data points using initial seed
    D_tr, D_ts = datagen.generate_data([1., 0., 0., 0, 0])

    print("Size of training dataset : ", len(D_tr))
    print("Size of test dataset :", len(D_ts))

    data_train_np = np.array(D_tr)
    data_test_np = np.array(D_ts)

    x_train_np = data_train_np[:, :4]
    y_train_np = data_train_np[:, 4]
    x_test_np = data_test_np[:, :4]
    y_test_np = data_test_np[:, 4]

    # Save data
    # Path("data").mkdir(parents=True, exist_ok=True)

    train_data_set = SimpleData(x_train_np, y_train_np)
    torch.save(train_data_set,
            f'data/train_data_geom_sampl_thresh_{H_THRESHOLD}_{len(train_data_set) / 1e3}k_nonlinear_paper.pt')

    val_data_set = SimpleData(x_test_np, y_test_np)
    torch.save(val_data_set, f'data/val_data_geom_sampl_thresh_{H_THRESHOLD}_{len(val_data_set) / 1e3}k_nonlinear_paper.pt')
