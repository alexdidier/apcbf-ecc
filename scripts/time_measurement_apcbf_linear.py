""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module measures the solve time for the approximate PCBF for the linear system"""
import pickle
from pytope import Polytope

from apcbf.approximate_pcbf_linear import *
from apcbf.controller import *
from apcbf.dynamic import LinearDiscreteDynamics
from apcbf.nn_model_types import *

TORCH_MODEL_NAME = 'models/model_lup_Aplus_06_04_18_46.pt'
model_type_name = LinModelType.APLUS

LOG_NAME = TORCH_MODEL_NAME[13:-5]
print(LOG_NAME)


@np.vectorize
def measure_times(x, y):
    """ Function which solves for the next input using one of the approximate approaches specified
    in the variable lerned_controller_time_test
    """
    try:
        u_test = learned_controller_time_test.input(
            np.array([x, y]).reshape(2, 1))
    except:
        num_errors += 1

    if isinstance(learned_controller_time_test, APCBFSafetyFilterKINFDEC):
        total_time = learned_controller_time_test.solver.stats()[
            time_str]
    else:
        total_time = learned_controller_time_test.solver1.stats()[time_str] + \
            learned_controller_time_test.solver2.stats()[time_str]

    return total_time


if __name__ == "__main__":
    model = torch.load(TORCH_MODEL_NAME)

    # A = np.mat([[-1, 0.5],[-1, 0.5]]) #stable
    A = np.mat([[1.05, 1], [0, 1]])  # unstable, from paper
    # B = np.mat([[1,],[0,]])
    B = np.mat([[1, ], [0.5, ]])  # from paper

    lin_sys = LinearDiscreteDynamics(A, B)
    sys = lin_sys
    print(sys)

    Hx = np.kron(np.eye(A.shape[0]), np.array([[1], [-1]]))
    hx = np.ones((2 * A.shape[0], 1))
    XConstraints = Polytope(Hx, hx)
    Hu = np.array([[1], [-1]])
    hu = np.ones((B.shape[0], 1))
    UConstraints = Polytope(Hu, hu)

    N = 20
    c = 0.001
    mu_x = np.sqrt(0.001)
    mu_u = np.sqrt(0.001)
    alpha_f = 1000
    def delta_i(i): return i * 0.005  # as done in example

    # load parameters values
    params_dict = pickle.load(open("params/lin_termset_params.p", "rb"))
    params_dict

    K = 10 * np.ones((1, 2))
    bad_ctrl = LinearController(K)

    # Choose approach to test
    learned_controller_time_test = APCBFSafetyFilterKINFDEC(model, model_type_name, sys, params_dict=params_dict,
                                                            performance_controller=bad_ctrl,
                                                            verbose=True)
    time_str = 't_proc_total'
    num_errors = 0

    n_points_time = 100
    x1 = np.linspace(-2, 2, n_points_time)
    x2 = np.linspace(-2, 2, n_points_time)
    xx1, xx2 = np.meshgrid(x1, x2)

    times = measure_times(xx1, xx2)

    # Print time statistics
    print(times)
    print(f'model type : {model_type_name}')
    print(f'torch name : {model}')
    print(f'number_of_points : {n_points_time * n_points_time}')
    print(f'max solve time ({time_str}) : {times.max()}')
    print(f'min solve time ({time_str}) : {times.min()}')
    print(f'mean solve time ({time_str}) : {times.mean()}')
    print(f'std dev solve time ({time_str}) : {times.std()}')
    print(f'------')
    print(f'->num errors  : {num_errors}')
