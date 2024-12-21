""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module is used to measure the solve time for the approximate safety filters for the nonlinear system"""
import pickle

from apcbf.approximate_pcbf_nonlinear import *
from apcbf.controller import *
from apcbf.nn_model_types import *
from apcbf.non_lin_sys import *

# Specify NN model
TORCH_MODEL_NAME = 'models/model_nlup_NNplus_100_02_06_11_35.pt'
model_type_name = NonLinModelType.THREEHIDDENLAYERS
# TORCH_MODEL_NAME = 'models/model_nlup_NNplus_100_04_06_13_54.pt'
# model_type_name = NonLinModelType.TWOHIDDENLAYERS
use_log_space_model = True
use_plus_models = True


@np.vectorize
def measure_times(x, y, z, w):
    """ This method measures the solve time at state [x,y,z,w] when using the approach specified
    in learned_controller_time_test
    """
    u_test = learned_controller_time_test.input(
        np.array([x, y, z, w]).reshape(4, 1))
    if isinstance(learned_controller_time_test, APCBFSafetyFilterMAXDEC):
        total_time = learned_controller_time_test.solver1.stats()[time_str] + \
            learned_controller_time_test.solver2.stats()[time_str]
    elif isinstance(learned_controller_time_test, APCBFSafetyFilterKINFDEC):
        total_time = learned_controller_time_test.solver2.stats()[time_str]
    return total_time


if __name__ == "__main__":
    LOG_NAME = TORCH_MODEL_NAME[13:-5]
    print(LOG_NAME)

    # Load model
    model = torch.load(TORCH_MODEL_NAME, map_location=torch.device('cpu'))

    # Load system parameter
    sys = non_lin_disc
    print(sys)
    constraint_dict = pickle.load(
        open("params/non_linear_constraints_params.p", 'rb'))
    X = constraint_dict['X']
    U = constraint_dict['U']

    N = 50
    c = 0.001
    mu_x = np.sqrt(0.001)
    mu_u = np.sqrt(0.001)
    alpha_f = 1000
    def delta_i(i): return i * 0.005

    # load parameters values
    params_dict = pickle.load(open("params/non_linear_termset_params.p", "rb"))

    K = 10 * np.ones((2, 4))
    bad_ctrl = LinearController(K)

    # Choose approach to test
    learned_controller_time_test = APCBFSafetyFilterKINFDEC(model=model, m_type=model_type_name,
                                                            use_log_space_model=use_log_space_model,
                                                            use_plus_model=use_plus_models, sys=sys,
                                                            params_dict=params_dict,
                                                            input_constraints=U,
                                                            performance_controller=bad_ctrl,
                                                            verbose=True)

    time_str = 't_proc_total'

    N_POINTS = int(1e4)
    # test_data = torch.load(
    #     'data/val_data_geom_sampl_thresh_100_13.155k_nonlinear_paper.pt')
    test_data = torch.load(
        'data/val_data_813k_nonlinear.pt')
 
    test_data.X = test_data.X[torch.randperm(test_data.X.shape[0]),:]
    test_data_sel_x = test_data.X[:N_POINTS, :]

    times = measure_times(
        test_data_sel_x[:, 0], test_data_sel_x[:, 1], test_data_sel_x[:, 2], test_data_sel_x[:, 3])

    # print out time statistics
    print(times)
    print(f'model type : {model_type_name}')
    print(f'torch name : {model}')
    print(f'number_of_points : {N_POINTS}')
    print(f'max solve time ({time_str}) : {times.max()}')
    print(f'min solve time ({time_str}) : {times.min()}')
    print(f'mean solve time ({time_str}) : {times.mean()}')
    print(f'std dev solve time ({time_str}) : {times.std()}')
