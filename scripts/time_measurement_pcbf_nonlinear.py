""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module measures the solve time for the original PCBF-SF procedure (Algorithm 1)"""
import pickle
from tqdm.notebook import tqdm
import torch

from apcbf.approximator import *
from apcbf.pcbf_nonlinear import *
from apcbf.controller import *
from apcbf.non_lin_sys import *


# Load System and parameter values
sys = non_lin_disc
print(sys)
N = 50
c = 0.001
mu_x = np.sqrt(0.001)
mu_u = np.sqrt(0.001)
alpha_f = 1000
def delta_i(i): return i * 0.005  # as done in example


# load parameters values
params_dict = pickle.load(open("params/non_linear_termset_params.p", "rb"))

# params_dict
constraint_dict = pickle.load(
    open("params/non_linear_constraints_params.p", 'rb'))
X = constraint_dict['X']
U = constraint_dict['U']

K = 10 * np.ones((2, 4))
bad_ctrl = LinearController(K)
algo = Algorithm(non_lin_cont, bad_ctrl, X,
                 U, delta_i, params_dict, N=50, verbose=True)

N_POINTS = int(1e2)
# test_data = torch.load(
#     'data/val_data_geom_sampl_thresh_100_13.155k_nonlinear_paper.pt')
test_data = torch.load(
        'data/val_data_813k_nonlinear.pt')
test_data.X = test_data.X[torch.randperm(test_data.X.shape[0]),:]
test_data_sel_x = test_data.X[:N_POINTS, :]
# test_data_sel_y = test_data.y[:100]

time_str = 't_proc_total'
slack_opt_solve_time = np.zeros((len(test_data_sel_x)))
sf_opt_solve_time = np.zeros((len(test_data_sel_x)))
total_time_opt = np.zeros((len(test_data_sel_x)))

error_ind = np.ones_like(slack_opt_solve_time, dtype=bool)

num_errors = 0

# Iterate over test_data to measure time
for idx, x in tqdm(enumerate(test_data_sel_x)):
    try:
        algo.input(np.array(x).reshape(sys.state_dim, 1))
        slack_opt_solve_time[idx] = algo.slack_opt.sol.stats()[time_str]
        sf_opt_solve_time[idx] = algo.safety_filter.sol.stats()[time_str]
        total_time_opt[idx] = slack_opt_solve_time[idx] + \
            sf_opt_solve_time[idx]
    except:
        print('error')
        num_errors += 1
        error_ind[idx] = 0
    # print(algo.slack_opt.sol.stats())
    algo.reset()

total_solve_time = [sum(x)
                    for x in zip(slack_opt_solve_time, sf_opt_solve_time)]
time_split_slack_opt = [x[0] / (x[0] + x[1])
                        for x in zip(slack_opt_solve_time, sf_opt_solve_time)]

slack_opt_solve_time_np = np.array(slack_opt_solve_time)
sf_opt_solve_time_np = np.array(sf_opt_solve_time)
total_solve_time_np = np.array(total_solve_time)
time_split_slack_opt = np.array(time_split_slack_opt)

print(f'Number of errors : {num_errors}')

if num_errors == 0:
    print(f'Solver : IPOPT')
    print(f'Number of points : {len(test_data_sel_x)}')
    print(f'Max solve time : {total_solve_time_np.max()}')
    print(f'Min solve time : {total_solve_time_np.min()}')
    print(f'Mean solve time : {total_solve_time_np.mean()}')
    print(f'Std dev. {total_solve_time_np.std()}')

    print(
        f'Mean percentage of total time spent in slack opt. {time_split_slack_opt.mean()} %')

if num_errors > 0:  # If have experienced solver errors, then ignore the solve time at that point for computing statistics
    print('STATISTICS EXCLUDING ERRORS')
    slack_opt_solve_time_np = slack_opt_solve_time_np[error_ind]
    sf_opt_solve_time_np = sf_opt_solve_time_np[error_ind]
    total_solve_time_np = total_solve_time_np[error_ind]
    time_split_slack_opt = time_split_slack_opt[error_ind]

    print(f'Solver : IPOPT')
    print(f'Number of points : {len(slack_opt_solve_time_np)}')
    print(f'Max solve time : {total_solve_time_np.max()}')
    print(f'Min solve time : {total_solve_time_np.min()}')
    print(f'Mean solve time : {total_solve_time_np.mean()}')
    print(f'Std dev. {total_solve_time_np.std()}')

    print(
        f'Mean percentage of total time spent in slack opt. {time_split_slack_opt.mean()} %')
