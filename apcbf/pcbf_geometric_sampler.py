""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

import copy
import itertools
import pickle
import os
import random
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from apcbf.pcbf_nonlinear import *
from apcbf.custom_dataset import *
from apcbf.dynamic import *


class DataSetGeneratorParallelized:
    """ This class implements the geometric random walk sampler introduced in Chen et al. 2022 """

    def __init__(self, Xconstraint, Npoints_tr, Npoints_bf, Npoints_ts, set_test_function, sample_scaling=1.5,
                 step_size=0.01, num_workers=6, jobs_per_worker=8, verbose=True):
        self.verbose = verbose
        self.num_workers = num_workers
        self.jobs_more_than_workers = jobs_per_worker
        self.jobs_btw_checkpoints = 1000
        self.d = Xconstraint.A.shape[1]
        self.sampler = qmc.Sobol(d=self.d, scramble=True)
        self.X = Xconstraint
        self.step_size = step_size
        self.l_bound = []
        self.u_bound = []
        self.Npoints_log2_tr = int(np.log2(Npoints_tr))
        self.Npoints_log2_bf = int(np.log2(Npoints_bf))
        self.Npoints_log2_ts = int(np.log2(Npoints_ts))
        self.set_test_function = set_test_function

        self.check_pt_path = "data/temp"
        try:
            if not os.path.exists(self.check_pt_path):
                os.makedirs(self.check_pt_path)
        except OSError:
            print(f"Error: Creating Directory: {self.check_pt_path}")

        print(
            f'Generate number of points (2^): {self.Npoints_log2_tr}, {self.Npoints_log2_bf}, {self.Npoints_log2_ts} ')

        # Bring state constraints into sobel sampler compatible form
        for l in range(Xconstraint.A.shape[0]):
            if l % 2 == 0:  # upper bound
                self.u_bound.append(Xconstraint.b[l][0] * sample_scaling)
            else:
                self.l_bound.append(-Xconstraint.b[l][0] * sample_scaling)

        print(f'Bounds (scaling :{sample_scaling}) : {self.l_bound}, {self.u_bound}')

    def _check_if_valid(self, x, opt_values=None):
        return self.set_test_function(x, opt_values)

    def _random_walk(self, G, Sin):
        D = []
        S = copy.copy(Sin)

        # for x in tqdm(G) :
        for x in G:
            # Sample seed tuple
            # seed = np.random.choice(S)
            # print(S)
            seed = random.sample(S, 1)[0]
            D_prime = self._line_solve(seed, x)
            # D.append(D_prime)
            if len(D_prime) > 0:
                D += D_prime
                S.append(D_prime[-1].tolist())

        return S, D

    def _train_random_walk(self, G):
        # print('start job')
        D = []
        Sin = self.seed
        S = copy.copy(Sin)

        for x in G:
            seed = random.sample(S, 1)[0]
            D_prime = self._line_solve(seed, x)
            # D.append(D_prime)
            if len(D_prime) > 0:
                D += D_prime
                S.append(D_prime[-1].tolist())

        return S, D

    def _wrapper(self, args):
        idx, split = args
        return (idx, self._train_random_walk(split))

    def _random_walk_parallel(self, G, Sin):
        self.seed = Sin
        split = np.array_split(G, self.num_workers * self.jobs_more_than_workers)
        print("Number of jobs : ", len(split))
        print("Number of goal points per job : ", len(split[0]))

        now = time.time()
        # Initialize multiprocessing pool
        pool = Pool(self.num_workers)

        s_sol = []
        d_sol = []
        for idx, result in tqdm(pool.imap_unordered(self._wrapper, enumerate(split)), total=len(split)):
            s_sol.append(result[0])
            d_sol.append(result[1])
            if idx % self.jobs_btw_checkpoints == 0:
                # print(f'Saving checkpoint at idx {idx}')
                s_sol_np = np.copy(np.array(s_sol))
                d_sol_np = np.copy(np.array(d_sol))
                np.save(f'{self.check_pt_path}/checkpt_s_sol_{len(s_sol_np)}.npy', s_sol_np)
                np.save(f'{self.check_pt_path}/checkpt_d_sol_{len(d_sol_np)}.npy', d_sol_np)
        #         with Pool(NUM_WORKERS) as pool :
        #             y_sol = pool.map(self._train_random_walk, split)

        then = time.time()
        print(f' Took : {then - now} seconds')

        s_res = list(itertools.chain.from_iterable(s_sol))
        d_res = list(itertools.chain.from_iterable(d_sol))
        # print(y_res)
        # print(s_sol)

        return s_res, d_res

    def _line_solve(self, seed, x):  # s/seed = [x,h_pb] i.e. 5-d
        n = np.linalg.norm(x - seed[:self.d])
        xn = (x - seed[:self.d]) / n
        D = []
        opt_values = None
        for i in range(int(np.ceil(n / self.step_size))):
            xi = seed[:self.d] + i * self.step_size * xn
            # print(xi)
            success, h, opt_values = self._check_if_valid(xi, opt_values)
            if success:  # accept data point
                h = np.array([h])
                si = np.concatenate((xi, h))
                D.append(si)
            else:
                break
        return D

    def generate_data(self, seed):
        """ Generates the data sets (train, buffer, test) which were specified during construction of the class"""
        if self.verbose:
            last = time.time()

        self.seed = [seed]
        # generate goal points for training data set
        tr_sobel = self.sampler.random_base2(self.Npoints_log2_tr)
        tr_sobel = qmc.scale(tr_sobel, self.l_bound, self.u_bound)
        # print(tr_sobel.max())
        # generate goal points for buffer data set
        bf_sobel = self.sampler.random(2 ** self.Npoints_log2_bf)
        bf_sobel = qmc.scale(bf_sobel, self.l_bound, self.u_bound)
        # generate goal points for test data set
        ts_sobel = self.sampler.random(2 ** self.Npoints_log2_ts)
        ts_sobel = qmc.scale(ts_sobel, self.l_bound, self.u_bound)

        print("Generating training data")
        S_tr, D_tr = self._random_walk_parallel(tr_sobel, [seed])

        print("Saving checkpoint training data")
        d_tr_np_temp = np.array(D_tr)
        s_tr_np_temp = np.array(S_tr)
        np.save(f'{self.check_pt_path}/d_tr_temp_{len(d_tr_np_temp)}.npy', d_tr_np_temp)
        np.save(f'{self.check_pt_path}/s_tr_temp_{len(s_tr_np_temp)}.npy', s_tr_np_temp)

        print("Generating buffer data")
        S_bf, D_bf = self._random_walk_parallel(bf_sobel, S_tr)

        print("Saving checkpoint buffer data")
        d_bf_np_temp = np.array(D_bf)
        s_bf_np_temp = np.array(S_bf)
        np.save(f'{self.check_pt_path}/d_bf_temp_{len(d_bf_np_temp)}.npy', d_bf_np_temp)
        np.save(f'{self.check_pt_path}/s_bf_temp_{len(s_bf_np_temp)}.npy', s_bf_np_temp)

        print("Generating test data")
        S_bf_tuple = [tuple(element) for element in S_bf]
        S_tr_tuple = [tuple(element) for element in S_tr]

        S_complement_tuple = list(set(S_bf_tuple) - set(S_tr_tuple))
        S_complement_list = list(S_complement_tuple)
        S_complement_list = [list(element) for element in S_complement_list]

        S_tst, D_tst = self._random_walk_parallel(ts_sobel, S_complement_list)

        print("Saving checkpoint test data")
        d_ts_np_temp = np.array(D_tst)
        s_ts_np_temp = np.array(S_tst)
        np.save(f'{self.check_pt_path}/d_ts_temp_{len(d_ts_np_temp)}.npy', d_ts_np_temp)
        np.save(f'{self.check_pt_path}/s_ts_temp_{len(s_ts_np_temp)}.npy', s_ts_np_temp)

        training_data = D_tr
        test_data = D_tst

        if self.verbose:
            now = time.time()
            difference = now - last
            print(f'Total time : {difference} seconds')

        return training_data, test_data
