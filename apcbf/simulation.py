""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module contains methods for simulating systems of type DiscreteDynamics and ContinuousDynamics"""
from typing import Type
import numpy as np
from scipy.integrate import solve_ivp
from apcbf.dynamic import *
from apcbf.controller import *
import matplotlib.pyplot as plt


def simulate_discrete(init_state, discrete_dynamics: Type[DiscreteDynamics], Nsteps: int = 10,
                      controller_object: Type[Controller] = None, verbose=False):
    """ Simulates the discrete time dynamical system for `Nsteps`

        If no `controller_object` is specified, a zero input will be applied for all times
    """
    x_traj = np.zeros(shape=(Nsteps + 1, discrete_dynamics.state_dim))
    u_traj = np.zeros(shape=(Nsteps, discrete_dynamics.input_dim))
    x_traj[0, :] = init_state

    if controller_object is None:
        print("Using zero input")
        controller_object = LinearController(
            np.zeros(shape=(discrete_dynamics.input_dim, discrete_dynamics.state_dim)))

    # Simulate step by step
    for i in range(Nsteps):
        if verbose:
            print(f'Step {i} of {Nsteps} at state : {x_traj[i, :]}')
        u_traj[i, :] = controller_object.input(x_traj[i, :])
        x_traj[i + 1, :] = discrete_dynamics.step(
            x_traj[i, :].reshape(-1, 1), u_traj[i, :]).flatten()
    return x_traj, u_traj


def simulate_continuous(init_state, continuous_dynamics: Type[ContinuousDynamics], Nsteps: int = 10, Tsample=0.1,
                        controller_object: Type[Controller] = None, verbose=False):
    """ Simulate the continuous time dynamical system for `Nsteps` using a step size of `Tsample`

        If no `controller_object` is specified, a zero input will be applied for all times
    """
    # ZOH - controller simulation
    x_traj = np.zeros(shape=(Nsteps + 1, continuous_dynamics.state_dim))
    u_traj = np.zeros(shape=(Nsteps, continuous_dynamics.input_dim))
    time = Tsample * np.arange(Nsteps + 1)
    x_traj[0, :] = init_state.flatten()

    if controller_object is None:
        print("Using zero input (with ZOH)")
        controller_object = LinearController(
            np.zeros(shape=(continuous_dynamics.input_dim, continuous_dynamics.state_dim)))

    for i in range(Nsteps):
        if verbose:
            print(
                f'Step {i} at t={i * Tsample:.3f} of {Nsteps} steps at state : {x_traj[i, :]}')
        u_traj[i, :] = controller_object.input(
            x_traj[i, :].reshape(-1, 1)).flatten()

        y0 = np.concatenate((x_traj[i, :], u_traj[i, :]))
        sol = solve_ivp(continuous_dynamics.xdot_extended,
                        t_span=(i * Tsample, (i + 1) * Tsample), y0=y0)
        x_traj[i + 1, :] = sol.y[:continuous_dynamics.state_dim, -1].flatten()

    return x_traj, u_traj, time


if __name__ == "__main__":
    # Test function
    def test_function(x, u):
        x_next = x + 0.2 * u.reshape(2, 1)
        return x_next
    test_function.__str__ = 'test dynamics'
    K = -np.eye(2)
    Nsteps = 50
    ctrl = LinearController(K)
    non_lin_sys = NonLinearDiscreteDynamics(test_function, 2, 2)
    print(non_lin_sys)

    x0 = np.array([1, .3])
    x, u = simulate_discrete(
        x0, non_lin_sys, Nsteps=Nsteps, controller_object=ctrl)

    plt.plot(np.arange(Nsteps + 1), x[:, 0], label="x1")
    plt.plot(np.arange(Nsteps + 1), x[:, 1], label="x2")
    plt.legend()
    plt.show()
