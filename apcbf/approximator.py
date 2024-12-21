""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module contains the neural network architectures used to approximate the CBF
 for the linear system"""
from torch import nn


class HpbApproximator(nn.Module):
    def __init__(self):
        super(HpbApproximator, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(2, 128)
        self.second_lin = nn.Linear(128, 64)
        self.third_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)
        x = self.second_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)
        output = self.third_lin(x)
        # output = nn.functional.softplus(output)
        # output = self.linear_relu_stack(x)
        return output


class HpbApproximatorAplus(nn.Module):
    def __init__(self):
        super(HpbApproximatorAplus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(2, 128)
        self.second_lin = nn.Linear(128, 64)
        self.third_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)
        x = self.second_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)
        output = self.third_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        # output = self.linear_relu_stack(x)
        return output


class HpbApproximatorASigmoid(nn.Module):
    def __init__(self):
        super(HpbApproximatorASigmoid, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(2, 128)
        self.second_lin = nn.Linear(128, 64)
        self.third_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.sigmoid(x)  # TODO change to torch.sigmoid()
        x = self.second_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.sigmoid(x)
        output = self.third_lin(x)
        # output = nn.functional.sigmoid(output)
        # output = self.linear_relu_stack(x)
        return output


class HpbApproximatorB(nn.Module):
    def __init__(self):
        super(HpbApproximatorB, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(2, 64)
        self.second_lin = nn.Linear(64, 128)
        self.third_lin = nn.Linear(128, 64)
        self.fourth_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        output = self.fourth_lin(x)

        # output = nn.functional.softplus(output)
        # output = self.linear_relu_stack(x)
        return output


class HpbApproximatorBplus(nn.Module):
    def __init__(self):
        super(HpbApproximatorBplus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(2, 64)
        self.second_lin = nn.Linear(64, 128)
        self.third_lin = nn.Linear(128, 64)
        self.fourth_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        output = self.fourth_lin(x)

        output = nn.functional.softplus(output)
        # output = self.linear_relu_stack(x)
        return output


class HpbApproximatorCplus(nn.Module):
    def __init__(self):
        super(HpbApproximatorCplus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(2, 128)
        self.second_lin = nn.Linear(128, 256)
        self.third_lin = nn.Linear(256, 64)
        self.fourth_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        # x = nn.functional.relu(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        output = self.fourth_lin(x)

        output = nn.functional.softplus(output)
        # output = self.linear_relu_stack(x)
        return output
