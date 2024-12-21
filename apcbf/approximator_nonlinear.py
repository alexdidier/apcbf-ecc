""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module contains the neural network architectures used to approximate the CBF for the nonlinear system"""
from torch import nn


class HpbApproximatorNonLinAplus(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinAplus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 128)
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


class HpbApproximatorNonLinAplusDropout(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinAplusDropout, self).__init__()
        self.first_lin = nn.Linear(4, 128)
        self.second_lin = nn.Linear(128, 64)
        self.third_lin = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.first_lin(x)
        x = self.dropout(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = self.dropout(x)
        x = nn.functional.softplus(x)

        output = self.third_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative

        return output


class HpbApproximatorNonLinCplus(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinCplus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 128)
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


class HpbApproximatorNonLinDplus(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinDplus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 128)
        self.second_lin = nn.Linear(128, 256)
        self.third_lin = nn.Linear(256, 256)
        self.fourth_lin = nn.Linear(256, 64)
        self.fifth_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        x = self.fourth_lin(x)
        x = nn.functional.softplus(x)

        output = self.fifth_lin(x)

        output = nn.functional.softplus(output)
        # output = self.linear_relu_stack(x)
        return output


class HpbApproximatorNonLinEplus(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinEplus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 128)
        self.second_lin = nn.Linear(128, 128)
        self.third_lin = nn.Linear(128, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        output = self.third_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output


class HpbApproximatorNonLinEplusDropout(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinEplusDropout, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 128)
        self.second_lin = nn.Linear(128, 128)
        self.third_lin = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.first_lin(x)
        x = self.dropout(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = self.dropout(x)
        x = nn.functional.softplus(x)

        output = self.third_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output


class HpbApproximatorNonLinGplus(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinGplus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 64)
        self.second_lin = nn.Linear(64, 64)
        self.third_lin = nn.Linear(64, 64)
        self.fourth_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        output = self.fourth_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output


class HpbApproximatorNonLinG32plus(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinG32plus, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 32)
        self.second_lin = nn.Linear(32, 32)
        self.third_lin = nn.Linear(32, 32)
        self.fourth_lin = nn.Linear(32, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        output = self.fourth_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output


class HpbApproximatorNonLinGplusDropout(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinGplusDropout, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 64)
        self.second_lin = nn.Linear(64, 64)
        self.third_lin = nn.Linear(64, 64)
        self.fourth_lin = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.first_lin(x)
        x = self.dropout(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = self.dropout(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = self.dropout(x)
        x = nn.functional.softplus(x)

        output = self.fourth_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output


class HpbApproximatorNonLinNplus_4x64(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinNplus_4x64, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 64)
        self.second_lin = nn.Linear(64, 64)
        self.third_lin = nn.Linear(64, 64)
        self.fourth_lin = nn.Linear(64, 64)
        self.final_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        x = self.fourth_lin(x)
        x = nn.functional.softplus(x)

        output = self.final_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output


class HpbApproximatorNonLinNplus_2x128(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinNplus_2x128, self).__init__()
        self.first_lin = nn.Linear(4, 128)
        self.second_lin = nn.Linear(128, 128)
        self.third_lin = nn.Linear(128, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        output = self.third_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output


class HpbApproximatorNonLinNplus_2x64(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinNplus_2x64, self).__init__()
        self.first_lin = nn.Linear(4, 64)
        self.second_lin = nn.Linear(64, 64)
        self.third_lin = nn.Linear(64, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        output = self.third_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output


class HpbApproximatorNonLinNplus_3x128(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinNplus_3x128, self).__init__()
        self.first_lin = nn.Linear(4, 128)
        self.second_lin = nn.Linear(128, 128)
        self.third_lin = nn.Linear(128, 128)
        self.final_lin = nn.Linear(128, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        output = self.final_lin(x)
        output = nn.functional.softplus(output)

        return output


class HpbApproximatorNonLinNplus_2x256(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinNplus_2x256, self).__init__()
        self.first_lin = nn.Linear(4, 256)
        self.second_lin = nn.Linear(256, 256)
        self.final_lin = nn.Linear(256, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        output = self.final_lin(x)
        output = nn.functional.softplus(output)

        return output


class HpbApproximatorNonLinNplus_4x128(nn.Module):
    def __init__(self):
        super(HpbApproximatorNonLinNplus_4x128, self).__init__()
        # self.linear_relu_stack = nn.Sequential(
        self.first_lin = nn.Linear(4, 128)
        self.second_lin = nn.Linear(128, 128)
        self.third_lin = nn.Linear(128, 128)
        self.fourth_lin = nn.Linear(128, 128)
        self.final_lin = nn.Linear(128, 1)

    def forward(self, x):
        x = self.first_lin(x)
        x = nn.functional.softplus(x)

        x = self.second_lin(x)
        x = nn.functional.softplus(x)

        x = self.third_lin(x)
        x = nn.functional.softplus(x)

        x = self.fourth_lin(x)
        x = nn.functional.softplus(x)

        output = self.final_lin(x)
        output = nn.functional.softplus(output)  # Use softplus for forcing function to be nonnegative
        return output
