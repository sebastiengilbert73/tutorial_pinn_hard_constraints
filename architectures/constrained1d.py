import torch
import numpy as np
import architectures.pinn1d as pinn1d
import architectures.interpolation1d as interpolation1d

class Bubble(torch.nn.Module):
    def __init__(self, x_left=0, x_right=1.0):
        super().__init__()
        self.x_left = x_left
        self.x_right = x_right

    def forward(self, x_tsr):  # x_tsr.shape = (B, 1)
        return (x_tsr - self.x_left) * (self.x_right - x_tsr)  # (B, 1)


class HardConstrained1dPinn(torch.nn.Module):
    def __init__(self, number_of_blocks, block_width, number_of_outputs,
                 initial_profile, time_bubble_beta=2.0):
        super().__init__()
        self.bubble = Bubble()
        self.z_predictor = pinn1d.Wang2020(
            number_of_inputs=2,
            number_of_blocks=number_of_blocks,
            block_width=block_width,
            number_of_outputs=number_of_outputs
        )
        self.initial_profile = initial_profile
        delta_x = 1.0/(len(self.initial_profile) - 1)
        xs = np.arange(0, 1 + delta_x/2, delta_x)
        xy_list = []
        for x_ndx in range(len(xs)):
            x = xs[x_ndx]
            xy_list.append((x, self.initial_profile[x_ndx]))
        #self.initial_profile_interpolator = interpolation1d.SumOfGaussians(xy_list, sigma_ratio=0.5)
        self.initial_profile_interpolator = interpolation1d.CubicSpline(
            xy_list, boundary_condition='2nd_derivative_0'
        )

        self.time_bubble_beta = torch.nn.Parameter(torch.tensor([time_bubble_beta]))

    def forward(self, x_t):  # x_t.shape = (B, 2)
        x_tsr = x_t[:, 0].unsqueeze(1)  # (B, 1)
        t_tsr = x_t[:, 1].unsqueeze(1)  # (B, 1)
        bubble_t_tsr = self.time_bubble(t_tsr)  # (B, 1)
        bubble_x_tsr = self.bubble(x_tsr)  # (B, 1)
        z_tsr = self.z_predictor(x_t)  # (B, 1)
        initial_interpolation_tsr = self.initial_profile_interpolator.batch_evaluate(x_tsr)
        return initial_interpolation_tsr + bubble_t_tsr * bubble_x_tsr * z_tsr

    def time_bubble(self, t_tsr):  # t_tsr.shape = (B, 1)
        return 1 - torch.exp(-self.time_bubble_beta * t_tsr)  # (B, 1)

class HardConstrained1dMLP(torch.nn.Module):
    def __init__(self, layer_widths, number_of_outputs,
                 initial_profile, time_bubble_beta=2.0):
        super().__init__()
        self.bubble = Bubble()
        self.z_predictor = pinn1d.MLP(
            number_of_inputs=2,
            layer_widths=layer_widths,
            number_of_outputs=number_of_outputs
        )
        self.initial_profile = initial_profile
        delta_x = 1.0/(len(self.initial_profile) - 1)
        xs = np.arange(0, 1 + delta_x/2, delta_x)
        xy_list = []
        for x_ndx in range(len(xs)):
            x = xs[x_ndx]
            xy_list.append((x, self.initial_profile[x_ndx]))
        self.initial_profile_interpolator = interpolation1d.CubicSpline(
            xy_list, boundary_condition='2nd_derivative_0'
        )

        self.time_bubble_beta = torch.nn.Parameter(torch.tensor([time_bubble_beta]))

    def forward(self, x_t):  # x_t.shape = (B, 2)
        x_tsr = x_t[:, 0].unsqueeze(1)  # (B, 1)
        t_tsr = x_t[:, 1].unsqueeze(1)  # (B, 1)
        bubble_t_tsr = self.time_bubble(t_tsr)  # (B, 1)
        bubble_x_tsr = self.bubble(x_tsr)  # (B, 1)
        z_tsr = self.z_predictor(x_t)  # (B, 1)
        initial_interpolation_tsr = self.initial_profile_interpolator.batch_evaluate(x_tsr)
        return initial_interpolation_tsr + bubble_t_tsr * bubble_x_tsr * z_tsr

    def time_bubble(self, t_tsr):  # t_tsr.shape = (B, 1)
        return 1 - torch.exp(-self.time_bubble_beta * t_tsr)  # (B, 1)

class HardConstrained1dResNet(torch.nn.Module):
    def __init__(self, number_of_blocks, block_width, number_of_outputs,
                 initial_profile, time_bubble_beta=2.0):
        super().__init__()
        self.bubble = Bubble()
        self.z_predictor = pinn1d.ResidualNet(
            number_of_inputs=2,
            number_of_blocks=number_of_blocks,
            block_width=block_width,
            number_of_outputs=number_of_outputs
        )
        self.initial_profile = initial_profile
        delta_x = 1.0/(len(self.initial_profile) - 1)
        xs = np.arange(0, 1 + delta_x/2, delta_x)
        xy_list = []
        for x_ndx in range(len(xs)):
            x = xs[x_ndx]
            xy_list.append((x, self.initial_profile[x_ndx]))
        self.initial_profile_interpolator = interpolation1d.CubicSpline(
            xy_list, boundary_condition='2nd_derivative_0'
        )

        self.time_bubble_beta = torch.nn.Parameter(torch.tensor([time_bubble_beta]))

    def forward(self, x_t):  # x_t.shape = (B, 2)
        x_tsr = x_t[:, 0].unsqueeze(1)  # (B, 1)
        t_tsr = x_t[:, 1].unsqueeze(1)  # (B, 1)
        bubble_t_tsr = self.time_bubble(t_tsr)  # (B, 1)
        bubble_x_tsr = self.bubble(x_tsr)  # (B, 1)
        z_tsr = self.z_predictor(x_t)  # (B, 1)
        initial_interpolation_tsr = self.initial_profile_interpolator.batch_evaluate(x_tsr)
        return initial_interpolation_tsr + bubble_t_tsr * bubble_x_tsr * z_tsr

    def time_bubble(self, t_tsr):  # t_tsr.shape = (B, 1)
        return 1 - torch.exp(-self.time_bubble_beta * t_tsr)  # (B, 1)