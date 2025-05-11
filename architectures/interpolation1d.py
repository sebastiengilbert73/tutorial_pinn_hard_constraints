import numpy as np
import torch

class SumOfGaussians:
    def __init__(self, xy_list, sigma=0.1):
        self.sigma = sigma
        self.centers = [x for (x, y) in xy_list]
        number_of_centers = len(xy_list)
        A = np.zeros((number_of_centers, number_of_centers), dtype=float)
        b = np.zeros(number_of_centers)

        row = 0
        for x, y in xy_list:
            for col in range(number_of_centers):
                neighbor_x, _ = xy_list[col]
                A[row, col] = self.gaussian(x - neighbor_x)
            b[row] = y
            row += 1
        self.coefs = np.linalg.solve(A, b)  # (N_centers)

        self.coefs_tsr = torch.from_numpy(self.coefs)  # (N_centers)
        self.centers_tsr = torch.tensor(self.centers)  # (N_centers)

    def evaluate(self, x):
        sum = 0
        for mu_ndx in range(len(self.centers)):
            mu = self.centers[mu_ndx]
            sum += self.coefs[mu_ndx] * self.gaussian(x - mu)
        return sum

    def batch_evaluate(self, x_tsr):  # x_tsr.shape = (B, 1)
        x_extended_tsr = x_tsr.repeat((1, self.centers_tsr.shape[0]))  # (B, N_centers)
        x_minus_mu_tsr = x_extended_tsr - self.centers_tsr.unsqueeze(0)  # (B, N_centers)
        gaussians_tsr = self.batch_gaussian(x_minus_mu_tsr)  # (B, N_centers)
        weighted_gaussians_tsr = self.coefs_tsr.unsqueeze(0) * gaussians_tsr  # (B, N_centers)
        sum_tsr = torch.sum(weighted_gaussians_tsr, dim=1)
        return sum_tsr

    def gaussian(self, x):
        return np.exp(-x**2/(2 * self.sigma**2))

    def batch_gaussian(self, x_tsr):  # x_tsr.shape = (B, N)
        y = torch.exp(-x_tsr**2/(2 * self.sigma**2))  # (B, N)
        return y