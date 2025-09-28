import numpy as np
import torch
import abc
import sys

class SumOfGaussians:
    def __init__(self, xy_list, sigma_ratio=1.0):
        if len(xy_list) < 2:
            raise ValueError(f"SumOfGaussians.__init__(): len(xy_list) ({len(xy_list)}) < 2")
        self.sorted_xy_list = sorted(xy_list, key=lambda x: x[0])
        self.centers = [x for (x, y) in self.sorted_xy_list]
        self.sigmas = self._sigmas(sigma_ratio)
        number_of_centers = len(self.centers)
        A = np.zeros((number_of_centers, number_of_centers), dtype=float)
        b = np.zeros(number_of_centers)

        row = 0
        for x_ndx in range(len(self.sorted_xy_list)):
            x = self.sorted_xy_list[x_ndx][0]
            y = self.sorted_xy_list[x_ndx][1]
            for col in range(number_of_centers):
                neighbor_x, _ = self.sorted_xy_list[col]
                neighbor_sigma = self.sigmas[col]
                A[row, col] = self.gaussian(x - neighbor_x, neighbor_sigma)
            b[row] = y
            row += 1
        self.coefs = np.linalg.solve(A, b)  # (N_centers)

        self.coefs_tsr = torch.from_numpy(self.coefs)  # (N_centers)
        self.centers_tsr = torch.tensor(self.centers)  # (N_centers)
        self.sigmas_tsr = torch.tensor(self.sigmas)  # (N_centers)

    def evaluate(self, x):
        sum = 0
        for mu_ndx in range(len(self.centers)):
            mu = self.centers[mu_ndx]
            sigma = self.sigmas[mu_ndx]
            sum += self.coefs[mu_ndx] * self.gaussian(x - mu, sigma)
        return sum

    def batch_evaluate(self, x_tsr):  # x_tsr.shape = (B, 1)
        x_extended_tsr = x_tsr.repeat((1, self.centers_tsr.shape[0]))  # (B, N_centers)
        gaussians_tsr = self.batch_gaussian(x_extended_tsr, self.centers_tsr.to(x_tsr.device), self.sigmas_tsr.to(x_tsr.device))  # (B, N_centers)
        weighted_gaussians_tsr = self.coefs_tsr.unsqueeze(0).to(x_tsr.device) * gaussians_tsr  # (B, N_centers)
        sum_tsr = torch.sum(weighted_gaussians_tsr, dim=1).unsqueeze(1)  # (B, 1)
        return sum_tsr  # (B, 1)

    def gaussian(self, x, sigma):
        return np.exp(-x**2/(2 * sigma**2))

    def batch_gaussian(self, x_tsr, mu_tsr, sigmas_tsr):  # x_tsr.shape = (B, N), mu_tsr.shape = (N), sigmas_tsr.shape = (N)
        y = torch.exp(-(x_tsr - mu_tsr)**2/(2 * sigmas_tsr**2))  # (B, N)
        return y  # (B, N)

    def _sigmas(self, sigma_ratio):
        xs = [x for (x, y) in self.sorted_xy_list]
        sigmas = [ sigma_ratio * (xs[1] - xs[0])]
        for x_ndx in range(1, len(self.sorted_xy_list) - 1):
            d_left = xs[x_ndx] - xs[x_ndx - 1]
            d_right = xs[x_ndx + 1] - xs[x_ndx]
            sigmas.append(sigma_ratio * min(d_left, d_right))
        sigmas.append(sigma_ratio * (xs[-1] - xs[-2]))
        return sigmas

class Spline1d(abc.ABC):
    def __init__(self, xys, ys, x0, xn, **kwargs):
        super().__init__()
        if xys is None:
            if ys is None or x0 is None or xn is None:
                raise ValueError(f"Spline1d.__init__(): You must specify either xys or (ys and x0 and xn)")
            delta_x = (xn - x0) / (len(ys) - 1)
            xs = np.arange(x0, xn + delta_x / 2, delta_x)
            self.xys = list(zip(xs, ys))
        else:
            # Make sure the xs are strictly increasing
            xs_are_increasing = True
            preceding_x = -np.inf
            for j in range(len(xys)):
                x = xys[j][0]
                if x <= preceding_x:
                    xs_are_increasing = False
                preceding_x = x
            if not xs_are_increasing:
                raise ValueError(f"Spline1d.__init__(): The x's are not strictly increasing")
            self.xys = xys
        self.n = len(self.xys) - 1
        self.coefs = None

    @abc.abstractmethod
    def _starting_column(self, j):  # The column from the A matrix in the A x = b equation where
        # the coefficients belonging the the j-th cell start
        pass

    @abc.abstractmethod
    def evaluate(self, x):
        pass

    def _corresponding_j(self, x):
        if x < self.xys[0][0]:
            return 0
        elif x >= self.xys[-1][0]:
            return self.n - 1
        else:
            for j in range(len(self.xys)):
                x_j = self.xys[j][0]
                x_jp1 = self.xys[j + 1][0]
                if x_j <= x < x_jp1:
                    return j
        # Should not reach this point
        raise ValueError(f"Spline1d._corresponding_j({x}): Could not find the corresponding j")

    def batch_evaluate(self, x_tsr):  # x_tsr.shape = (B, 1)
        # self.coefs.shape = (n, degree + 1)
        degree = self.coefs.shape[1] - 1
        # Select the cell index for each x
        indices = torch.bucketize(x_tsr, boundaries=torch.tensor([x for x, y in self.xys]).to(x_tsr.device)) - 1  # (B, 1)
        # Clip indices smaller than 0 and larger or equal to self.coefs.shape[0]
        indices = torch.clamp(indices, min=0, max=self.coefs.shape[0] - 1)
        coefs_tsr = torch.tensor(self.coefs).to(x_tsr.device)
        batch_coefs = coefs_tsr[indices.squeeze(), :]  # (B, d + 1)
        if len(batch_coefs.shape) == 1:  # B=1 was collapsed by squeeze(): Add a dummy batch dimension
            batch_coefs = batch_coefs.unsqueeze(0)  # (1, d + 1)
        y_tsr = torch.zeros_like(x_tsr)  # (B, 1)
        for p in range(degree + 1):  # 0, 1, ..., d
            coefs_col = degree - p  # ..., 3, 2, 1, 0
            if p == 0:
                y_tsr[:, 0] += batch_coefs[:, coefs_col]
            else:
                y_tsr[:, 0] += batch_coefs[:, coefs_col] * torch.pow(x_tsr.squeeze(), p)
        return y_tsr


class LinearSpline(Spline1d):
    def __init__(self, xys, ys=None, x0=None, xn=None):
        super().__init__(xys, ys, x0, xn)

        # a x + b = y
        A = np.zeros((2 * self.n, 2 * self.n), float)
        b = np.zeros(2 * self.n)
        row = 0
        # Values at the nodes
        for j in range(self.n):  # 0, 1, ... n-1
            x_j = self.xys[j][0]
            x_jp1 = self.xys[j + 1][0]
            y_j = self.xys[j][1]
            y_jp1 = self.xys[j + 1][1]
            col = self._starting_column(j)
            A[row, col: col + 2] = [x_j, 1]
            b[row] = y_j
            A[row + 1, col: col + 2] = [x_jp1, 1]
            b[row + 1] = y_jp1
            row += 2

        self.coefs = np.linalg.solve(A, b)  # (n * 2,)
        self.coefs = self.coefs.reshape(self.n, 2)  # (n, 2)

    def _starting_column(self, j):
        return 2 * j

    def evaluate(self, x):
        j = self._corresponding_j(x)
        a, b = self.coefs[j, :]
        return a * x + b

class QuadraticSpline(Spline1d):
    def __init__(self, xys, ys=None, x0=None, xn=None, **kwargs):
        super().__init__(xys, ys, x0, xn, **kwargs)

        # a x^2 + b x + c = 0
        A = np.zeros((3 * self.n, 3 * self.n), float)
        b = np.zeros(3 * self.n)
        row = 0
        # Values at the knots
        for j in range(self.n):  # 0, 1, ... n-1
            x_j = self.xys[j][0]
            x_jp1 = self.xys[j + 1][0]
            y_j = self.xys[j][1]
            y_jp1 = self.xys[j + 1][1]
            col = self._starting_column(j)
            A[row, col: col + 3] = [x_j**2, x_j, 1]
            b[row] = y_j
            A[row + 1, col: col + 3] = [x_jp1**2, x_jp1, 1]
            b[row + 1] = y_jp1
            row += 2
        # Derivative at the interfaces
        # 2 a_j x_j+1 + b_j - 2 a_j+1 x_j+1 - b_j+1 = 0
        for j in range(self.n - 1):  # 0, 1, ... n-2
            x_jp1 = self.xys[j + 1][0]  # x_j+1
            col_j = self._starting_column(j)
            col_jp1 = self._starting_column(j + 1)
            A[row, col_j: col_j + 3] = [2 * x_jp1, 1, 0]
            A[row, col_jp1: col_jp1 + 3] = [-2 * x_jp1, -1, 0]
            row += 1
        if kwargs['boundary_condition'] == 'left_1st_derivative_0':
            col = self._starting_column(0)
            x_0 = self.xys[0][0]
            A[row, col: col + 3] = [2 * x_0, 1, 0]
            row += 1
        elif kwargs['boundary_condition'] == 'left_2nd_derivative_0':
            col = self._starting_column(0)
            x_0 = self.xys[0][0]
            A[row, col: col + 3] = [2, 0, 0]
            row += 1
        elif kwargs['boundary_condition'] == 'right_1st_derivative_0':
            col = self._starting_column(self.n - 1)
            x_r = self.xys[-1][0]
            A[row, col: col + 3] = [2 * x_r, 1, 0]
            row += 1
        elif kwargs['boundary_condition'] == 'right_2nd_derivative_0':
            col = self._starting_column(self.n - 1)
            x_r = self.xys[-1][0]
            A[row, col: col + 3] = [2, 0, 0]
            row += 1
        else:
            raise NotImplementedError(f"QuadraticSpline.__init__(): Not implemented kwargs['boundary_condition'] '{kwargs['boundary_condition']}'")


        self.coefs = np.linalg.solve(A, b)  # (n * 3,)
        self.coefs = self.coefs.reshape(self.n, 3)  # (n, 3)

    def _starting_column(self, j):
        return 3 * j

    def evaluate(self, x):
        j = self._corresponding_j(x)
        a, b, c = self.coefs[j, :]
        return a * x**2 + b * x + c

class CubicSpline(Spline1d):
    def __init__(self, xys, ys=None, x0=None, xn=None, **kwargs):
        super().__init__(xys, ys, x0, xn)

        # a x^3 + b x^2 + c x + d = y
        A = np.zeros((4 * self.n, 4 * self.n), float)
        b = np.zeros(4 * self.n)
        row = 0
        # Values at the knots
        for j in range(self.n):  # 0, 1, ... n-1
            x_j = self.xys[j][0]
            x_jp1 = self.xys[j + 1][0]
            y_j = self.xys[j][1]
            y_jp1 = self.xys[j + 1][1]
            col = self._starting_column(j)
            A[row, col: col + 4] = [x_j**3, x_j**2, x_j, 1.0]
            b[row] = y_j
            A[row + 1, col: col + 4] = [x_jp1**3, x_jp1**2, x_jp1, 1.0]
            b[row + 1] = y_jp1
            row += 2
        # Continuity of the 1st derivative at the interfaces
        # 3 a_j x_j+1^2 + 2 b_j x_j+1 + c_j - 3 a_j+1 x_j+1^2 - 2 b_j+1 x_j+1 - c_j+1 = 0
        for j in range(self.n - 1):  # 0, 1, ... n - 2
            x_jp1 = self.xys[j + 1][0]  # x_j+1
            col_left = self._starting_column(j)
            col_right = self._starting_column(j + 1)
            A[row, col_left: col_left + 4] = [3 * x_jp1**2, 2 * x_jp1, 1.0, 0]
            A[row, col_right: col_right + 4] = [-3 * x_jp1**2, -2 * x_jp1, -1.0, 0]
            row += 1
        # Continuity of the 2nd derivative at the interfaces
        # 6 a_j x_j+1 + 2 b_j - 6 a_j+1 x_j+1 - 2 b_j+1 = 0
        for j in range(self.n - 1):  # 0, 1, ... n - 2
            x_jp1 = self.xys[j + 1][0]
            col_left = self._starting_column(j)
            col_right = self._starting_column(j + 1)
            A[row, col_left: col_left + 4] = [6 * x_jp1, 2, 0, 0]
            A[row, col_right: col_right + 4] = [-6 * x_jp1, -2, 0, 0]
            row += 1
        # Set the derivative to 0 at the boundaries
        if kwargs['boundary_condition'] == '1st_derivative_0':
            col_0 = self._starting_column(0)
            x_0 = self.xys[0][0]
            A[row, col_0: col_0 + 4] = [3 * x_0**2, 2 * x_0, 1, 0]
            col_r = self._starting_column(self.n - 1)
            x_r = self.xys[-1][0]
            A[row + 1, col_r: col_r + 4] = [3 * x_r**2, 2 * x_r, 1, 0]
            row += 2
        elif kwargs['boundary_condition'] == '2nd_derivative_0':
            col_0 = self._starting_column(0)
            x_0 = self.xys[0][0]
            A[row, col_0: col_0 + 4] = [6 * x_0, 2, 0, 0]
            col_r = self._starting_column(self.n - 1)
            x_r = self.xys[-1][0]
            A[row + 1, col_r: col_r + 4] = [6 * x_r, 2, 0, 0]
        else:
            raise NotImplementedError(f"CubicSpline.__init__(): Not implemented kwargs['boundary_condition'] '{kwargs['boundary_condition']}'")

        self.coefs = np.linalg.solve(A, b)  # (n * 4,)
        self.coefs = self.coefs.reshape(self.n, 4)  # (n, 4)

    def _starting_column(self, j):
        return 4 * j
    def evaluate(self, x):
        j = self._corresponding_j(x)
        a, b, c, d = self.coefs[j, :]
        return a * x**3 + b * x**2 + c * x + d