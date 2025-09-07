import numpy as np
import math
from scipy import integrate

def cos_vectorize(s):
    return np.vectorize(math.cos)(s)

def sin_vectorize(s):
    return np.vectorize(math.sin)(s)

class Expander:
    def __init__(self, length, expansion_type='odd'):
        self.expansion_types = ['odd', 'even', 'quarter_odd', 'quarter_even', 'duplicate']
        if not expansion_type in self.expansion_types:
            raise NotImplementedError(f"Expander.__init__(): Not implemented expansion type {expansion_type}. The valid expansion types are {self.expansion_types}.")
        self.L = length
        self.expansion_type = expansion_type

    def coefficients(self, signal, maximum_n=1000):
        if not type(signal) == np.ndarray:
            raise ValueError(f"Expander.coefficients(): type(signal) = {type(signal)}. We expect np.ndarray.")
        maximum_n = min(maximum_n, len(signal)//2)
        if self.expansion_type == 'odd':
            return self._half_range_odd(signal, maximum_n)
        elif self.expansion_type == 'even':
            return self._half_range_even(signal, maximum_n)
        elif self.expansion_type == 'quarter_odd':
            return self._quarter_range_odd(signal, maximum_n)
        elif self.expansion_type == 'quarter_even':
            return self._quarter_range_even(signal, maximum_n)
        elif self.expansion_type == 'duplicate':
            return self._half_range_duplicate(signal, maximum_n)
        else:
            raise NotImplementedError(f"Expander.coefficients(): Not implemented expansion type '{self.expansion_type}'")


    def evaluate(self, a_n, b_n, x):
        if self.expansion_type == 'odd':
            s = 0
            for n in range(1, len(b_n)):
                s += b_n[n] * math.sin(n * math.pi * x/self.L)
            return s
        elif self.expansion_type == 'even':
            s = a_n[0]
            for n in range(1, len(a_n)):
                s += a_n[n] * math.cos(n * math.pi * x/self.L)
            return s
        elif self.expansion_type == 'quarter_odd':
            s = 0
            for n in range(1, len(b_n)):
                s += b_n[n] * math.sin(n * math.pi * x/(2.0 * self.L))
            return s
        elif self.expansion_type == 'quarter_even':
            s = a_n[0]
            for n in range(1, len(a_n)):
                s += a_n[n] * math.cos(n * math.pi * x/(2.0 * self.L))
            return s
        elif self.expansion_type == 'duplicate':
            s = a_n[0]
            for n in range(1, len(a_n)):
                s += a_n[n] * math.cos(n * math.pi * x/self.L)
                s += b_n[n] * math.sin(n * math.pi * x/self.L)
            return s
        else:
            raise NotImplementedError(f"Expander.evaluate(): Not implemented expansion type '{self.expansion_type}'")

    def derivative(self, a_n, b_n, x):
        if self.expansion_type == 'odd':
            s = 0
            for n in range(1, len(b_n)):
                s += b_n[n] * n * math.pi/self.L * math.cos(n * math.pi * x/self.L)
            return s
        elif self.expansion_type == 'even':
            s = 0
            for n in range(1, len(a_n)):
                s += -a_n[n] * n * math.pi/self.L * math.sin(n * math.pi * x/self.L)
            return s
        elif self.expansion_type == 'quarter_odd':
            s = 0
            for n in range(1, len(b_n)):
                s += b_n[n] * n * math.pi/(2 * self.L) * math.cos(n * math.pi * x/(2.0 * self.L))
            return s
        elif self.expansion_type == 'quarter_even':
            s = 0
            for n in range(1, len(a_n)):
                s += -a_n[n] * n * math.pi/(2 * self.L) * math.sin(n * math.pi * x/(2.0 * self.L))
            return s
        elif self.expansion_type == 'duplicate':
            s = 0
            for n in range(1, len(a_n)):
                s += -a_n[n] * n * math.pi/self.L * math.sin(n * math.pi * x/self.L)
                s += b_n[n] * n * math.pi/self.L * math.cos(n * math.pi * x/self.L)
            return s
        else:
            raise NotImplementedError(f"Expander.derivative(): Not implemented expansion type '{self.expansion_type}'")

    def derivative_vector(self, a_n, b_n, number_of_points):
        delta_x = self.L/(number_of_points - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)
        deriv = np.zeros_like(xs)
        for k in range(len(xs)):
            x = xs[k]
            deriv[k] = self.derivative(a_n, b_n, x)
        return deriv

    def second_derivative(self, a_n, b_n, x):
        if self.expansion_type == 'odd':
            s = 0
            for n in range(1, len(b_n)):
                s += -b_n[n] * (n * math.pi/self.L)**2 * math.sin(n * math.pi * x/self.L)
            return s
        elif self.expansion_type == 'even':
            s = 0
            for n in range(1, len(a_n)):
                s += -a_n[n] * (n * math.pi/self.L)**2 * math.cos(n * math.pi * x/self.L)
            return s
        elif self.expansion_type == 'quarter_odd':
            s = 0
            for n in range(1, len(b_n)):
                s += -b_n[n] * (n * math.pi/(2 * self.L))**2 * math.sin(n * math.pi * x/(2.0 * self.L))
            return s
        elif self.expansion_type == 'quarter_even':
            s = 0
            for n in range(1, len(a_n)):
                s += -a_n[n] * (n * math.pi/(2 * self.L))**2 * math.cos(n * math.pi * x/(2.0 * self.L))
            return s
        elif self.expansion_type == 'duplicate':
            s = 0
            for n in range(1, len(a_n)):
                s += -a_n[n] * (n * math.pi/self.L)**2 * math.cos(n * math.pi * x/self.L)
                s += -b_n[n] * (n * math.pi/self.L)**2 * math.sin(n * math.pi * x/self.L)
            return s
        else:
            raise NotImplementedError(f"Expander.second_derivative(): Not implemented expansion type '{self.expansion_type}'")

    def second_derivative_vector(self, a_n, b_n, number_of_points, expansi):
        delta_x = self.L/(number_of_points - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)
        second_deriv = np.zeros_like(xs)
        for k in range(len(xs)):
            x = xs[k]
            second_deriv[k] = self.second_derivative(a_n, b_n, x)
        return second_deriv

    def reconstruct(self, a_n, b_n, signal_length):
        delta_x = self.L/(signal_length - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)
        reconstruction = np.zeros((signal_length), dtype=float)
        for k in range(signal_length):
            x = xs[k]
            reconstruction[k] = self.evaluate(a_n, b_n, x)
        return reconstruction

    def _half_range_odd(self, signal, maximum_n):
        a_n = np.zeros((maximum_n + 1), dtype=float)
        b_n = np.zeros((maximum_n + 1), dtype=float)
        delta_x = self.L/(len(signal) - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)  # [0, dx, ..., L]
        for n in range(1, maximum_n + 1):
            sinnpix_L = sin_vectorize(n * math.pi * xs/self.L)
            b = 2.0 / self.L * integrate.simpson(y=(signal * sinnpix_L), x=xs)
            b_n[n] = b
        return a_n, b_n

    def _half_range_even(self, signal, maximum_n):
        a_n = np.zeros((maximum_n + 1), dtype=float)
        b_n = np.zeros((maximum_n + 1), dtype=float)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)  # [0, dx, ..., L]
        a_n[0] = 1.0/self.L * integrate.simpson(y=(signal), x=xs)
        for n in range(1, maximum_n + 1):
            cosnpix_L = cos_vectorize(n * math.pi * xs/self.L)
            a = 2.0/self.L * integrate.simpson(y=(signal * cosnpix_L), x=xs)
            a_n[n] = a
        return a_n, b_n

    def _quarter_range_odd(self, signal, maximum_n):
        a_n = np.zeros((maximum_n + 1), dtype=float)
        b_n = np.zeros((maximum_n + 1), dtype=float)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)  # [0, dx, ..., L]
        for n in range(1, maximum_n + 1):
            sinnpix_2L = sin_vectorize(n * math.pi * xs/(2.0 * self.L))
            b = 1.0/self.L * ((-1)**(n + 1) + 1) * integrate.simpson(y=(signal * sinnpix_2L), x=xs)
            b_n[n] = b
        return a_n, b_n

    def _quarter_range_even(self, signal, maximum_n):
        a_n = np.zeros((maximum_n + 1), dtype=float)
        b_n = np.zeros((maximum_n + 1), dtype=float)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)  # [0, dx, ..., L]
        f_L = signal[-1]
        a_n[0] = f_L
        for n in range(1, maximum_n + 1):
            sinnpi_2 = sin_vectorize(n * math.pi/2)
            cosnpix_2L = cos_vectorize(n * math.pi * xs/(2 * self.L))
            t1 = -8.0 * self.L * f_L/(n * math.pi) * sinnpi_2
            t2 = 2.0 * (1 - (-1)**n) * integrate.simpson(y=(signal * cosnpix_2L), x=xs)
            a = 1.0/(2 * self.L) * (t1 + t2)
            a_n[n] = a
        return a_n, b_n

    def _half_range_duplicate(self, signal, maximum_n):
        a_n = np.zeros((maximum_n + 1), dtype=float)
        b_n = np.zeros((maximum_n + 1), dtype=float)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)  # [0, dx, ..., L]
        a_n[0] = 1.0/self.L * integrate.simpson(y=(signal), x=xs)
        for n in range(1, maximum_n + 1):
            cosnpix_L = cos_vectorize(n * math.pi * xs/self.L)
            sinnpix_L = sin_vectorize(n * math.pi * xs/self.L)
            a_n[n] = (1 + (-1)**n)/self.L * integrate.simpson(y=(signal * cosnpix_L), x=xs)
            b_n[n] = (1 + (-1)**n)/self.L * integrate.simpson(y=(signal * sinnpix_L), x=xs)
        return a_n, b_n

class PeriodicSignal:
    def __init__(self, half_period):
        self.L = half_period


    def coefficients(self, signal, maximum_n=1000):
        if not type(signal) == np.ndarray:
            raise ValueError(f"PeriodicSignal.coefficients(): type(signal) = {type(signal)}. We expect np.ndarray.")
        maximum_n = min(maximum_n, len(signal)//2)
        a = np.zeros((maximum_n + 1))
        b = np.zeros((maximum_n + 1))
        delta_x = 2.0 * self.L/(len(signal) - 1)
        x = np.arange(-self.L, self.L + delta_x/2, delta_x)
        a[0] = 1.0/(2 * self.L) * integrate.simpson(y=signal, x=x)
        for n in range(1, maximum_n + 1):
            cosnpix_L = cos_vectorize(n * math.pi * x/self.L)
            a[n] = 1.0/self.L * integrate.simpson(y=(signal * cosnpix_L), x=x)
            sinnpix_L = sin_vectorize(n * math.pi * x/self.L)
            b[n] = 1.0/self.L * integrate.simpson(y=(signal * sinnpix_L), x=x)
        return a, b

    def evaluate(self, x, a, b):
        sum = a[0]
        for n in range(1, len(a)):
            sum += a[n] * math.cos(n * math.pi * x/self.L) + b[n] * math.sin(n * math.pi * x/self.L)
        return sum

    def evaluate_vector(self, xs, a, b):
        ys = np.zeros_like(xs)
        for k in range(len(xs)):
            x = xs[k]
            ys[k] = self.evaluate(x, a, b)
        return ys