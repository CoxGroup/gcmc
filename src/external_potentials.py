'''
    GCMC simulation for fluids with short-ranged potentials
    Copyright (C) 2024  Anna Bui

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np
from constants import very_large_number

class ExtPotential:
    def __call__(self, position):
        raise NotImplementedError("This method should be overridden by subclasses")

    def calculate_multiple(self, positions):
        raise NotImplementedError("This method should be overridden by subclasses")

class NoExternalPotential(ExtPotential):
    def __call__(self, position):
        return 0.0
    def calculate_multiple(self, positions):
        return 0.0  
    
class WallPotential(ExtPotential):
    def __init__(self, width):
        self.width = width
    
    def __call__(self, position):
        return np.where((position[0] > self.width), very_large_number, 0.0)

    def calculate_multiple(self, positions):
        energies = np.where((positions[:, 0] > self.width), very_large_number, 0.0)
        return np.sum(energies)

class SlitPotential(ExtPotential):
    def __init__(self, low, high):
        self.low = low
        self.high = high
    
    def __call__(self, position):
        return np.where((position[0] > self.high) | (position[0] < self.low), very_large_number, 0.0)

    def calculate_multiple(self, positions):
        energies = np.where((positions[:, 0] > self.high) | (positions[:, 0] < self.low), very_large_number, 0.0)
        return np.sum(energies)

class SlitLJPotential(ExtPotential):
    def __init__(self, low, high, epsilon, sigma):
        self.low = low
        self.high = high
        self.epsilon = epsilon
        self.sigma = sigma
        
    def __call__(self, position):
        r_low = position[0] - self.low
        r_high = self.high - position[0]
        
        energy_low = 4 * self.epsilon * ((self.sigma / r_low)**12 - (self.sigma / r_low)**6)
        energy_high = 4 * self.epsilon * ((self.sigma / r_high)**12 - (self.sigma / r_high)**6)
        
        return np.where((position[0] > self.high) | (position[0] < self.low), very_large_number, energy_low + energy_high)


    def calculate_multiple(self, positions):
        energies = np.zeros(positions.shape[0])
        
        outside_low = positions[:, 0] < self.low
        outside_high = positions[:, 0] > self.high
        inside = ~outside_low & ~outside_high

        energies[outside_low | outside_high] = very_large_number
        
        r_low = positions[inside, 0] - self.low
        r_high = self.high - positions[inside, 0]
        
        energy_low = 4 * self.epsilon * ((self.sigma / r_low)**12 - (self.sigma / r_low)**6)
        energy_high = 4 * self.epsilon * ((self.sigma / r_high)**12 - (self.sigma / r_high)**6)
        
        energies[inside] = np.where(r_low < self.sigma, energy_low, 0.0) + np.where(r_high < self.sigma, energy_high, 0.0)
        
        return np.sum(energies)
    
    
class SlitLJ93Potential(ExtPotential):
    def __init__(self, low, high, epsilon, sigma, cutoff):
        self.low = low
        self.high = high
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        
        preratio3 = (self.sigma / self.cutoff) ** 3
        preratio9 = preratio3 ** 3
        
        self.shift = self.epsilon * (preratio9 * 2/15 - preratio3)

    def __call__(self, position):
        
        r_low = position[0] - self.low
        r_high = self.high - position[0]
        

        energy_low =  self.epsilon * ((2/15)*(self.sigma / r_low)**9 - (self.sigma / r_low)**3) - self.shift
        energy_high = self.epsilon * ((2/15)*(self.sigma / r_high)**9 - (self.sigma / r_high)**3) - self.shift
 
        energy_low = np.where(r_low >= self.cutoff, 0.0, energy_low)
        energy_high = np.where(r_high >= self.cutoff, 0.0, energy_high)

        return np.where((position[0] > self.high) | (position[0] < self.low), very_large_number, energy_low + energy_high)

    def calculate_multiple(self, positions):

        
        energies = np.zeros(positions.shape[0])

        outside_low = positions[:, 0] < self.low
        outside_high = positions[:, 0] > self.high
        inside = ~outside_low & ~outside_high

        
        energies[outside_low | outside_high] = very_large_number

        r_low = positions[inside, 0] - self.low
        r_high = self.high - positions[inside, 0]

        energy_low =  self.epsilon * ((2/15)*(self.sigma / r_low)**9 - (self.sigma / r_low)**3) - self.shift
        energy_high = self.epsilon * ((2/15)*(self.sigma / r_high)**9 - (self.sigma / r_high)**3) - self.shift

        energy_low[r_low >= self.cutoff] = 0.0
        energy_high[r_high >= self.cutoff] = 0.0

        energies[inside] = energy_low + energy_high

        return np.sum(energies)

    
    
class BaseTrainingPotential(ExtPotential):
    def __init__(self, A1, A2, A3, A4, phi1, phi2, phi3, phi4, L):
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4
        self.phi1 = phi1
        self.phi2 = phi2
        self.phi3 = phi3
        self.phi4 = phi4
        self.L = L

    def calculate_sines(self, position):
        ratio = 2 * np.pi * position[0] / self.L
        sine_terms = sum(
            A * np.sin(ratio * i + phi)
            for i, (A, phi) in enumerate(
                [(self.A1, self.phi1), (self.A2, self.phi2), (self.A3, self.phi3), (self.A4, self.phi4)], start=1
            )
        )
        return sine_terms

    def calculate_linear_potential(self, Va, Vb, xa, xb, position):
        Vlin = Va + (Vb - Va) * (position[0] - xa) / (xb - xa)
        return np.where((xa <= position[0]) & (position[0] <= xb), Vlin, 0.0)

class TrainingPotential(BaseTrainingPotential):
    def __init__(self, A1, A2, A3, A4, phi1, phi2, phi3, phi4, L, linear_potentials):
        super().__init__(A1, A2, A3, A4, phi1, phi2, phi3, phi4, L)
        self.linear_potentials = linear_potentials

    def __call__(self, position):
        sine_terms = self.calculate_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        return sine_terms + linear_terms

    def calculate_multiple(self, positions):
        ratio = 2 * np.pi * positions[:, 0] / self.L
        sine_terms = (
            self.A1 * np.sin(ratio + self.phi1) +
            self.A2 * np.sin(2 * ratio + self.phi2) +
            self.A3 * np.sin(3 * ratio + self.phi3) +
            self.A4 * np.sin(4 * ratio + self.phi4)
        )

        linear_terms = np.zeros_like(positions[:, 0])
        for Va, Vb, xa, xb in self.linear_potentials:
            Vlin = Va + (Vb - Va) * (positions[:, 0] - xa) / (xb - xa)
            linear_terms += np.where((xa <= positions[:, 0]) & (positions[:, 0] <= xb), Vlin, 0.0)

        energies = sine_terms + linear_terms
        return np.sum(energies)


class BaseTrainingPotentialWithCharge(ExtPotential):
    def __init__(self, q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, L):
        self.q = q
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4
        self.phi1 = phi1
        self.phi2 = phi2
        self.phi3 = phi3
        self.phi4 = phi4
        self.q_A1 = q_A1
        self.q_A2 = q_A2
        self.q_A3 = q_A3
        self.q_A4 = q_A4
        self.q_phi1 = q_phi1
        self.q_phi2 = q_phi2
        self.q_phi3 = q_phi3
        self.q_phi4 = q_phi4
        self.L = L

    def calculate_sines(self, position):
        ratio = 2 * np.pi * position[0] / self.L
        sine_terms = sum(
            A * np.sin(ratio * i + phi)
            for i, (A, phi) in enumerate(
                [(self.A1, self.phi1), (self.A2, self.phi2), (self.A3, self.phi3), (self.A4, self.phi4)], start=1
            )
        )
        
        return sine_terms 

    def calculate_cosines(self, position):
        ratio = 2 * np.pi * position[0] / self.L
        sine_terms = sum(
            A * np.cos(ratio * i + phi)
            for i, (A, phi) in enumerate(
                [(self.A1, self.phi1), (self.A2, self.phi2), (self.A3, self.phi3), (self.A4, self.phi4)], start=1
            )
        )
        
        return sine_terms 

    def calculate_q_sines(self, position):
        ratio = 2 * np.pi * position[0] / self.L
        sine_terms_q = sum(
            A * np.sin(ratio * i + phi)
            for i, (A, phi) in enumerate(
                [(self.q_A1, self.q_phi1), (self.q_A2, self.q_phi2), (self.q_A3, self.q_phi3), (self.q_A4, self.q_phi4)], start=1
            )
        )
        return sine_terms_q 

    def calculate_q_cosines(self, position):
        ratio = 2 * np.pi * position[0] / self.L
        cosine_terms_q = sum(
            A * np.cos(ratio * i + phi)
            for i, (A, phi) in enumerate(
                [(self.q_A1, self.q_phi1), (self.q_A2, self.q_phi2), (self.q_A3, self.q_phi3), (self.q_A4, self.q_phi4)], start=1
            )
        )
        return cosine_terms_q 

    def calculate_linear_potential(self, Va, Vb, xa, xb, position):
        
        Vlin = Va + (Vb - Va) * (position[0] - xa) / (xb - xa)
        
        return np.where((xa <= position[0]) & (position[0] <= xb), Vlin, 0.0)
    
    def calculate_q_linear_potential(self, qVa, qVb, qxa, qxb, position):
        
        Vlin_q = qVa + (qVb - qVa) * (position[0] - qxa) / (qxb - qxa)
        
        return np.where((qxa <= position[0]) & (position[0] <= qxb), Vlin_q, 0.0)


class TrainingPotentialWithChargeCos(BaseTrainingPotentialWithCharge):
    def __init__(self, q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, linear_potentials, q_linear_potentials, L):
        super().__init__(q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, L)
        self.linear_potentials = linear_potentials
        self.q_linear_potentials = q_linear_potentials

    def __call__(self, position):
        sine_terms = self.calculate_cosines(position)
        sine_terms_q = self.calculate_q_cosines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        linear_terms_q = sum(
            self.calculate_q_linear_potential(qVa, qVb, qxa, qxb, position)
            for qVa, qVb, qxa, qxb in self.q_linear_potentials
        )
        return sine_terms + linear_terms + linear_terms_q * self.q + sine_terms_q * self.q 

    def get_electrostatic(self, position):
        sine_terms_q = self.calculate_q_cosines(position)
        linear_terms_q = sum(
            self.calculate_q_linear_potential(qVa, qVb, qxa, qxb, position)
            for qVa, qVb, qxa, qxb in self.q_linear_potentials
        )
        return linear_terms_q * self.q + sine_terms_q * self.q 

    def get_non_electrostatic(self, position):
        sine_terms = self.calculate_cosines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        return sine_terms + linear_terms 

    def calculate_multiple(self, positions):
        ratio = 2 * np.pi * positions[:, 0] / self.L
        sine_terms = (
            self.A1 * np.cos(ratio + self.phi1) +
            self.A2 * np.cos(2 * ratio + self.phi2) +
            self.A3 * np.cos(3 * ratio + self.phi3) +
            self.A4 * np.cos(4 * ratio + self.phi4)
        )

        cosine_terms_q = (
            self.q_A1 * np.cos(ratio + self.q_phi1) +
            self.q_A2 * np.cos(2 * ratio + self.q_phi2) +
            self.q_A3 * np.cos(3 * ratio + self.q_phi3) +
            self.q_A4 * np.cos(4 * ratio + self.q_phi4)
        )
        
        linear_terms = np.zeros_like(positions[:, 0])
        for Va, Vb, xa, xb in self.linear_potentials:
            Vlin = Va + (Vb - Va) * (positions[:, 0] - xa) / (xb - xa)
            linear_terms += np.where((xa <= positions[:, 0]) & (positions[:, 0] <= xb), Vlin, 0.0)

        linear_terms_q = np.zeros_like(positions[:, 0])
        for qVa, qVb, qxa, qxb in self.q_linear_potentials:
            Vlin_q = qVa + (qVb - qVa) * (positions[:, 0] - qxa) / (qxb - qxa)
            linear_terms_q += np.where((qxa <= positions[:, 0]) & (positions[:, 0] <= qxb), Vlin_q, 0.0)
        
        energies = sine_terms + linear_terms + cosine_terms_q * self.q + linear_terms_q * self.q
        return np.sum(energies)


class TrainingPotentialWithCharge(BaseTrainingPotentialWithCharge):
    def __init__(self, q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, linear_potentials, q_linear_potentials, L):
        super().__init__(q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, L)
        self.linear_potentials = linear_potentials
        self.q_linear_potentials = q_linear_potentials

    def __call__(self, position):
        sine_terms = self.calculate_sines(position)
        sine_terms_q = self.calculate_q_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        linear_terms_q = sum(
            self.calculate_q_linear_potential(qVa, qVb, qxa, qxb, position)
            for qVa, qVb, qxa, qxb in self.q_linear_potentials
        )
        return sine_terms + linear_terms + linear_terms_q * self.q + sine_terms_q * self.q 

    def get_electrostatic(self, position):
        sine_terms_q = self.calculate_q_sines(position)
        linear_terms_q = sum(
            self.calculate_q_linear_potential(qVa, qVb, qxa, qxb, position)
            for qVa, qVb, qxa, qxb in self.q_linear_potentials
        )
        return linear_terms_q * self.q + sine_terms_q * self.q 

    def get_non_electrostatic(self, position):
        sine_terms = self.calculate_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        return sine_terms + linear_terms 

    def calculate_multiple(self, positions):
        ratio = 2 * np.pi * positions[:, 0] / self.L
        sine_terms = (
            self.A1 * np.sin(ratio + self.phi1) +
            self.A2 * np.sin(2 * ratio + self.phi2) +
            self.A3 * np.sin(3 * ratio + self.phi3) +
            self.A4 * np.sin(4 * ratio + self.phi4)
        )

        sine_terms_q = (
            self.q_A1 * np.sin(ratio + self.q_phi1) +
            self.q_A2 * np.sin(2 * ratio + self.q_phi2) +
            self.q_A3 * np.sin(3 * ratio + self.q_phi3) +
            self.q_A4 * np.sin(4 * ratio + self.q_phi4)
        )
        
        linear_terms = np.zeros_like(positions[:, 0])
        for Va, Vb, xa, xb in self.linear_potentials:
            Vlin = Va + (Vb - Va) * (positions[:, 0] - xa) / (xb - xa)
            linear_terms += np.where((xa <= positions[:, 0]) & (positions[:, 0] <= xb), Vlin, 0.0)

        linear_terms_q = np.zeros_like(positions[:, 0])
        for qVa, qVb, qxa, qxb in self.q_linear_potentials:
            Vlin_q = qVa + (qVb - qVa) * (positions[:, 0] - qxa) / (qxb - qxa)
            linear_terms_q += np.where((qxa <= positions[:, 0]) & (positions[:, 0] <= qxb), Vlin_q, 0.0)
        
        energies = sine_terms + linear_terms + sine_terms_q * self.q + linear_terms_q * self.q
        return np.sum(energies)



class TrainingPotentialWithWalls(BaseTrainingPotential):
    def __init__(self, A1, A2, A3, A4, phi1, phi2, phi3, phi4, L, width, linear_potentials):
        super().__init__(A1, A2, A3, A4, phi1, phi2, phi3, phi4, L)
        self.width = width
        self.linear_potentials = linear_potentials

    def __call__(self, position):
        sine_terms = self.calculate_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        return np.where((self.width/2 > position[0]) | (position[0] > self.L - self.width/2), very_large_number, sine_terms + linear_terms) 

    def calculate_multiple(self, positions):
        ratio = 2 * np.pi * positions[:, 0] / self.L
        sine_terms = (
            self.A1 * np.sin(ratio + self.phi1) +
            self.A2 * np.sin(2 * ratio + self.phi2) +
            self.A3 * np.sin(3 * ratio + self.phi3) +
            self.A4 * np.sin(4 * ratio + self.phi4)
        )

        linear_terms = np.zeros_like(positions[:, 0])
        for Va, Vb, xa, xb in self.linear_potentials:
            Vlin = Va + (Vb - Va) * (positions[:, 0] - xa) / (xb - xa)
            linear_terms += np.where((xa <= positions[:, 0]) & (positions[:, 0] <= xb), Vlin, 0.0)

        energies = np.where((self.width/2 > positions[:, 0]) | (positions[:, 0] > self.L - self.width/2), very_large_number, sine_terms + linear_terms)
        return np.sum(energies)


class TrainingPotentialWithLJ93Walls(BaseTrainingPotential):
    def __init__(self, A1, A2, A3, A4, phi1, phi2, phi3, phi4, L, low, high, epsilon, sigma, cutoff, linear_potentials):
        super().__init__(A1, A2, A3, A4, phi1, phi2, phi3, phi4, L)
        self.low = low
        self.high = high
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.linear_potentials = linear_potentials
        
        preratio3 = (self.sigma / self.cutoff) ** 3
        preratio9 = preratio3 ** 3
        
        self.shift = self.epsilon * (preratio9 * 2/15 - preratio3)

    def __call__(self, position):
        
        r_low = position[0] - self.low
        r_high = self.high - position[0]
        
        energy_low =  self.epsilon * ((2/15)*(self.sigma / r_low)**9 - (self.sigma / r_low)**3) - self.shift
        energy_high = self.epsilon * ((2/15)*(self.sigma / r_high)**9 - (self.sigma / r_high)**3) - self.shift
        
        energy_low = np.where(r_low >= self.cutoff, 0.0, energy_low)
        energy_high = np.where(r_high >= self.cutoff, 0.0, energy_high)

        
        sine_terms = self.calculate_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        
        return np.where((position[0] > self.high) | (position[0] < self.low), very_large_number, energy_low + energy_high + sine_terms + linear_terms)


    def calculate_multiple(self, positions):
        ratio = 2 * np.pi * positions[:, 0] / self.L
        sine_terms = (
            self.A1 * np.sin(ratio + self.phi1) +
            self.A2 * np.sin(2 * ratio + self.phi2) +
            self.A3 * np.sin(3 * ratio + self.phi3) +
            self.A4 * np.sin(4 * ratio + self.phi4)
        )

        linear_terms = np.zeros_like(positions[:, 0])
        for Va, Vb, xa, xb in self.linear_potentials:
            Vlin = Va + (Vb - Va) * (positions[:, 0] - xa) / (xb - xa)
            linear_terms += np.where((xa <= positions[:, 0]) & (positions[:, 0] <= xb), Vlin, 0.0)

        
        energies = np.zeros(positions.shape[0])
        outside_low = positions[:, 0] < self.low
        outside_high = positions[:, 0] > self.high
        inside = ~outside_low & ~outside_high
        
        energies[outside_low | outside_high] = very_large_number

        r_low = positions[inside, 0] - self.low
        r_high = self.high - positions[inside, 0]
        
        energy_low =  self.epsilon * ((2/15)*(self.sigma / r_low)**9 - (self.sigma / r_low)**3) - self.shift
        energy_high = self.epsilon * ((2/15)*(self.sigma / r_high)**9 - (self.sigma / r_high)**3) - self.shift

        energy_low[r_low >= self.cutoff] = 0.0
        energy_high[r_high >= self.cutoff] = 0.0

        energies[inside] = sine_terms + linear_terms + energy_low + energy_high
        
        return np.sum(energies)


class TrainingPotentialWithChargeWithWalls(BaseTrainingPotentialWithCharge):
    def __init__(self, q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, linear_potentials, q_linear_potentials, width, L):
        super().__init__(q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, L)
        self.width = width
        self.linear_potentials = linear_potentials
        self.q_linear_potentials = q_linear_potentials
        
    def __call__(self, position):
        sine_terms = self.calculate_sines(position)
        sine_terms_q = self.calculate_q_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        linear_terms_q = sum(
            self.calculate_q_linear_potential(qVa, qVb, qxa, qxb, position)
            for qVa, qVb, qxa, qxb in self.q_linear_potentials
        )
        return np.where((self.width/2 > position[0]) | (position[0] > self.L - self.width/2), very_large_number, sine_terms + linear_terms + sine_terms_q * self.q + linear_terms_q * self.q) 

    def get_electrostatic(self, position):
        sine_terms_q = self.calculate_q_sines(position)
        linear_terms_q = sum(
            self.calculate_q_linear_potential(qVa, qVb, qxa, qxb, position)
            for qVa, qVb, qxa, qxb in self.q_linear_potentials
        )
        return np.where((self.width/2 > position[0]) | (position[0] > self.L - self.width/2), 0.0, sine_terms_q * self.q + linear_terms_q * self.q) 

    def get_non_electrostatic(self, position):
        sine_terms = self.calculate_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        return np.where((self.width/2 > position[0]) | (position[0] > self.L - self.width/2), very_large_number, sine_terms + linear_terms ) 

    def calculate_multiple(self, positions):
        ratio = 2 * np.pi * positions[:, 0] / self.L
        sine_terms = (
            self.A1 * np.sin(ratio + self.phi1) +
            self.A2 * np.sin(2 * ratio + self.phi2) +
            self.A3 * np.sin(3 * ratio + self.phi3) +
            self.A4 * np.sin(4 * ratio + self.phi4)
        )

        sine_terms_q = (
            self.q_A1 * np.sin(ratio + self.q_phi1) +
            self.q_A2 * np.sin(2 * ratio + self.q_phi2) +
            self.q_A3 * np.sin(3 * ratio + self.q_phi3) +
            self.q_A4 * np.sin(4 * ratio + self.q_phi4)
        )
        
        linear_terms = np.zeros_like(positions[:, 0])
        for Va, Vb, xa, xb in self.linear_potentials:
            Vlin = Va + (Vb - Va) * (positions[:, 0] - xa) / (xb - xa)
            linear_terms += np.where((xa <= positions[:, 0]) & (positions[:, 0] <= xb), Vlin, 0.0)

        linear_terms_q = np.zeros_like(positions[:, 0])
        for qVa, qVb, qxa, qxb in self.q_linear_potentials:
            Vlin_q = qVa + (qVb - qVa) * (positions[:, 0] - qxa) / (qxb - qxa)
            linear_terms_q += np.where((qxa <= positions[:, 0]) & (positions[:, 0] <= qxb), Vlin_q, 0.0)
        
        energies = np.where((self.width/2 > positions[:, 0]) | (positions[:, 0] > self.L - self.width/2), very_large_number, sine_terms + linear_terms + sine_terms_q * self.q + linear_terms_q * self.q)
        return np.sum(energies)



class TrainingPotentialWithChargeWithLJ93Walls(BaseTrainingPotentialWithCharge):
    def __init__(self, q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, linear_potentials, q_linear_potentials, low, high, epsilon, sigma, cutoff, L):
        super().__init__(q, A1, A2, A3, A4, phi1, phi2, phi3, phi4, q_A1, q_A2, q_A3, q_A4, q_phi1, q_phi2, q_phi3, q_phi4, L)
        self.low = low
        self.high = high
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.linear_potentials = linear_potentials
        self.q_linear_potentials = q_linear_potentials
        
        preratio3 = (self.sigma / self.cutoff) ** 3
        preratio9 = preratio3 ** 3
        
        self.shift = self.epsilon * (preratio9 * 2/15 - preratio3)        
        
    def __call__(self, position):
        
        r_low = position[0] - self.low
        r_high = self.high - position[0]
        
        energy_low =  self.epsilon * ((2/15)*(self.sigma / r_low)**9 - (self.sigma / r_low)**3) - self.shift
        energy_high = self.epsilon * ((2/15)*(self.sigma / r_high)**9 - (self.sigma / r_high)**3) - self.shift
        
        energy_low = np.where(r_low >= self.cutoff, 0.0, energy_low)
        energy_high = np.where(r_high >= self.cutoff, 0.0, energy_high)

        sine_terms = self.calculate_sines(position)
        sine_terms_q = self.calculate_q_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        linear_terms_q = sum(
            self.calculate_q_linear_potential(qVa, qVb, qxa, qxb, position)
            for qVa, qVb, qxa, qxb in self.q_linear_potentials
        )
        return np.where((position[0] > self.high) | (position[0] < self.low), very_large_number, energy_low + energy_high + sine_terms + linear_terms + sine_terms_q * self.q + linear_terms_q * self.q)

    def get_electrostatic(self, position):
        sine_terms_q = self.calculate_q_sines(position)
        linear_terms_q = sum(
            self.calculate_q_linear_potential(qVa, qVb, qxa, qxb, position)
            for qVa, qVb, qxa, qxb in self.q_linear_potentials
        )
        return np.where((position[0] > self.high) | (position[0] < self.low), 0.0, sine_terms_q * self.q + linear_terms_q * self.q) 

    def get_non_electrostatic(self, position):
        sine_terms = self.calculate_sines(position)
        linear_terms = sum(
            self.calculate_linear_potential(Va, Vb, xa, xb, position)
            for Va, Vb, xa, xb in self.linear_potentials
        )
        return np.where((position[0] > self.high) | (position[0] < self.low), very_large_number, sine_terms + linear_terms ) 

    def calculate_multiple(self, positions):
        ratio = 2 * np.pi * positions[:, 0] / self.L
        sine_terms = (
            self.A1 * np.sin(ratio + self.phi1) +
            self.A2 * np.sin(2 * ratio + self.phi2) +
            self.A3 * np.sin(3 * ratio + self.phi3) +
            self.A4 * np.sin(4 * ratio + self.phi4)
        )

        sine_terms_q = (
            self.q_A1 * np.sin(ratio + self.q_phi1) +
            self.q_A2 * np.sin(2 * ratio + self.q_phi2) +
            self.q_A3 * np.sin(3 * ratio + self.q_phi3) +
            self.q_A4 * np.sin(4 * ratio + self.q_phi4)
        )
        
        linear_terms = np.zeros_like(positions[:, 0])
        for Va, Vb, xa, xb in self.linear_potentials:
            Vlin = Va + (Vb - Va) * (positions[:, 0] - xa) / (xb - xa)
            linear_terms += np.where((xa <= positions[:, 0]) & (positions[:, 0] <= xb), Vlin, 0.0)

        linear_terms_q = np.zeros_like(positions[:, 0])
        for qVa, qVb, qxa, qxb in self.q_linear_potentials:
            Vlin_q = qVa + (qVb - qVa) * (positions[:, 0] - qxa) / (qxb - qxa)
            linear_terms_q += np.where((qxa <= positions[:, 0]) & (positions[:, 0] <= qxb), Vlin_q, 0.0)
        
        energies = np.zeros(positions.shape[0])
        outside_low = positions[:, 0] < self.low
        outside_high = positions[:, 0] > self.high
        inside = ~outside_low & ~outside_high
        
        energies[outside_low | outside_high] = very_large_number

        r_low = positions[inside, 0] - self.low
        r_high = self.high - positions[inside, 0]
        
        energy_low =  self.epsilon * ((2/15)*(self.sigma / r_low)**9 - (self.sigma / r_low)**3) - self.shift
        energy_high = self.epsilon * ((2/15)*(self.sigma / r_high)**9 - (self.sigma / r_high)**3) - self.shift

        energy_low[r_low >= self.cutoff] = 0.0
        energy_high[r_high >= self.cutoff] = 0.0

        energies[inside] = sine_terms + linear_terms + sine_terms_q * self.q + linear_terms_q * self.q + energy_low + energy_high

        return np.sum(energies)

def initialize_external_potentials(config):
    external_potentials_dict = {}

    for type, params in config['particle_types'].items():
        ext_type = params.get('Vext')
        if ext_type:
            if ext_type == "WallPotential":
                external_potentials_dict[type] = WallPotential(width=params['width'])
            elif ext_type == "SlitPotential":
                external_potentials_dict[type] = SlitPotential(low=params['low'], high=params['high'])
            elif ext_type == "SlitLJPotential":
                kB = config['kB']
                T = config['T']
                external_potentials_dict[type] = SlitLJPotential(
                    low=params['low'], high=params['high'], epsilon=params['epsilon']*kB*T, sigma=params['sigma'])
            elif ext_type == "SlitLJ93Potential":
                kB = config['kB']
                T = config['T']
                external_potentials_dict[type] = SlitLJ93Potential(
                    low=params['low'], high=params['high'], epsilon=params['epsilon']*kB*T, sigma=params['sigma'], cutoff=params['cutoff'])
            elif ext_type.startswith("TrainingPotentialWithChargeWithWalls"):
                kB = config['kB']
                T = config['T']
                linear_potentials = [
                    (params[f'Va{i}']*kB*T, params[f'Vb{i}']*kB*T, params[f'xa{i}'], params[f'xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                q_linear_potentials = [
                    (params[f'q_Va{i}']*kB*T, params[f'q_Vb{i}']*kB*T, params[f'q_xa{i}'], params[f'q_xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                external_potentials_dict[type] = TrainingPotentialWithChargeWithWalls(
                    q=params['q'], A1=params['A1']*kB*T, A2=params['A2']*kB*T, A3=params['A3']*kB*T, A4=params['A4']*kB*T,
                    phi1=params['phi1'], phi2=params['phi2'], phi3=params['phi3'], phi4=params['phi4'],
                    q_A1=params['q_A1']*kB*T, q_A2=params['q_A2']*kB*T, q_A3=params['q_A3']*kB*T, q_A4=params['q_A4']*kB*T,
                    q_phi1=params['q_phi1'], q_phi2=params['q_phi2'], q_phi3=params['q_phi3'], q_phi4=params['q_phi4'],
                    linear_potentials=linear_potentials, q_linear_potentials=q_linear_potentials, L=params['L'], width=params['width']
                )
            elif ext_type.startswith("TrainingPotentialWithChargeWithLJ93Walls"):
                kB = config['kB']
                T = config['T']
                linear_potentials = [
                    (params[f'Va{i}']*kB*T, params[f'Vb{i}']*kB*T, params[f'xa{i}'], params[f'xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                q_linear_potentials = [
                    (params[f'q_Va{i}']*kB*T, params[f'q_Vb{i}']*kB*T, params[f'q_xa{i}'], params[f'q_xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                external_potentials_dict[type] = TrainingPotentialWithChargeWithLJ93Walls(
                    q=params['q'], A1=params['A1']*kB*T, A2=params['A2']*kB*T, A3=params['A3']*kB*T, A4=params['A4']*kB*T,
                    phi1=params['phi1'], phi2=params['phi2'], phi3=params['phi3'], phi4=params['phi4'],
                    q_A1=params['q_A1']*kB*T, q_A2=params['q_A2']*kB*T, q_A3=params['q_A3']*kB*T, q_A4=params['q_A4']*kB*T,
                    q_phi1=params['q_phi1'], q_phi2=params['q_phi2'], q_phi3=params['q_phi3'], q_phi4=params['q_phi4'],
                    linear_potentials=linear_potentials, q_linear_potentials=q_linear_potentials, L=params['L'],
                    low=params['low'], high=params['high'], epsilon=params['epsilon']*kB*T,
                    sigma=params['sigma'], cutoff=params['cutoff']
                )
            elif ext_type.startswith("TrainingPotentialWithChargeCos"):
                kB = config['kB']
                T = config['T']
                linear_potentials = [
                    (params[f'Va{i}']*kB*T, params[f'Vb{i}']*kB*T, params[f'xa{i}'], params[f'xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                q_linear_potentials = [
                    (params[f'q_Va{i}']*kB*T, params[f'q_Vb{i}']*kB*T, params[f'q_xa{i}'], params[f'q_xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                external_potentials_dict[type] = TrainingPotentialWithChargeCos(
                    q=params['q'], A1=params['A1']*kB*T, A2=params['A2']*kB*T, A3=params['A3']*kB*T, A4=params['A4']*kB*T,
                    phi1=params['phi1'], phi2=params['phi2'], phi3=params['phi3'], phi4=params['phi4'],
                    q_A1=params['q_A1']*kB*T, q_A2=params['q_A2']*kB*T, q_A3=params['q_A3']*kB*T, q_A4=params['q_A4']*kB*T,
                    q_phi1=params['q_phi1'], q_phi2=params['q_phi2'], q_phi3=params['q_phi3'], q_phi4=params['q_phi4'],
                    linear_potentials=linear_potentials, q_linear_potentials=q_linear_potentials, L=params['L']
                )
            elif ext_type.startswith("TrainingPotentialWithCharge"):
                kB = config['kB']
                T = config['T']
                linear_potentials = [
                    (params[f'Va{i}']*kB*T, params[f'Vb{i}']*kB*T, params[f'xa{i}'], params[f'xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                q_linear_potentials = [
                    (params[f'q_Va{i}']*kB*T, params[f'q_Vb{i}']*kB*T, params[f'q_xa{i}'], params[f'q_xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                external_potentials_dict[type] = TrainingPotentialWithCharge(
                    q=params['q'], A1=params['A1']*kB*T, A2=params['A2']*kB*T, A3=params['A3']*kB*T, A4=params['A4']*kB*T,
                    phi1=params['phi1'], phi2=params['phi2'], phi3=params['phi3'], phi4=params['phi4'],
                    q_A1=params['q_A1']*kB*T, q_A2=params['q_A2']*kB*T, q_A3=params['q_A3']*kB*T, q_A4=params['q_A4']*kB*T,
                    q_phi1=params['q_phi1'], q_phi2=params['q_phi2'], q_phi3=params['q_phi3'], q_phi4=params['q_phi4'],
                    linear_potentials=linear_potentials, q_linear_potentials=q_linear_potentials, L=params['L']
                )
            elif ext_type.startswith("TrainingPotentialWithWalls"):
                kB = config['kB']
                T = config['T']
                linear_potentials = [
                    (params[f'Va{i}']*kB*T, params[f'Vb{i}']*kB*T, params[f'xa{i}'], params[f'xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                external_potentials_dict[type] = TrainingPotentialWithWalls(
                    A1=params['A1']*kB*T, A2=params['A2']*kB*T, A3=params['A3']*kB*T, A4=params['A4']*kB*T,
                    phi1=params['phi1'], phi2=params['phi2'], phi3=params['phi3'], phi4=params['phi4'],
                    L=params['L'], width=params['width'], linear_potentials=linear_potentials
                )
            elif ext_type.startswith("TrainingPotentialWithLJ93Walls"):
                kB = config['kB']
                T = config['T']
                linear_potentials = [
                    (params[f'Va{i}']*kB*T, params[f'Vb{i}']*kB*T, params[f'xa{i}'], params[f'xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                external_potentials_dict[type] = TrainingPotentialWithLJ93Walls(
                    A1=params['A1']*kB*T, A2=params['A2']*kB*T, A3=params['A3']*kB*T, A4=params['A4']*kB*T,
                    phi1=params['phi1'], phi2=params['phi2'], phi3=params['phi3'], phi4=params['phi4'],
                    L=params['L'], low=params['low'], high=params['high'], epsilon=params['epsilon']*kB*T,
                    sigma=params['sigma'], cutoff=params['cutoff'], linear_potentials=linear_potentials
                )
            elif ext_type.startswith("TrainingPotential"):
                kB = config['kB']
                T = config['T']
                linear_potentials = [
                    (params[f'Va{i}']*kB*T, params[f'Vb{i}']*kB*T, params[f'xa{i}'], params[f'xb{i}'])
                    for i in range(1, int(ext_type[-1]) + 1)
                ]
                external_potentials_dict[type] = TrainingPotential(
                    A1=params['A1']*kB*T, A2=params['A2']*kB*T, A3=params['A3']*kB*T, A4=params['A4']*kB*T,
                    phi1=params['phi1'], phi2=params['phi2'], phi3=params['phi3'], phi4=params['phi4'],
                    L=params['L'], linear_potentials=linear_potentials
                )

        else:
            external_potentials_dict[type] = NoExternalPotential()
    return external_potentials_dict





