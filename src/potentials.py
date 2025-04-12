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
from scipy.special import erfc
import scipy.constants as const
from constants import very_large_number

# Utility functions for unit conversions
def angstroms_to_meters(value):
    return value * 1e-10

def elementary_charge_to_coulombs(value):
    return value * const.elementary_charge

class Potential:
    def calculate(self, r):
        raise NotImplementedError
    
class LennardJonesPotential(Potential):
    def __init__(self, epsilon, sigma, rc):
        self.epsilon = epsilon
        self.sigma = sigma
        self.rc = rc

    def calculate(self, r):
        r6 = (self.sigma / r) ** 6
        r12 = r6 ** 2
        potential = 4 * self.epsilon * (r12 - r6)
        shift = 4 * self.epsilon * ((self.sigma / self.rc) ** 12 - (self.sigma / self.rc) ** 6)
        return np.where(r < self.rc, potential - shift, 0.0)    

class WCAPotential(Potential):
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma
        
    def calculate(self, r):
        r6 = (self.sigma / r) ** 6
        r12 = r6 ** 2
        potential = 4 * self.epsilon * (r12 - r6)
        rc = 2**(1/6) * self.sigma
        shift = 4 * self.epsilon * ((self.sigma / rc) ** 12 - (self.sigma / rc) ** 6)
        potential_cut = np.where(r < rc, potential - shift, 0.0)
        return np.where(r < rc, potential_cut, 0.0)

class HardSpherePotential(Potential):
    def __init__(self, sigma):
        self.sigma = sigma  

    def calculate(self, r):
        return np.where(r < self.sigma, very_large_number, 0.0)  

class OneComponentPlasmaPotential(Potential):
    def __init__(self, epsilon, q, sigma):
        self.epsilon = epsilon 
        self.q = q  
        self.kappa_inv = kappa_inv  
        self.prefactor =  (self.q*const.elementary_charge)**2 / (4 * const.pi * const.epsilon_0 * 1e-10 * self.epsilon )

    def calculate(self, r):
        potential =  self.prefactor * erfc(r/ self.kappa_inv) / r  
        return potential

class HardSphereCoulombPotential(Potential):
    def __init__(self, diameter, epsilon, q1, q2, kappa_inv):
        self.diameter = diameter 
        self.epsilon = epsilon
        self.q1 = q1
        self.q2 = q2
        self.kappa_inv = kappa_inv
        self.prefactor =  (const.elementary_charge)**2 / (4 * const.pi * const.epsilon_0 * 1e-10 * self.epsilon )

    def calculate(self, r):
        return np.where(r < self.diameter, very_large_number, self.prefactor * self.q1 * self.q2 * erfc(r/ self.kappa_inv) / r)  

class LennardJonesCoulombPotential(Potential):
    def __init__(self, epsilon_lj, sigma_lj, rc, epsilon_c, q1, q2, kappa_inv):
        self.epsilon_lj = epsilon_lj * 4184 / const.Avogadro # Convert kcal/mol to J
        self.sigma_lj = sigma_lj 
        self.rc = rc
        self.epsilon_c = epsilon_c
        self.q1 = q1
        self.q2 = q2
        self.kappa_inv = kappa_inv
        self.prefactor =  (const.elementary_charge)**2 / (4 * const.pi * const.epsilon_0 * 1e-10 * self.epsilon_c )
        
    def calculate(self, r):
        r6 = (self.sigma_lj / r) ** 6
        r12 = r6 ** 2
        potential_lj = 4 * self.epsilon_lj * (r12 - r6)
        shift_lj = 4 * self.epsilon_lj * ((self.sigma_lj / self.rc) ** 12 - (self.sigma_lj / self.rc) ** 6)
        potential_c = self.prefactor * self.q1 * self.q2 * erfc(r/ self.kappa_inv) / r
        return np.where(r < self.rc, potential_lj - shift_lj + potential_c, 0.0)

def initialize_potentials(config):
    potential_dict = {}
    
    for pair, params in config['potential_pairs'].items():
        pair_type = params['type']
        if pair_type == 'LJ':
            potential_dict[pair] = LennardJonesPotential(epsilon=params['epsilon'], sigma=params['sigma'], rc=params['rc'])
        elif pair_type == 'WCA':
            potential_dict[pair] = WCAPotential(epsilon=params['epsilon'], sigma=params['sigma'])
        elif pair_type == 'HS':
            potential_dict[pair] = HardSpherePotential(sigma=params['sigma'])
        elif pair_type == 'OCP':
            potential_dict[pair] = OneComponentPlasmaPotential(epsilon=params['epsilon'], q=params['q'], kappa_inv=params['kappa_inv'])
        elif pair_type == 'HS+C':
            potential_dict[pair] = HardSphereCoulombPotential(diameter=params['diameter'], epsilon=params['epsilon'], q1=params['q1'], q2=params['q2'], kappa_inv=params['kappa_inv'])
        elif pair_type == 'LJ+C':
            potential_dict[pair] = LennardJonesCoulombPotential(epsilon_lj=params['epsilon_lj'], sigma_lj=params['sigma_lj'], rc=params['rc'], epsilon_c=params['epsilon_c'], q1=params['q1'], q2=params['q2'], kappa_inv=params['kappa_inv'])
        else:
            raise ValueError(f"Unknown potential type: {pair_type}")
    return potential_dict