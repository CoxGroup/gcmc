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

import scipy.constants as const


ev_to_kjmol = const.electron_volt * const.Avogadro / 1000
kB_kj_per_mol_k = const.Boltzmann * const.Avogadro / 1000
kB_eV_per_k = const.Boltzmann / const.electron_volt	
very_large_number = 1.0e30