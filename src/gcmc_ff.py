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
from collections import Counter
import gzip

class GCMC_FF_SingleType_Simulation:
    def __init__(self, config, potentials, external_potentials, input_folder):
        """
        Initialize the GCMC Simulation with parameters from a configuration dictionary.
        """
        
        initial_config = input_folder +  '/' +  config.get('init_config', 'initial.xyz')
        self.logfile = input_folder + '/' + config.get('logfile', 'gcmc.log')
        self.output_xyz = input_folder + '/' + 'output.xyz'
        
        (key, self.potentials), = potentials.items()
        (key, self.external_potentials), = external_potentials.items()        
        
        self.global_rc = config['global_rc']
        self.T = config['T']
        self.kB = config.get('kB', 1.0)
        self.beta = 1.0 / (self.kB * self.T)
        self.box_length_x = config['box_length_x']
        self.box_length_y = config['box_length_y']
        self.box_length_z = config['box_length_z']
        self.box_length = np.array([self.box_length_x, self.box_length_y, self.box_length_z])
        
        key, value = next(iter(config['particle_types'].items()))
        self.type = key
        self.mu = value['mu'] * self.kB * self.T
            
        self.max_steps = config['max_steps']
        self.equilibration_steps = config.get('equilibration', 1000)
        self.output_interval = config['output_interval']
        self.output_steps = set(range(0, self.max_steps + 1, self.output_interval))
        
        
        self.positions, self.types = self.load_xyz(initial_config)
        self.volume = self.box_length_x * self.box_length_y * self.box_length_z
        self.number = len(self.positions)
        self.type = self.types[0]


        self.weights = config.get('weights', {'insert': 1.0, 'delete': 1.0, 'displace': 1.0})
        total_weight = sum(self.weights.values())
        self.insert_prob = self.weights['insert'] / total_weight
        self.delete_prob = self.weights['delete'] / total_weight
        self.displace_prob = self.weights['displace'] / total_weight
        
        self.maxdispl = config.get('maxdispl', 3.0)
        
    def load_xyz(self, filename):
        """
        Load particle positions from an XYZ file.
        """
        try:
            data = np.genfromtxt(filename, skip_header=2, dtype='str')
            types = data[:, 0].tolist()
            positions = data[:, 1:].astype(float)
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty configuration.")
            types = ['Ar']
            positions = np.array([[0.0, 0.0, 0.0]])
        return np.array(positions), types
        
    def minimum_image(self, pos):
        """
        Apply periodic boundary conditions using the minimum image convention.
        """
        return pos - self.box_length * np.round(pos / self.box_length)
    
    def wrap_pbc(self, pos):
        """
        Apply periodic boundary conditions by wrapping positions back into the box.
        """
        return pos - self.box_length * np.floor(pos / self.box_length)

    def total_energy(self):
        """
        Calculate the total potential energy of the system.
        """
        # Create a matrix of differences between each pair of particles
        rij = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        rij = self.minimum_image(rij)
        r = np.linalg.norm(rij, axis=2)
        np.fill_diagonal(r, np.inf)  # Exclude self-interaction
        within_rc = (r < self.global_rc)
        
        # Calculate potential energies only for pairs within cutoff distance
        energies = np.zeros_like(r)
        energies[within_rc] = self.potentials.calculate(r[within_rc])
        
        # Sum only the upper triangle of the symmetric energy matrix
        upper_triangle_indices = np.triu_indices(self.number, k=1)
        total_pairwise_energy = np.sum(energies[upper_triangle_indices])
        
        # Add external potential energy if applicable
        if self.external_potentials:
            external_energy = self.external_potentials.calculate_multiple(self.positions)
            total_energy = total_pairwise_energy + external_energy
        else:
            total_energy = total_pairwise_energy
        
        return total_energy


    def local_energy(self, pos, positions):
        """
        Calculate the local energy of a particle with respect to all other particles.
        Add external potential energy for the given particle.
        """
        rij = positions - pos
        rij = self.minimum_image(rij)
        r = np.linalg.norm(rij, axis=1)
        within_rc = r < self.global_rc
        r_within_rc = r[within_rc]

        energies = np.zeros_like(r)
        energies[within_rc] = self.potentials.calculate(r_within_rc)
        
        E = np.sum(energies)
        
        if self.external_potentials:
            E += self.external_potentials(pos)
            
        return E

    def gcmc_step(self):
        """
        Perform a GCMC step, attempting either insertion, deletion, or displacement.
        """
        rand_num = np.random.rand()
        if rand_num < self.insert_prob:
            self.insert_particle()
        elif rand_num < self.insert_prob + self.delete_prob:
            self.delete_particle()
        else:
            self.displace_particle()

    def insert_particle(self):
        """
        Attemp to insert a particle
        """
        new_pos = np.random.uniform(0, self.box_length, 3)
        delta_E = self.local_energy(new_pos, self.positions)
        
        #prob = np.exp(-self.beta * delta_E) * self.volume * activity / (N_atoms+1)
        prob = np.exp(-self.beta * (delta_E - self.mu)) * self.volume / (self.number+1)
        if np.random.rand() < prob:
            self.positions = np.vstack([self.positions, new_pos])
            self.number = self.number + 1
           
        
    def delete_particle(self):
        """
        Attemp to delete a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            del_pos = self.positions[idx]
            remain_positions = np.delete(self.positions, idx, axis=0)
            delta_E = -self.local_energy(del_pos, remain_positions)
        
        
            #prob = np.exp(-self.beta * delta_E) * N_atoms / (self.volume * activity)
            #prob = np.exp(-self.beta * (delta_E + self.mu)) * self.number / self.volume 
            
            log_prob = -self.beta * (delta_E + self.mu) + np.log(self.number) - np.log(self.volume)
            if log_prob < np.log(np.finfo(float).max):
                prob = np.exp(log_prob)
            else:
                prob = 0

            if np.random.rand() < prob:
                self.positions = remain_positions
                self.number = self.number - 1           
        
        
    def displace_particle(self):
        """
        Attemp to displace a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            new_pos = old_pos + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy


            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos
                    
                
         
    def write_xyz(self, step):
        """
        Write the positions to an Extended XYZ file format for a given step.
        """
        
        with gzip.open(self.output_xyz + '.gz', 'at') as f:
            cell = f"{self.box_length_x} 0.0 0.0 0.0 {self.box_length_y} 0.0 0.0 0.0 {self.box_length_z}"
            f.write(f"{self.number}\n")
            f.write(f"Step {step} Lattice=\"{cell}\" Properties=species:S:1:pos:R:3\n")
            for i, pos in enumerate(self.positions):
                f.write(f"{self.type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n") 

    def write_xyz_header(self):
        """
        Open a new xyz file.
        """
        
        with gzip.open(self.output_xyz + '.gz', 'wt') as f:
            pass
                        
    def log(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        total_energy = self.total_energy()
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {self.number} {total_energy}\n")
            
    def log_no_energy(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {self.number} \n")

    def write_log_header(self):
        """
        Write the header to the log file.
        """
        with open(self.logfile, 'w') as f:
            f.write("Step Total_number Energy\n")

    def run_simulation(self):
        """
        Run the GCMC simulation for the configured number of steps.
        """
        self.write_log_header()
        self.write_xyz_header()
        
        # Equilibration phase
        for step in range(self.equilibration_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log(step)
            
        # Production phase
        for step in range(self.equilibration_steps, self.max_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log(step)
                self.write_xyz(step)
                
        final_step = self.max_steps
        self.log(final_step)
        self.write_xyz(self.max_steps)

    def run_simulation_no_energy(self):
        """
        Run the GCMC simulation for the configured number of steps.
        """
        self.write_log_header()
        self.write_xyz_header()
        
        # Equilibration phase
        for step in range(self.equilibration_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log(step)
            
        # Production phase
        for step in range(self.equilibration_steps, self.max_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log_no_energy(step)
                self.write_xyz(step)
                
        final_step = self.max_steps
        self.log(final_step)
        self.write_xyz(self.max_steps)


class GCMC_FF_TwoType_Simulation:
    def __init__(self, config, potentials, external_potentials, input_folder):
        """
        Initialize the GCMC Simulation with parameters from a configuration dictionary.
        """

        initial_config = input_folder +  '/' +  config.get('init_config', 'initial.xyz')
        self.logfile = input_folder + '/' + config.get('logfile', 'gcmc.log')
        self.output_xyz = input_folder + '/' + 'output.xyz'
        
   
        self.global_rc = config['global_rc']
        self.T = config['T']
        self.kB = config.get('kB', 1.0)
        self.beta = 1.0 / (self.kB * self.T)
        self.box_length_x = config['box_length_x']
        self.box_length_y = config['box_length_y']
        self.box_length_z = config['box_length_z']
        self.box_length = np.array([self.box_length_x, self.box_length_y, self.box_length_z])
        
        keys = list(config['particle_types'].keys())
        values = list(config['particle_types'].values())
        self.type1, self.type2 = keys[0], keys[1]
        self.mu1 = values[0]['mu'] * self.kB * self.T
        self.mu2 = values[1]['mu'] * self.kB * self.T
     
        
        self.potential_1_1 = potentials[f'{self.type1}_{self.type1}']
        self.potential_2_2 = potentials[f'{self.type2}_{self.type2}']
        type_pair = '_'.join(sorted([self.type1, self.type2]))
        self.potential_1_2 = potentials[type_pair]
        
        self.external_potential_1 = external_potentials[self.type1]
        self.external_potential_2 = external_potentials[self.type2]
        
                
        self.max_steps = config['max_steps']
        self.equilibration_steps = config.get('equilibration', 1000)
        self.output_interval = config['output_interval']
        self.output_steps = set(range(0, self.max_steps + 1, self.output_interval))
        
        self.volume = self.box_length_x * self.box_length_y * self.box_length_z
        
        self.positions_1, self.positions_2 = self.load_xyz(initial_config)
        
        self.number1 = len(self.positions_1)
        self.number2 = len(self.positions_2)
        
        self.weights = config.get('weights', {'insert': 1.0, 'delete': 1.0, 'displace': 1.0, 'mutate': 0.0})
        total_weight = sum(self.weights.values())
        self.insert_prob = self.weights['insert'] / total_weight
        self.delete_prob = self.weights['delete'] / total_weight
        self.displace_prob = self.weights['displace'] / total_weight
        self.mutate_prob = self.weights['mutate'] / total_weight
        
        self.swap_prob = config.get('swap_weights', 0.1)
        self.selection_prob = config.get('selection_weight', (1 - self.swap_prob)*0.5)
        
        self.maxdispl = config.get('maxdispl', 3.0)
    
    def load_xyz(self, filename):
        """
        Load particle positions from an XYZ file.
        """
        try:
            data = np.genfromtxt(filename, skip_header=2, dtype='str')
            types = data[:, 0].tolist()
            positions = data[:, 1:].astype(float)
            positions_1 = []
            positions_2 = []
            unique_types = np.unique(types)
            if len(unique_types) != 2:
                raise ValueError("Expected exactly two distinct particle types.")
            
            for type_, pos in zip(types, positions):
                if type_ == unique_types[0]:
                    positions_1.append(pos)
                else:
                    positions_2.append(pos)
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty configuration.")
            positions_1, positions_2 = [], []
        return np.array(positions_1), np.array(positions_2)
        
    def minimum_image(self, pos):
        """
        Apply periodic boundary conditions using the minimum image convention.
        """
        return pos - self.box_length * np.round(pos / self.box_length)
    
    def wrap_pbc(self, pos):
        """
        Apply periodic boundary conditions by wrapping positions back into the box.
        """
        return pos - self.box_length * np.floor(pos / self.box_length)
    
    
    def total_energy(self):
        """
        Calculate the total potential energy of the system with two types of particles.
        """
        positions_1 = self.positions_1
        positions_2 = self.positions_2
      
        global_rc = self.global_rc


        # Helper function to calculate pairwise energy for given positions and potential
        def calculate_pairwise_energy(positions1, positions2, potential, exclude_self=True):
            deltas = positions1[:, np.newaxis, :] - positions2[np.newaxis, :, :]
            deltas = self.minimum_image(deltas)  # Apply periodic boundary conditions
            distances = np.linalg.norm(deltas, axis=2)
            np.fill_diagonal(distances, np.inf) # Exclude self-interaction
            within_rc = (distances < global_rc)
            energies = np.zeros_like(distances)
            energies[within_rc] = potential.calculate(distances[within_rc])
            
            if exclude_self:
                return np.sum(np.triu(energies, k=1))
            else:
                return np.sum(energies)

        total_pairwise_energy = 0.0
        
        if self.number1 > 0:
            total_pairwise_energy += calculate_pairwise_energy(positions_1, positions_1, self.potential_1_1)

        if self.number2 > 0:
            total_pairwise_energy += calculate_pairwise_energy(positions_2, positions_2, self.potential_2_2)
            
        if self.number1 > 0 and self.number2 > 0:
            total_pairwise_energy += calculate_pairwise_energy(positions_1, positions_2, self.potential_1_2, exclude_self=False)
            
            
        # Add external potential energy if applicable
        external_energy_1 = np.sum(self.external_potential_1.calculate_multiple(positions_1)) if self.external_potential_1 else 0.0
        external_energy_2 = np.sum(self.external_potential_2.calculate_multiple(positions_2)) if self.external_potential_2 else 0.0
        total_external_energy = external_energy_1 + external_energy_2

        total_energy = total_pairwise_energy + total_external_energy
        return total_energy
    
    def local_energy_1(self, pos1, positions_1, positions_2):
        """
        Calculate the local energy of a particle type 1 with respect to all other particles.
        Add external potential energy for the given particle.
        """
        E = 0.0

        # Helper function to calculate pairwise energy
        def calculate_energy(ref_pos, other_positions, potential):
            deltas = other_positions - ref_pos
            deltas = self.minimum_image(deltas)
            distances = np.linalg.norm(deltas, axis=1)
            within_rc = distances < self.global_rc
            return np.sum(potential.calculate(distances[within_rc]))

        # Calculate pair energies with type 1 particles
        if self.number1 > 0:
            E += calculate_energy(pos1, positions_1, self.potential_1_1)
            
        # Calculate pair energies with type 2 particles
        if self.number2 > 0:
            E += calculate_energy(pos1, positions_2, self.potential_1_2)

        # Add external potential energy for type 1 particle
        if self.external_potential_1:
            E += self.external_potential_1(pos1)
        return E

    def local_energy_2(self, pos2, positions_2, positions_1):
        """
        Calculate the local energy of a particle type 2 with respect to all other particles.
        Add external potential energy for the given particle.
        """
        E = 0.0


        # Helper function to calculate pairwise energy
        def calculate_energy(ref_pos, other_positions, potential):
            deltas = other_positions - ref_pos
            deltas = self.minimum_image(deltas)
            distances = np.linalg.norm(deltas, axis=1)
            within_rc = distances < self.global_rc
            return np.sum(potential.calculate(distances[within_rc]))
        
        # Calculate pair energies with type 2 particles
        if self.number2 > 0:
            E += calculate_energy(pos2, positions_2, self.potential_2_2)
            
        # Calculate pair energies with type 1 particles
        if self.number1 > 0:
            E += calculate_energy(pos2, positions_1, self.potential_1_2)
            
        # Add external potential energy for type 2 particle
        if self.external_potential_2:
            E += self.external_potential_2(pos2)
            
        return E
    
    def gcmc_step(self):
        """
        Perform a GCMC step, attempting either insertion, deletion, or displacement.
        """
        rand_num1 = np.random.rand()
        rand_num2 = np.random.rand()
        if rand_num1 < self.swap_prob:
            self.swap_particle_1_2()
        elif rand_num1 < self.swap_prob + self.selection_prob:
            if rand_num2 < self.insert_prob:
                self.insert_particle_1()
            elif rand_num2 < self.insert_prob + self.delete_prob:
                self.delete_particle_1()
            elif rand_num2 < self.insert_prob + self.delete_prob + self.mutate_prob:
                self.mutate_particle_1()
            else:
                self.displace_particle_1()
        else:
            if rand_num2 < self.insert_prob:
                self.insert_particle_2()
            elif rand_num2 < self.insert_prob + self.delete_prob:
                self.delete_particle_2()
            elif rand_num2 < self.insert_prob + self.delete_prob + self.mutate_prob:
                self.mutate_particle_2()
            else:
                self.displace_particle_2()  

    def insert_particle_1(self):
        """
        Attemp to insert a particle type 1
        """
        new_pos = np.random.uniform(0, self.box_length, 3)
        delta_E = self.local_energy_1(new_pos, self.positions_1, self.positions_2)
        prob = np.exp(-self.beta * (delta_E - self.mu1)) * self.volume / (self.number1+1)
        if np.random.rand() < prob:
            self.positions_1 = np.vstack([self.positions_1, new_pos])
            self.number1 = self.number1 + 1

    def insert_particle_2(self):
        """
        Attemp to insert a particle type 2
        """
        new_pos = np.random.uniform(0, self.box_length, 3)
        delta_E = self.local_energy_2(new_pos, self.positions_2, self.positions_1)
        prob = np.exp(-self.beta * (delta_E - self.mu2)) * self.volume / (self.number2+1)
        if np.random.rand() < prob:
            self.positions_2 = np.vstack([self.positions_2, new_pos])     
            self.number2 = self.number2 + 1    
           
    def delete_particle_1(self):
        """
        Attemp to delete a particle type 1
        """
        if self.number1 > 0:
            idx = np.random.randint(0, self.number1)
            del_pos = self.positions_1[idx]
            remain_positions_1 = np.delete(self.positions_1, idx, axis=0)
            delta_E = -self.local_energy_1(del_pos, remain_positions_1, self.positions_2)
            prob = np.exp(-self.beta * (delta_E + self.mu1)) * self.number1 / self.volume 
            if np.random.rand() < prob:
                self.positions_1 = remain_positions_1
                self.number1 = self.number1 - 1
        
    def delete_particle_2(self):
        """
        Attemp to delete a particle type 2
        """
        if self.number2 > 0:
            idx = np.random.randint(0, self.number2)
            del_pos = self.positions_2[idx]
            remain_positions_2 = np.delete(self.positions_2, idx, axis=0)
            delta_E = -self.local_energy_2(del_pos, remain_positions_2, self.positions_1)
            prob = np.exp(-self.beta * (delta_E + self.mu2)) * self.number2 / self.volume 
            if np.random.rand() < prob:
                self.positions_2 = remain_positions_2
                self.number2 = self.number2 - 1    
        
    def displace_particle_1(self):
        """
        Attemp to displace a particle of type 1
        """
        if self.number1 > 0:
            idx = np.random.randint(0, self.number1)
            old_pos = self.positions_1[idx]
            new_pos = old_pos + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            remain_positions_1 = np.delete(self.positions_1, idx, axis=0)
            old_energy = self.local_energy_1(old_pos, remain_positions_1, self.positions_2)
            new_energy = self.local_energy_1(new_pos, remain_positions_1, self.positions_2)
            delta_E = new_energy - old_energy
            if np.random.rand() < np.exp(-self.beta * delta_E):
                self.positions_1[idx] = new_pos
                
    def displace_particle_2(self):
        """
        Attemp to displace a particle of type 2
        """
        if self.number2 > 0:
            idx = np.random.randint(0, self.number2)
            old_pos = self.positions_2[idx]
            new_pos = old_pos + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            remain_positions_2 = np.delete(self.positions_2, idx, axis=0)
            old_energy = self.local_energy_2(old_pos, remain_positions_2, self.positions_1)
            new_energy = self.local_energy_2(new_pos, remain_positions_2, self.positions_1)
            delta_E = new_energy - old_energy
            if np.random.rand() < np.exp(-self.beta * delta_E):
                self.positions_2[idx] = new_pos


    def swap_particle_1_2(self):
        """
        Attemp to swap a particle of type 1 with that of type 2
        Acceptance probability:
        p = exp(-beta * (delta_E - mu2 + mu1))
        """
        if self.number2 > 0 and self.number1 > 0:
            idx1 = np.random.randint(0, self.number1)
            idx2 = np.random.randint(0, self.number2)
            old_pos1 = self.positions_1[idx1]
            old_pos2 = self.positions_2[idx2]
            remain_positions_1 = np.delete(self.positions_1, idx1, axis=0)
            remain_positions_2 = np.delete(self.positions_2, idx2, axis=0)
            new_pos1 = self.wrap_pbc(old_pos2)
            new_pos2 = self.wrap_pbc(old_pos1)
            old_energy = self.local_energy_1(old_pos1, remain_positions_1, remain_positions_2)
            old_energy += self.local_energy_2(old_pos2, remain_positions_2, remain_positions_1)
            new_energy = self.local_energy_1(new_pos1, remain_positions_1, remain_positions_2)
            new_energy += self.local_energy_2(new_pos2, remain_positions_2, remain_positions_1)
            delta_E = new_energy - old_energy
            prob = np.exp(-self.beta * delta_E) 
            if np.random.rand() < prob:
                self.positions_2[idx2] = new_pos2
                self.positions_1[idx1] = new_pos1

    def mutate_particle_1(self):
        """
        Attemp to mutate particle of type 1 to type 2
        Acceptance probability:
        p = exp(-beta * (delta_E - mu2 + mu1)) * N1 / (N2 + 1)
        """
        if self.number1 > 0:
            idx1 = np.random.randint(0, self.number1)
            old_pos = self.positions_1[idx1]
            remain_positions_1 = np.delete(self.positions_1, idx1, axis=0)
            
            old_energy = self.local_energy_1(old_pos, remain_positions_1, self.positions_2)
            new_energy = self.local_energy_2(old_pos, self.positions_2, remain_positions_1)
            
            delta_E = new_energy - old_energy
            prob = np.exp(-self.beta * (delta_E - self.mu2 + self.mu1 )) * self.number1/(self.number2+1)
            if np.random.rand() < prob:
                self.positions_1 = remain_positions_1
                self.number1 = self.number1 - 1
                self.positions_2 = np.vstack([self.positions_2, old_pos])     
                self.number2 = self.number2 + 1 

    def mutate_particle_2(self):
        """
        Attemp to mutate particle of type 2 to type 1
        Acceptance probability:
        p = exp(-beta * (delta_E - mu1 + mu2)) * N2 / (N1 + 1)
        """
        if self.number2 > 0:
            idx2 = np.random.randint(0, self.number2)
            old_pos = self.positions_2[idx2]
            remain_positions_2 = np.delete(self.positions_2, idx2, axis=0)
            
            old_energy = self.local_energy_2(old_pos, remain_positions_2, self.positions_1)
            new_energy = self.local_energy_1(old_pos, self.positions_1, remain_positions_2)
            
            delta_E = new_energy - old_energy
            prob = np.exp(-self.beta * (delta_E - self.mu1 + self.mu2 )) * self.number2/(self.number1+1)
            if np.random.rand() < prob:
                self.positions_2 = remain_positions_2
                self.number2 = self.number2 - 1
                self.positions_1 = np.vstack([self.positions_1, old_pos])     
                self.number1 = self.number1 + 1 


    def write_xyz_header(self):
        """
        Open a new xyz file.
        """
        with gzip.open(self.output_xyz + '.gz', 'wt') as f:
            pass

    def write_xyz(self, step):
        """
        Write the positions to an Extended XYZ file format for a given step.
        """
        with gzip.open(self.output_xyz + '.gz', 'at') as f:
            num_atoms = self.number1 + self.number2
            cell = f"{self.box_length_x} 0.0 0.0 0.0 {self.box_length_y} 0.0 0.0 0.0 {self.box_length_z}"
            f.write(f"{num_atoms}\n")
            f.write(f"Step {step} Lattice=\"{cell}\" Properties=species:S:1:pos:R:3\n")
            for pos in self.positions_1:
                f.write(f"{self.type1} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n") 
            for pos in self.positions_2:
                f.write(f"{self.type2} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n") 
            

    def log(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        num_particles = self.number1 + self.number2
        total_energy = self.total_energy()
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {num_particles} {self.number1} {self.number2} {total_energy}\n")

    def log_no_energy(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        num_particles = self.number1 + self.number2
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {num_particles} {self.number1} {self.number2}\n")

    def write_log_header(self):
        """
        Write the header to the log file.
        """
        with open(self.logfile, 'w') as f:
            f.write("Step Total_number " + f"{self.type1} {self.type2}\n")

    def run_simulation(self):
        """
        Run the GCMC simulation for the configured number of steps.
        """
        self.write_log_header()
        self.write_xyz_header()
        
        # Equilibration phase
        for step in range(self.equilibration_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log(step)
            
        # Production phase
        for step in range(self.equilibration_steps, self.max_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log(step)
                self.write_xyz(step)
                
        final_step = self.max_steps
        self.log(final_step)
        self.write_xyz(self.max_steps)


    def run_simulation_no_energy(self):
        """
        Run the GCMC simulation for the configured number of steps, without printing energy
        """
        self.write_log_header()
        self.write_xyz_header()
        
        # Equilibration phase
        for step in range(self.equilibration_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log_no_energy(step)


        # Production phase
        for step in range(self.equilibration_steps, self.max_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log_no_energy(step)
                self.write_xyz(step)
                
                
        final_step = self.max_steps
        self.log(final_step)
        self.write_xyz(self.max_steps)
        
class GCMC_FF_MultiType_Simulation:
    def __init__(self, config, potentials, external_potentials, input_folder):
        """
        Initialize the GCMC Simulation with parameters from a configuration dictionary.
        """
        
        initial_config = input_folder +  '/' +  config.get('init_config', 'initial.xyz')
        self.logfile = input_folder + '/' + config.get('logfile', 'gcmc.log')
        self.output_xyz = input_folder + '/' + 'output.xyz'
        
        self.potentials = potentials
        self.external_potentials = external_potentials
        self.global_rc = config['global_rc']
        self.T = config['T']
        self.kB = config.get('kB', 1.0)
        self.beta = 1.0 / (self.kB * self.T)
        self.box_length_x = config['box_length_x']
        self.box_length_y = config['box_length_y']
        self.box_length_z = config['box_length_z']
        self.box_length = np.array([self.box_length_x, self.box_length_y, self.box_length_z])
        
        self.mu = {k: v['mu']* self.kB * self.T for k, v in config['particle_types'].items()}
        self.max_steps = config['max_steps']
        self.output_interval = config['output_interval']
        self.positions, self.types = self.load_xyz(initial_config)
        self.volume = self.box_length_x * self.box_length_y * self.box_length_z
        self.number = len(self.positions)


        self.weights = config.get('weights', {'insert': 1.0, 'delete': 1.0, 'displace': 1.0})
        total_weight = sum(self.weights.values())
        self.insert_prob = self.weights['insert'] / total_weight
        self.delete_prob = self.weights['delete'] / total_weight
        self.displace_prob = self.weights['displace'] / total_weight
        
        self.maxdispl = config.get('maxdispl', 3.0)

    def load_xyz(self, filename):
        """
        Load particle positions from an XYZ file.
        """
        try:
            data = np.genfromtxt(filename, skip_header=2, dtype='str')
            types = data[:, 0].tolist()
            positions = data[:, 1:].astype(float)
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty configuration.")
        return np.array(positions), types
        
    def minimum_image(self, pos):
        """
        Apply periodic boundary conditions using the minimum image convention.
        """
        return pos - self.box_length * np.round(pos / self.box_length)
    
    def wrap_pbc(self, pos):
        """
        Apply periodic boundary conditions by wrapping positions back into the box.
        """
        return pos - self.box_length * np.floor(pos / self.box_length)
    
    def total_energy(self, positions, types):
        """
        Calculate the total energy of the system. This is not used for anything yet.
        """
        E = 0.0
        N = len(positions)
        r = np.zeros((N, N))
        
        for i in range(N-1):
            rij = positions[i+1:] - positions[i]
            rij = self.minimum_image(rij)
            r[i, i+1:] = np.linalg.norm(rij, axis=1)
        
        r += r.T

        for i in range(N-1):
            for j in range(i+1, N):
                type_pair = f"{types[i]}_{types[j]}"
                E += self.potentials[type_pair].calculate(r[i, j])
        
        return E
        
    def pair_energy(self, pos1, type1, pos2, type2):
        """
        Calculate the pair energy between two particles.
        """
        rij = pos1 - pos2
        rij = self.minimum_image(rij)
        r = np.linalg.norm(rij)
        type_pair = '_'.join(sorted([type1, type2]))
        return self.potentials[type_pair].calculate(r) if r < self.global_rc else 0.0


    def local_energy(self, pos, type, positions, types):
        """
        Calculate the local energy of a particle with respect to all other particles.
        Add external potential energy for the given particle.
        """
        E = 0.0
        for i in range(len(positions)):
            E += self.pair_energy(pos, type, positions[i], types[i])

        
        if type in self.external_potentials:
            E += self.external_potentials[type](pos)
            
        return E

    def gcmc_step(self):
        """
        Perform a GCMC step, attempting either insertion, deletion, or displacement.
        """
        rand_num = np.random.rand()
        if rand_num < self.insert_prob:
            self.insert_particle()
        elif rand_num < self.insert_prob + self.delete_prob:
            self.delete_particle()
        else:
            self.displace_particle()

    def insert_particle(self):
        """
        Attemp to insert a particle
        """
        new_pos = np.random.uniform(0, self.box_length, 3)
        new_type = np.random.choice(list(self.mu.keys()))
        delta_E = self.local_energy(new_pos, new_type, self.positions, self.types)
        N_atoms = self.types.count(new_type)
        #prob = np.exp(-self.beta * delta_E) * self.volume * activity / (N_atoms+1)
        prob = np.exp(-self.beta * (delta_E - self.mu[new_type])) * self.volume / (N_atoms+1)
        if np.random.rand() < prob:
            self.positions = np.vstack([self.positions, new_pos])
            self.types = self.types + [new_type]
           
        
    def delete_particle(self):
        """
        Attemp to delete a particle
        """
        if self.positions.size > 0:
            idx = np.random.randint(0, self.number)
            del_pos = self.positions[idx]
            del_type = self.types[idx]
            remain_positions = np.delete(self.positions, idx, axis=0)
            remain_types = self.types[:idx] + self.types[idx+1:]
            delta_E = -self.local_energy(del_pos, del_type, remain_positions, remain_types)
            N_atoms = self.types.count(del_type)
            #prob = np.exp(-self.beta * delta_E) * N_atoms / (self.volume * activity)
            prob = np.exp(-self.beta * (delta_E + self.mu[del_type])) * N_atoms / self.volume 
            if np.random.rand() < prob:
                self.positions = remain_positions
                self.types = remain_types            
        
        
        
    def displace_particle(self):
        """
        Attemp to displace a particle
        """
        if self.positions.size > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            new_pos = old_pos + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            dis_type = self.types[idx]
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, dis_type, remain_positions, self.types)
            new_energy = self.local_energy(new_pos, dis_type, remain_positions, self.types)
            delta_E = new_energy - old_energy
            if np.random.rand() < np.exp(-self.beta * delta_E):
                self.positions[idx] = new_pos
                
    def write_xyz(self, step):
        """
        Write the positions to an Extended XYZ file format for a given step.
        """
        with gzip.open(self.output_xyz + '.gz', 'at') as f:
            num_atoms = self.number
            cell = f"{self.box_length_x} 0.0 0.0 0.0 {self.box_length_y} 0.0 0.0 0.0 {self.box_length_z}"
            f.write(f"{num_atoms}\n")
            f.write(f"Step {step} Lattice=\"{cell}\" Properties=species:S:1:pos:R:3\n")
            for i, pos in enumerate(self.positions):
                f.write(f"{self.types[i]} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n") 
            
    def log(self, step, num_particles):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        type_counts = Counter(self.types)
        type_counts_str = ' '.join([f"{count}" for type, count in type_counts.items()])
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {num_particles} {type_counts_str}\n")

    def write_log_header(self):
        """
        Write the header to the log file.
        """
        type_counts = Counter(self.types)
        type_counts_str = ' '.join([f"{type}" for type, count in type_counts.items()])
        with open(self.logfile, 'w') as f:
            f.write("Step Total_number " + f"{type_counts_str}\n")

    def write_xyz_header(self):
        """
        Open a new xyz file.
        """
        with gzip.open(self.output_xyz + '.gz', 'wt') as f:
            pass

    def run_simulation(self):
        """
        Run the GCMC simulation for the configured number of steps.
        """
        self.write_log_header()
        self.write_xyz_header()
        
        
        # Equilibration phase
        for step in range(self.max_steps):
            self.gcmc_step()
            if step % self.output_interval == 0:
                num_particles = self.number
                self.log(step, num_particles)
                self.write_xyz(step)
                
        final_step = self.max_steps
        final_num_particles = self.number
        self.log(final_step, final_num_particles)
        self.write_xyz(self.max_steps)
