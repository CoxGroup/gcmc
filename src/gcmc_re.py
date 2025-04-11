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
import gzip
import mpi4py.MPI as MPI
import argparse
import read_input
import external_potentials 
import potentials
from gcmc_ff import GCMC_FF_SingleType_Simulation, GCMC_FF_TwoType_Simulation


class GCMC_FF_SingleType_Replica(GCMC_FF_SingleType_Simulation):
    def __init__(self, config, potentials, external_potentials, input_folder, temperature, mu):
        super().__init__(config, potentials, external_potentials, input_folder)
        self.T = temperature
        self.beta = 1.0 / (self.kB * self.T)
        self.mu = mu * self.kB * self.T
     

class GCMC_FF_TwoType_Replica(GCMC_FF_TwoType_Simulation):
    def __init__(self, config, potentials, external_potentials, input_folder, temperature, mu1, mu2):
        super().__init__(config, potentials, external_potentials, input_folder)
        self.T = temperature
        self.beta = 1.0 / (self.kB * self.T)
        self.mu1 = mu1 * self.kB * self.T
        self.mu2 = mu2 * self.kB * self.T
  

def select_partner(rank, size, swap_type):
    if swap_type:
        if rank % 2 == 0:
            partner = rank + 1
        else:
            partner = rank - 1
    else:
        if rank == size - 1:
            partner = 0
        elif rank == 0:
            partner = size - 1
        elif rank % 2 == 0:
            partner = rank - 1
        else:
            partner = rank + 1
    return partner

def single_type_exchange_probability(partner_beta, partner_energy, partner_number, partner_mu, 
                                    replica_beta, total_energy, replica_number, replica_mu):
    delta_beta = partner_beta - replica_beta
    delta_betamu = partner_mu * partner_beta - replica_mu * replica_beta
    delta_energy = partner_energy - total_energy
    delta_number = partner_number - replica_number
    prob = np.exp(delta_beta * delta_energy - delta_betamu * delta_number)
    return prob


def two_type_exchange_probability(partner_beta, partner_energy, partner_number1, partner_number2, partner_mu1, partner_mu2, 
                                    replica_beta, total_energy, replica_number1, replica_number2, replica_mu1, replica_mu2):
    delta_beta = partner_beta - replica_beta
    delta_betamu1 = partner_mu1 * partner_beta - replica_mu1 * replica_beta
    delta_betamu2 = partner_mu2 * partner_beta - replica_mu2 * replica_beta
    delta_energy = partner_energy - total_energy
    delta_number1 = partner_number1 - replica_number1
    delta_number2 = partner_number2 - replica_number2
    prob = np.exp(delta_beta * delta_energy - delta_betamu1 * delta_number1 - delta_betamu2 * delta_number2)
    return prob

def run_parallel_single_type(config, input_folder):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    assert size % 2 == 0, "Number of MPI processes must be even."
    
    config = read_input.load_config(input_folder + '/' + 'input.yaml')    
    ext_potentials = external_potentials.initialize_external_potentials(config)
    pair_potentials = potentials.initialize_potentials(config)
    
    replicas = config['replicas']
    init_T = replicas[rank]['T']
    initial_mu = replicas[rank]['mu']
    
    print(f"Rank {rank}, T = {init_T}, beta_mu = {initial_mu}")
    
    replica = GCMC_FF_SingleType_Replica(config, pair_potentials, ext_potentials, input_folder, init_T, initial_mu)
    
    logname = f"gcmc_T_{replica.T:.2f}_mu_{initial_mu:.2f}.log"
    xyzname = f"output_T_{replica.T:.2f}_mu_{initial_mu:.2f}.xyz"
    with open(logname, "w") as f:
        f.write(f"Step Number\n")
    with gzip.open(xyzname + '.gz', 'wt') as f:
            pass
    
    
    max_steps = config['max_steps']
    output_interval = config['output_interval']
    exchange_frequency = config['exchange_frequency']
    
    
    for step in range(max_steps):
        
        replica.gcmc_step()
        
        if rank == 0:
            attempt_swap = np.random.rand() < exchange_frequency
        else:
            attempt_swap = None
            
        attempt_swap = comm.bcast(attempt_swap, root=0)
        
        if attempt_swap:
            
            if rank == 0:
                rand_num = np.random.randint(0, 2) 
            else:
                rand_num = None
                
            rand_num = comm.bcast(rand_num, root=0)
            swap_type = rand_num == 1
               
            partner = select_partner(rank, size, swap_type)
            
            comm.Barrier()
            
            # determine criteria
            total_energy = replica.total_energy()
            send_data = (replica.T, total_energy, replica.number, replica.mu)
            comm.send(send_data, dest=partner, tag=1)
            rev_data = comm.recv(source=partner, tag=1)
            
            partner_T, partner_energy, partner_number, partner_mu = rev_data
            partner_beta, replica_beta = 1.0 / (partner_T * replica.kB), 1.0 / (replica.T * replica.kB)
            prob = single_type_exchange_probability(partner_beta, partner_energy, partner_number, partner_mu, 
                                                  replica_beta, total_energy, replica.number, replica.mu)
            
            if rank < partner:
                swap_decision = np.random.rand() < prob
                comm.send(swap_decision, dest=partner, tag=4)
            else:
                swap_decision = comm.recv(source=partner, tag=4)
                comm.send(swap_decision, dest=partner, tag=4)
                
            
            if swap_decision:
                # Perform the swap using blocking send and receive
                if (rank % 2 == 0 and swap_type) or (rank % 2 == 1 and not swap_type):
                    comm.send((replica.T, replica.mu), dest=partner, tag=0)
                    new_temperature, new_mu = comm.recv(source=partner, tag=0)
                else:
                    new_temperature, new_mu = comm.recv(source=partner, tag=0)
                    comm.send((replica.T, replica.mu), dest=partner, tag=0)
                    
                #logname = f"gcmc_T_{replica.T:.2f}_mu_{initial_mu:.2f}.log"
                #with open(logname, "a") as f:
                    #f.write(f"Successful swap from {partner_T}\n")
                    
                replica.T, replica.mu = new_temperature,  new_mu
            
            #logname = f"gcmc_T_{replica.T:.2f}_mu_{initial_mu:.2f}.log"
            #with open(logname, "a") as f:
                #f.write(f"Unsuccessful swap from {partner_T}\n")
                    
                     
            # Synchronize after swapping
            #comm.Barrier()
        
        if step % output_interval == 0:
            beta_mu = replica.mu / (replica.T * replica.kB)
            logname = f"gcmc_T_{replica.T:.2f}_mu_{beta_mu:.2f}.log"
            with open(logname, "a") as f:
                f.write(f"{step} {replica.number}\n")
            xyzname = f"output_T_{replica.T:.2f}_mu_{beta_mu:.2f}.xyz"
            with gzip.open(xyzname + '.gz', 'at') as f:
                cell = f"{replica.box_length} 0.0 0.0 0.0 {replica.box_length} 0.0 0.0 0.0 {replica.box_length}"
                f.write(f"{replica.number}\n")
                f.write(f"Step {step} Lattice=\"{cell}\" Properties=species:S:1:pos:R:3\n")
                for i, pos in enumerate(replica.positions):
                    f.write(f"{replica.type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n") 

            
                
def run_parallel_two_type(config, input_folder):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    assert size % 2 == 0, "Number of MPI processes must be even."
    
    config = read_input.load_config(input_folder + '/' + 'input.yaml')    
    ext_potentials = external_potentials.initialize_external_potentials(config)
    pair_potentials = potentials.initialize_potentials(config)
    
    replicas = config['replicas']
    temperatures = [replica['T'] for replica in replicas]
    mus1 = [replica['mu1'] for replica in replicas]
    mus2 = [replica['mu2'] for replica in replicas]
    init_T = temperatures[rank]
    init_mu1 = mus1[rank]
    init_mu2 = mus2[rank]
    
    print(f"Rank {rank}, T = {init_T}, beta_mu1 = {init_mu1}, beta_mu2 = {init_mu2}")
    
    replica = GCMC_FF_TwoType_Replica(config, pair_potentials, ext_potentials, input_folder, init_T, init_mu1, init_mu2)
    
    logname = f"gcmc_T_{replica.T:.2f}_mu_{init_mu1:.2f}_{init_mu2:.2f}.log"
    xyzname = f"output_T_{replica.T:.2f}_mu_{init_mu1:.2f}_{init_mu2:.2f}.xyz"
    with open(logname, "w") as f:
        f.write("Step Total_number " + f"{replica.type1} {replica.type2}\n")
        
    with gzip.open(xyzname + '.gz', 'wt') as f:
        pass

    max_steps = config['max_steps']
    output_interval = config['output_interval']
    exchange_frequency = config['exchange_frequency']
    
    for step in range(max_steps):
        
        replica.gcmc_step()
        
        if rank == 0:
            attempt_swap = np.random.rand() < exchange_frequency
        else:
            attempt_swap = None
            
        attempt_swap = comm.bcast(attempt_swap, root=0)
        
        if attempt_swap:
            
            if rank == 0:
                rand_num = np.random.randint(0, 2) 
            else:
                rand_num = None
                
            rand_num = comm.bcast(rand_num, root=0)
            swap_type = rand_num == 1
               
            partner = select_partner(rank, size, swap_type)
            
            comm.Barrier()
            
            # determine criteria
            total_energy = replica.total_energy()
            send_data = (replica.T, total_energy, replica.number1, replica.number2, replica.mu1, replica.mu2)
            comm.send(send_data, dest=partner, tag=1)
            rev_data = comm.recv(source=partner, tag=1)
            
            partner_T, partner_energy, partner_number1, partner_number2, partner_mu1, partner_mu2  = rev_data
            partner_beta, replica_beta = 1.0 / (partner_T * replica.kB), 1.0 / (replica.T * replica.kB)
            prob = two_type_exchange_probability(partner_beta, partner_energy, partner_number1, partner_number2, partner_mu1, partner_mu2, 
                                                  replica_beta, total_energy, replica.number1, replica.number2, replica.mu1, replica.mu2)
            
            if rank < partner:
                swap_decision = np.random.rand() < prob
                comm.send(swap_decision, dest=partner, tag=4)
            else:
                swap_decision = comm.recv(source=partner, tag=4)
                comm.send(swap_decision, dest=partner, tag=4)
                
            
            if swap_decision:
                # Perform the swap using blocking send and receive
                if (rank % 2 == 0 and swap_type) or (rank % 2 == 1 and not swap_type):
                    comm.send((replica.T, replica.mu1, replica.mu2), dest=partner, tag=0)
                    new_temperature, new_mu1, new_mu2 = comm.recv(source=partner, tag=0)
                else:
                    new_temperature, new_mu1, new_mu2 = comm.recv(source=partner, tag=0)
                    comm.send((replica.T, replica.mu1, replica.mu2), dest=partner, tag=0)
                    
                    
                replica.T, replica.mu1, replica.mu2 = new_temperature, new_mu1, new_mu2
                        
            # Synchronize after swapping
            #comm.Barrier()
        
        if step % output_interval == 0:
            beta_mu1 = replica.mu1 / (replica.T * replica.kB)
            beta_mu2 = replica.mu2 / (replica.T * replica.kB)
            number  = replica.number1 + replica.number2
            
            
            logname = f"gcmc_T_{replica.T:.2f}_mu_{beta_mu1:.2f}_{beta_mu2:.2f}.log"
            with open(logname, "a") as f:
                f.write(f"{step} {number} {replica.number1} {replica.number2}\n")
            xyzname = f"output_T_{replica.T:.2f}_mu_{beta_mu1:.2f}_{beta_mu2:.2f}.xyz"
            with gzip.open(xyzname + '.gz', 'at') as f:
                num_atoms = replica.number1 + replica.number2
                cell = f"{replica.box_length} 0.0 0.0 0.0 {replica.box_length} 0.0 0.0 0.0 {replica.box_length}"
                f.write(f"{num_atoms}\n")
                f.write(f"Step {step} Lattice=\"{cell}\" Properties=species:S:1:pos:R:3\n")
                for pos in replica.positions_1:
                    f.write(f"{replica.type1} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n") 
                for pos in replica.positions_2:
                    f.write(f"{replica.type2} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n") 
            
def main(config, input_folder):

    if len(config['particle_types']) == 1:
        run_parallel_single_type(config, input_folder)
    elif len(config['particle_types']) == 2:
        run_parallel_two_type(config, input_folder)
    else:
        print("Only single and two type simulations are supported.")
        exit()
    
    
    
