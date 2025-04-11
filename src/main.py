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

import read_input
import external_potentials 
import argparse
#from mpi4py import MPI  # Uncomment if using MPI for replica exchange

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run GCMC simulations for short-ranged potentials."
    )
        
    parser.add_argument("-in", "--input_folder",    
        required=False, type=str, default=".",
        help="the relative path to folder containing YAML input.",
    )
        
    args = parser.parse_args()
        
    input_folder = args.input_folder
    config = read_input.load_config(input_folder + '/' + 'input.yaml')    
    ext_potentials = external_potentials.initialize_external_potentials(config)
    replica_exchange = config.get('replica_exchange', False)

    print_energy = config.get('print_energy', True)
    
    if replica_exchange == False:
        
        import potentials
        import gcmc_ff
        pair_potentials = potentials.initialize_potentials(config)
            
        if len(config['particle_types']) == 1:
            simulation = gcmc_ff.GCMC_FF_SingleType_Simulation(config, pair_potentials, ext_potentials, input_folder)
            if print_energy:
                simulation.run_simulation()
            else:
                simulation.run_simulation_no_energy()
        elif len(config['particle_types']) == 2:
            simulation = gcmc_ff.GCMC_FF_TwoType_Simulation(config, pair_potentials, ext_potentials, input_folder)
            if print_energy:
                simulation.run_simulation()
            else:
                simulation.run_simulation_no_energy()
        else:
            simulation = gcmc_ff.GCMC_FF_MultiType_Simulation(config, pair_potentials, ext_potentials, input_folder)
            simulation.run_simulation()
                
    else:
        import gcmc_re
        gcmc_re.main(config, input_folder)