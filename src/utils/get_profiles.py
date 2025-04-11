import numpy as np
import gzip
import sys
import yaml
#sys.path.insert(0, '/scratch/btb32/code/GCMC/src')
from external_potentials import initialize_external_potentials
import csv
from collections import defaultdict
import argparse

def read_extended_xyz(file_path):
    """
    Function to read an extended XYZ file and extract positions and lattice vectors for each timestep.
    
    Parameters:
    - file_path (str): Path to the extended XYZ file.
    
    Returns:
    - positions_list (list of dict): List of dictionaries of particle positions for each timestep, keyed by species.
    - lattice_vectors_list (list of np.ndarray): List of arrays of lattice vectors for each timestep.
    """
    positions_list = []
    lattice_vectors_list = []
    
    with gzip.open(file_path, 'rt') as f:
        lines = f.readlines()
        num_particles = int(lines[0].strip())
        
        timestep_positions = defaultdict(list)
        timestep_lattice_vectors = None
        
        for line in lines[1:]:
            line = line.strip()
            
            if line.startswith('Step'):
                if timestep_lattice_vectors is not None:
                    positions_list.append(dict(timestep_positions))
                    lattice_vectors_list.append(timestep_lattice_vectors)
                    timestep_positions = defaultdict(list)
                lattice_str = line.split("Lattice=")[1].split("Properties")[0].strip('"').strip()
                lattice_values = lattice_str.split()
                timestep_lattice_vectors = np.array([float(val.strip('"')) for val in lattice_values]).reshape(3, 3)
            else:
                parts = line.split()
                if len(parts) == 4:  # assuming XYZ format with 3D coordinates
                    species = parts[0]
                    x, y, z = map(float, parts[1:4])  
                    timestep_positions[species].append([x, y, z])
        
        # Append the last timestep
        if timestep_lattice_vectors is not None:
            positions_list.append(dict(timestep_positions))
            lattice_vectors_list.append(timestep_lattice_vectors)
    
    return positions_list, lattice_vectors_list

def calculate_volume(lattice_vectors):
    """
    Function to calculate the volume of the simulation box from lattice vectors.
    
    Parameters:
    - lattice_vectors (np.ndarray): Array of lattice vectors (3x3 matrix).
    
    Returns:
    - volume (float): Volume of the simulation box.
    """
    volume = np.abs(np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])))
    return volume

def calculate_density_profile(positions, lattice_vectors, bin_edges):
    """
    Function to calculate the spatial density profile along the x direction for a single timestep.
    
    Parameters:
    - positions (np.ndarray): Array of particle positions (3D).
    - lattice_vectors (np.ndarray): Array of lattice vectors (3x3 matrix).
    - bin_edges (np.ndarray): Array of bin edges along the x direction.
    
    Returns:
    - bin_centers (np.ndarray): Centers of the bins along the x direction.
    - density_profile (np.ndarray): Density profile along the x direction, normalized by volume.
    """
    x_positions = positions[:, 0]  # extract x coordinates
    
    # Calculate density profile
    counts, bin_edges = np.histogram(x_positions, bins=bin_edges, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume = calculate_volume(lattice_vectors) / (len(bin_edges) - 1)
    
    # Normalize density profile by volume
    density_profile = counts / volume
    
    return bin_centers, density_profile

def average_density_profiles(positions_list, lattice_vectors_list, bin_edges):
    """
    Function to compute the average density profile over multiple timesteps for each species.
    
    Parameters:
    - positions_list (list of np.ndarray): List of arrays of particle positions for each timestep.
    - lattice_vectors_list (list of np.ndarray): List of arrays of lattice vectors for each timestep.
    - bin_edges (np.ndarray): Array of bin edges along the x direction.
    
    Returns:
    - bin_centers (np.ndarray): Centers of the bins along the x direction.
    - average_density_profile (np.ndarray): Average density profile along the x direction, normalized by volume.
    """
    
    species_density_profiles = defaultdict(list)
    
    for positions, lattice_vectors in zip(positions_list, lattice_vectors_list):
        for species, positions_species in positions.items():
            positions_array = np.array(positions_species)
            bin_centers, density_profile = calculate_density_profile(positions_array, lattice_vectors, bin_edges)
            species_density_profiles[species].append(density_profile)
    
    # Compute average density profile for each species
    average_density_profiles = {species: np.mean(profiles, axis=0) for species, profiles in species_density_profiles.items()}
   
    return bin_centers, average_density_profiles

def initialize_potentials(config, particle_type):
    """
    Function to initialize the external potentials based on the configuration.
    
    Parameters:
    - config (dict): Configuration dictionary.
    
    Returns:
    - beta (float): Inverse temperature (1/kB*T).
    - Vext (function): External potential function for the given particle type.
    - mu (float): Chemical potential for the given particle type.
    """
    external_potentials = initialize_external_potentials(config)
    beta = 1 / (config['kB'] * config['T'])
    
    Vext = external_potentials[particle_type]
    mu = config['particle_types'][particle_type]['mu'] * config['kB'] * config['T']
    
    return beta, Vext, mu

def filter_results(bin_centers, average_density_profile, muloc_profile):
    """
    Function to filter the density and mu_loc profiles.
    
    Parameters:
    - bin_centers (np.ndarray): Array of bin centers.
    - average_density_profile (np.ndarray): Array of average density profile values.
    - muloc_profile (np.ndarray): Array of mu_loc profile values.
    
    Returns:
    - filtered_bin_centers (np.ndarray): Filtered bin centers.
    - filtered_density_profile (np.ndarray): Filtered average density profile.
    - filtered_muloc_profile (np.ndarray): Filtered mu_loc profile.
    """
    valid_indices = (average_density_profile > 0) & (muloc_profile < 1e10)
    filtered_bin_centers = bin_centers[valid_indices]
    filtered_density_profile = average_density_profile[valid_indices]
    filtered_muloc_profile = muloc_profile[valid_indices]
    return filtered_bin_centers, filtered_density_profile, filtered_muloc_profile

def write_profile(filename, centers, densities, mulocs, c1s):
    """
    Function to write the density profile to a file.
    
    Parameters:
    - filename (str): Output file name.
    - centers (np.ndarray): Bin centers.
    - densities (np.ndarray): Density values.
    - mulocs (np.ndarray): MuLoc values.
    - c1s (np.ndarray): C1 values.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(["xbins", "rho", "muloc", "c1"])  # Include headers for your columns
        for center, density, muloc, c1 in zip(centers, densities, mulocs, c1s):
            writer.writerow([f"{center:.4f}", f"{density:.20f}", f"{muloc:.16f}", f"{c1:.16f}"])


if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description="Run GCMC simulations for short-ranged potentials."
    )
    
    parser.add_argument("-in", "--input_folder",
        required=False, type=str, default=".",
        help="the relative path to folder containing YAML input.",
    )

    args = parser.parse_args()
    
    input_folder = args.input_folder
    
    
    config_file = input_folder + '/' + 'input.yaml'
    output_xyz_file = input_folder + '/' + 'output.xyz.gz'
    
    # Load the configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        
    # Read the XYZ file
    positions_list, lattice_vectors_list = read_extended_xyz(output_xyz_file) 
    
    # Define bin edges
    bin_edges = x_positions = np.linspace(0, config['box_length'], 2001) 
    

    # Compute the average density profile
    bin_centers, average_density_profiles = average_density_profiles(positions_list, lattice_vectors_list, bin_edges)
    

    for species, density_profile in average_density_profiles.items():
        # Initialize the external potentials and chemical potential
        beta, Vext, mu = initialize_potentials(config, species)
        
        # Compute Vext values and mu_loc profile
        x_positions = np.array([bin_centers])
        Vext_values = Vext(x_positions) 
        muloc_profile = Vext_values*beta - mu*beta

        # Get density
        average_density_profile = average_density_profiles[species]
        
        # Compute C1 profile
        valid_indices = (average_density_profile > 0) & (muloc_profile < 1e10)
        c1_profile = np.full(len(bin_centers), np.nan)
        c1_profile[valid_indices] = np.log(average_density_profile[valid_indices]) + muloc_profile[valid_indices]
    
        muloc_profile[~valid_indices] = np.nan
        average_density_profile[~valid_indices] = 0.0
    
        # Write the results to a file
        name_file = input_folder + '/' + f"{species}_profiles.out"
        write_profile(name_file, bin_centers, average_density_profile, -muloc_profile, c1_profile)
    


