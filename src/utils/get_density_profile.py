import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gzip

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
        
        timestep_positions = defaultdict(list)
        timestep_lattice_vectors = None
        
        for line in lines:
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
                    x, y, z = map(float, parts[1:4])  # assuming x, y, z are columns 2, 3, 4
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

def calculate_density_profile(positions, lattice_vectors, bins=100):
    """
    Function to calculate the spatial density profile along the x direction for a single timestep.
    
    Parameters:
    - positions (np.ndarray): Array of particle positions (3D).
    - lattice_vectors (np.ndarray): Array of lattice vectors (3x3 matrix).
    - bins (int): Number of bins for the histogram along the x direction.
    
    Returns:
    - bin_centers (np.ndarray): Centers of the bins along the x direction.
    - density_profile (np.ndarray): Density profile along the x direction, normalized by volume.
    """
    x_positions = positions[:, 0]  # extract x coordinates
    
    # Calculate density profile
    counts, bin_edges = np.histogram(x_positions, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume = calculate_volume(lattice_vectors)/bins
    
    # Normalize density profile by volume
    density_profile = counts / volume
    
    return bin_centers, density_profile

def average_density_profiles(positions_list, lattice_vectors_list, bins=100):
    """
    Function to compute the average density profile over multiple timesteps for each species.
    
    Parameters:
    - positions_list (list of dict): List of dictionaries of particle positions for each timestep, keyed by species.
    - lattice_vectors_list (list of np.ndarray): List of arrays of lattice vectors for each timestep.
    - bins (int): Number of bins for the histogram along the x direction.
    
    Returns:
    - bin_centers (np.ndarray): Centers of the bins along the x direction.
    - average_density_profiles (dict): Average density profiles for each species, keyed by species.
    """
    species_density_profiles = defaultdict(list)
    
    for timestep_positions, lattice_vectors in zip(positions_list, lattice_vectors_list):
        for species, positions in timestep_positions.items():
            positions_array = np.array(positions)
            bin_centers, density_profile = calculate_density_profile(positions_array, lattice_vectors, bins=bins)
            species_density_profiles[species].append(density_profile)
    
    # Compute average density profile for each species
    average_density_profiles = {species: np.mean(profiles, axis=0) for species, profiles in species_density_profiles.items()}
    
    return bin_centers, average_density_profiles

def plot_density_profiles(bin_centers, average_density_profiles):
    """
    Function to plot the density profiles for each species.
    
    Parameters:
    - bin_centers (np.ndarray): Centers of the bins along the x direction.
    - average_density_profiles (dict): Average density profiles for each species, keyed by species.
    """
    plt.figure(figsize=(10, 6))
    params = {
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    }
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update(params)

    plt.tick_params(direction="in", which="minor", length=3)
    plt.tick_params(direction="in", which="major", length=5, labelsize=14)
    plt.grid(which="major", ls="dashed", dashes=(1, 3), lw=1, zorder=0)

    for species, density_profile in average_density_profiles.items():
        plt.plot(bin_centers, density_profile, label=f'Species {species}', lw=2.5)
    
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho(x)$')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == '__main__':
    file_path = 'output.xyz.gz'  # replace with your extended XYZ file path
    positions_list, lattice_vectors_list = read_extended_xyz(file_path)
    bin_centers, average_density_profiles = average_density_profiles(positions_list, lattice_vectors_list, bins=800)
    plot_density_profiles(bin_centers, average_density_profiles)

    # Output the density profiles to density.out
    with open('density.out', 'w') as f:
        for species, density_profile in average_density_profiles.items():
            for center, density in zip(bin_centers, density_profile):
                f.write(f"{species} {center} {density}\n")
