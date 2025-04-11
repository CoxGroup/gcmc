import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

plt.figure(figsize=(5, 4))

params = {
    "axes.labelsize": 16,
    "axes.titlesize": 15,
}

plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams.update(params)

plt.rcParams['svg.fonttype'] = 'none'

plt.tick_params(direction="in", which="minor", length=3)
plt.tick_params(direction="in", which="major", length=5, labelsize=14)
plt.grid(which="major", ls="dashed", dashes=(1, 3), lw=1, zorder=0)


# Add the parent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the potentials module
from potentials import initialize_potentials

# Load the configuration file
with open('input.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize the pair potentials
pair_potentials = initialize_potentials(config)

# Define the distances to evaluate the potentials
r_distances = np.linspace(0.01, config['global_rc'], 1000)  # Avoid zero to prevent division by zero
beta = 1/ (config['kB'] * config['T'])
# Plot the pair potentials for all pairs
for pair, potential in pair_potentials.items():
    potential_values = potential.calculate(r_distances)
    plt.plot(r_distances, beta*potential_values, label=f'{pair}', lw=2.5)

plt.xlabel(r'$r$')
plt.ylabel(r'$\beta u(r)$')
plt.ylim(-5,20)
plt.xlim(0,config['global_rc']+1)
plt.legend()
plt.show()
plt.tight_layout()

