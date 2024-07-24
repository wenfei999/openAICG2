import mdtraj as md
import numpy as np

# Load the trajectory
trajectory = md.load('../output/monomer_%03d.dcd'%10,top='../output/monomer_%03d.pdb'%10)

# Define the atom indices for which you want to calculate the hydrodynamic radius
# Example: use all atoms
atom_indices = trajectory.topology.select('all')
n_atoms = len(atom_indices)

# Compute pairwise distances between selected atoms for each frame
pairs = np.array([(i, j) for i in atom_indices for j in atom_indices if i < j])
pairwise_distances = md.compute_distances(trajectory, pairs)

# Compute the hydrodynamic radius R_hyd
inverse_distances = 1.0 / pairwise_distances  # Compute the inverse of distances
average_inverse_distances = np.mean(inverse_distances, axis=0)  # Ensemble average over all frames
R_hyd_inverse = np.sum(average_inverse_distances) / (n_atoms * (n_atoms - 1) / 2)
R_hyd = 1.0 / R_hyd_inverse

print(f'Estimated Hydrodynamic Radius: {R_hyd:.2f} nm')
