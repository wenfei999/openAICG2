import numpy as np
import mdtraj as md
import sys
from scipy.spatial.distance import pdist

def Compute_Hydrodynamic_Radius(distance,Ntot):
    return 1/(np.mean(np.sum(1/distance,axis=1))/Ntot**2)

traj = md.load('../output/monomer_%03d.dcd'%10,top='../output/monomer_%03d.pdb'%10)

top = traj.topology
atoms = list(top.atoms)
atom_pairs = [[ai.index,aj.index] for i,ai in enumerate(atoms[:-1]) for aj in atoms[i+1:]]
distance = md.compute_distances(traj,periodic=False,atom_pairs=atom_pairs)
inve_rij = np.mean(1/distance,axis=1)
rh = 1/inve_rij
Rh = np.mean(rh)
md_rg = md.compute_rg(traj,masses=None)
rh2 = np.sum(np.mean(1/distance,axis=0))/len(atom_pairs)
print('Rh:',Rh,1/rh2,'Rg',np.mean(md_rg),'Rh/Rg:',Rh/np.mean(md_rg))
