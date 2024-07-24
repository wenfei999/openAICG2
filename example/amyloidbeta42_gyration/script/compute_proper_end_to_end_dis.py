import mdtraj as md
import numpy as np
import json
import matplotlib.pyplot as plt

_A_to_nm = 0.1 
lambdascale = 1.018 
traj = md.load('../output/monomer_%03d.dcd'%11,top='../output/monomer_%03d.pdb'%11)
Ree = md.compute_distances(traj,atom_pairs=[[0,76]])
mRee = np.mean(Ree)
seRee  = np.std(Ree)/(np.sqrt(len(Ree)))
lambdakh = lambdascale * 0.228
print('lambdakh:',lambdakh,'Ree:',mRee,'Reese:',seRee)

#rg_proper = np.load('../output/monomer_%.1f.npy'%lambdascale)
hist,bin_edges = np.histogram(Ree,bins=14,density=True)
x = (bin_edges[1:]+bin_edges[:-1])/2
probability = hist

fig2,ax2 = plt.subplots(figsize=(4.6,4))
ax2.tick_params(which='both',labelsize='large',width=2)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.spines['top'].set_linewidth(2)
ax2.spines['right'].set_linewidth(2)
ax2.plot(x,probability,color='k',lw=2.0)
ax2.set_xlabel(r'$R_{ee}$ ($\AA$)',fontsize=24)
ax2.set_ylabel('Probability' ,fontsize=24)
plt.savefig('../output/proper_Ree_distribution_%03d.png'%11,dpi=300,bbox_inches='tight')
plt.show()
