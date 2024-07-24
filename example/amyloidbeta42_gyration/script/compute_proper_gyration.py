import mdtraj as md
import numpy as np
import json
import matplotlib.pyplot as plt
import os 

_A_to_nm = 0.1 
config_para = json.load(open('../input/simulationparams.json','r'))
num_traj = config_para['num_traj']
Rg = np.zeros(num_traj,dtype=float)
rg_all = []
for i in range(num_traj):
    path_traj = '../output/monomer_%03d.dcd'%i
    if not os.path.exists(path_traj):
        continue
    pdb = md.load('../output/monomer_%03d.pdb'%i)
    top = pdb.topology
    ca_aotm_indices = [i.index for i in list(top.atoms) if i.name =='CA']
    traj = md.load(path_traj, atom_indices=ca_aotm_indices,top='../output/monomer_%03d.pdb'%i)
    rg = md.compute_rg(traj,masses=None)
    Rg[i] = np.mean(rg)/_A_to_nm
    rg_all.extend(rg)
RSE  = np.std(Rg)/(_A_to_nm*np.sqrt(len(Rg)))
print('Rg:',Rg,'Rse:',RSE)
np.savetxt('../output/Rg.txt',Rg,fmt='%f')
#rg_all = np.array(rg_all)
#rg_proper = np.load('../output/monomer_%.1f.npy'%lambdascale)
hist,bin_edges = np.histogram(rg_all,bins=11,density=True)
x = (bin_edges[1:]+bin_edges[:-1])/2#(2*_A_to_nm)
probability = hist
fig2,ax2 = plt.subplots(figsize=(4.6,4))
ax2.tick_params(which='both',labelsize='large',width=2)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.spines['top'].set_linewidth(2)
ax2.spines['right'].set_linewidth(2)
ax2.plot(x*10,probability,color='k',lw=2.0)
ax2.set_xlabel(r'$R_{g}$ ($\AA$)',fontsize=24)
ax2.set_ylabel(r'P($R_{g}$)' ,fontsize=2)
plt.savefig('../output/proper_rg_distribution.png',dpi=300,bbox_inches='tight')
plt.show()
