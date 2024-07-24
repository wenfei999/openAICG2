import numpy as np 
import matplotlib.pyplot as plt
import mdtraj as md
import sys
def Ree(r,C,g,alpha,delta):
    return C*r**g*np.exp(-alpha*r**delta)

dr = 0.1
r = np.arange(0,3,dr)

traj = md.load('../output/monomer.dcd',top='../output/monomer.pdb')
ree = md.compute_distances(traj,atom_pairs=[[0,76]])
stdree  = np.std(ree)/(np.sqrt(len(ree)))
ree2 = np.mean(ree**2)
ree = ree/ree2**0.5
#rg_proper = np.load('../output/monomer_%.1f.npy'%lambdascale)
hist,bin_edges = np.histogram(ree,bins=r,density=True)
x = (bin_edges[1:]+bin_edges[:-1])/2
Probability_md = x**2*hist*np.diff(bin_edges)

delta_gc = 1/(1-0.5)
delta_saw = 1/(1-0.588)

C_gc = (3/(2*np.pi))**(3/2)
alpha_gc = 1.5

C_saw = 0.278
alpha_saw = 1.206

g_gc = 0
g_saw = 0.28


px_gc = Ree(r,C_gc,g_gc,alpha_gc,delta_gc)
px_saw = Ree(r,C_saw,g_saw,alpha_saw,delta_saw)
fig,ax = plt.subplots(figsize=(4.6,4))
Probability_gc = 4*np.pi*r**2*dr*px_gc
Probability_saw = 4*np.pi*r**2*dr*px_saw
print(np.sum(Probability_gc/2))
ax.plot(r,Probability_gc,lw=2,color='b',alpha=0.4,label='Gaussion')
ax.plot(r,Probability_saw,lw=2,color='g',alpha=0.4,label='SAW')
ax.plot(x,Probability_md,lw=2,color='r',alpha=0.4,label=r'A$\beta42$')
ax.set_xlabel(r'$R_{ee}$',fontsize=20)
ax.set_ylabel(r'$P(R_{ee})4 \pi R^2dR$',fontsize=20)
plt.legend(fontsize=14)
plt.savefig('../output/')
plt.show()
