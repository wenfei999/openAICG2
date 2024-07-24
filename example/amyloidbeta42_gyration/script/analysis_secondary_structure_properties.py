import numpy as np
import matplotlib.pyplot as plt

sec_struc_key = {'C':0,'H':1,'E':2}
sec_struc = []
with open('../output/monomer_010.ss','r') as read_f:
    for line in read_f:
        if 'C' in line or 'E' in line or 'H' in line:
           line = line.strip('\n')
           line = line.split()
           sec_struci =[sec_struc_key[i] for i in line[1:]]
           sec_struc.append(sec_struci)

sec_struc = np.array(sec_struc)
num_snapshot,num_residue = np.shape(sec_struc)
fraction_coil = 100*np.sum(sec_struc==0,axis=0)/num_snapshot
fraction_helix = 100*np.sum(sec_struc==1,axis=0)/num_snapshot
fraction_strand = 100*np.sum(sec_struc==2,axis=0)/num_snapshot
res_idx = range(1,num_residue+1)
'''
fig,ax = plt.subplots(figsize=(4.6,4))
ax.tick_params(which='both',labelsize='large',width=2)
ax.plot(res_idx,fraction_coil,'-',lw=2,color='k',label='coil/loop',alpha=0.8)
ax.set_xlabel('Residue index', fontsize=24)
ax.set_ylabel('loop/coil(%)', fontsize=24)
x_major_locator=plt.MultipleLocator(4)
ax.xaxis.set_major_locator(x_major_locator)
ax.legend(fontsize=14)
'''

fig,ax = plt.subplots(figsize=(4.6,4))
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
#ax.spines['right'].set_linewidth(2)

tkw = dict(which='both',labelsize='large', width=2)
part_coil = ax.twinx()
part_coil.spines['right'].set_linewidth(2)
c3, = part_coil.plot(res_idx,fraction_coil,'-',lw=2,color='r',label='coil/loop',alpha=0.8)
h1, = ax.plot(res_idx,fraction_helix,'-',lw=2,color='k',label='helix',alpha=0.8)
e2, = ax.plot(res_idx,fraction_strand,'--',lw=2,color='k',label='strand',alpha=0.8)
ax.set_xlabel('Residue index', fontsize=24)
ax.set_ylabel('Percent(%)', fontsize=24)
part_coil.set_ylabel('Percent(%)',fontsize=24)
part_coil.yaxis.label.set_color(c3.get_color())

ax.tick_params(**tkw)
part_coil.tick_params(axis='y',colors=c3.get_color(),**tkw)
part_coil.spines['right'].set_edgecolor(c3.get_color())

x_major_locator=plt.MultipleLocator(4)
ax.xaxis.set_major_locator(x_major_locator)

lines = [c3,e2,h1]
ax.legend(lines,[l.get_label() for l in lines], fontsize=14,loc='best', bbox_to_anchor=(0.38, 0.18, 0.5, 0.5),frameon=False)
plt.savefig('../output/secondary_structure_feature_010.png',dpi=600,bbox_inches='tight')
'''
fig,ax = plt.subplots(figsize=(4.6,4))
ax.tick_params(which='both',labelsize='large',width=2)
ax.plot(res_idx,fraction_strand,'-',lw=2,color='k',label='strand',alpha=0.8)
ax.set_xlabel('Residue index', fontsize=24)
ax.set_ylabel('Strand(%)', fontsize=24)
x_major_locator=plt.MultipleLocator(4)
ax.xaxis.set_major_locator(x_major_locator)
ax.legend(fontsize=12)
'''
plt.show()
