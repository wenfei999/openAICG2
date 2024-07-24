import json
import os
import numpy as np

config_para = json.load(open('../input/simulationparams.json','r'))
queue = config_para['queue']
num_compute_card = config_para['num_compute_card']
num_cpu_to_gpu = config_para['num_cpu_to_gpu']
box_size = int(config_para['box_vector']['x']/10)
num_traj = config_para['num_traj']
start = 0
for i in range(start,num_traj):
    config_para['output_file_name'] = "monomer_%03d"%(i)
    with open('../output/%s.json'%(config_para['output_file_name']),'w') as simu_para:
         json.dump(config_para,simu_para,indent="")
    with open('sub','w') as sub:
         sub.write('#BSUB -q %s\n'%queue)
         sub.write('#BSUB -gpu "num=%d"\n'%num_compute_card) 
         sub.write('#BSUB -n %d\n'%num_cpu_to_gpu)
         sub.write('#BSUB -J g%03d\n'%(i))
         sub.write('#BSUB -o ../output/out_%03d\n'%(i))
         sub.write('#BSUB -e ../output/err_%03d\n'%(i))
         sub.write('python amyloid_beta42.py ../output/%s.json\n'%(config_para['output_file_name']))
    #os.system('bsub < sub')
    os.system('python amyloid_beta42.py ../output/%s.json\n'%(config_para['output_file_name']))
