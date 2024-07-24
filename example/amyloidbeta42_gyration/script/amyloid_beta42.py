import openmm as mm
from openmm import app
from openmm import unit
import sys
import numpy as np
import pandas as pd
import mdtraj as md
import json
import os 

sys.path.append('../../../')
from openaicg2.forcefield.aicgmodel import AICG2Model
from openaicg2 import utils 

######################################################################################
#                              Initialioze parameters                                #
######################################################################################
       
# load pdb and psf file to get topology and bonds
simu_params_path = str(sys.argv[1])
configure_params = json.load(open(simu_params_path,'r'))
T = configure_params['Temperature'] 
friction = configure_params['friction']
timestep = configure_params['timestep']
tot_simu_steps = configure_params['total_mdsteps']
report_period = configure_params['report_period']
output_file = configure_params['output_file_name']
platform_type = configure_params['platform_type']
initial_pdb_path = configure_params['initial_pdb']
monomer_psf_path = configure_params['monomer_psf']
box = configure_params['box_vector']
native_info_path = configure_params['native_information']
k_go_con_scale = configure_params['k_go_scale']
lambdakh_scale = configure_params['lambdakh_scale']
cutoff_kh = configure_params['cutoff_kh']

pdb = md.load(initial_pdb_path) 
psf = md.load_psf(monomer_psf_path)
top = pdb.topology.to_openmm()
top._bonds = []
bonds = psf._bonds
# Refine the bond in topology
redefine_top = utils.RedefineTopology()
redefine_top.redefine_bond(top,bonds)

# load parameter in native information file
ParserNinfo=utils.ParserNinfo()
ParserNinfo.get_ninfo(native_info_path)
######################################################################################
#                              Create model and input parameter                      #
######################################################################################
# create model
model = AICG2Model()
model.create_system(top,use_pbc=True,
                    box_a=box['x'], box_b=box['y'], box_c=box['z'],
                    nonbondedMethod=app.CutoffPeriodic,
                    remove_cmmotion=False)
# input native information
model.append_ff_params(ParserNinfo)
######################################################################################
#                              Add force to model                                    #
######################################################################################
model.add_all_default_ener_function(oriented_Hbond=False,cutoffkh=cutoff_kh,kh_epsilon_scale=lambdakh_scale,temperature=T)

######################################################################################
#                              Create a simulation                                   #
######################################################################################
integrator = mm.LangevinMiddleIntegrator(T*unit.kelvin,friction/unit.picosecond,timestep*unit.femtosecond)
#integrator = mm.VerletIntegrator(timestep*unit.femtosecond)
init_coord = pdb.xyz[0,:,:] * unit.nanometer
model.set_simulation(integrator, platform_name=platform_type,properties={'Precision': 'single'},init_coord=init_coord)
model.move_COM_to_box_center(use_pbc=False)
model.simulation.context.setVelocitiesToTemperature(T*unit.kelvin)
model.simulation.minimizeEnergy()
model.add_reporters(tot_simu_steps, report_period, 
                    output_traj_name='../output/%s'%(output_file),report_traj_format='dcd'
                    ,report_traj=True,report_state_log=True)
print('Running simulation!!!') 
try:
    model.simulation.step(tot_simu_steps)
except:
    print('simulation error')
    model.save_system('../output/system_%s.xml'%(output_file))
    model.save_state('../output/state_%s.xml'%(output_file))

