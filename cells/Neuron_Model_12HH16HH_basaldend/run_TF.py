from NeuronModelClass import NeuronModel
from NrnHelper import *
import NrnHelper as nh
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np
from Na12HH_dendModel_TF import *
import Na12HH_dendModel_TF as tf
import os
# import efel_feature_extractor as ef
from currentscape.currentscape import plot_currentscape
import logging
import pandas as pd

sim_config_soma = {
                'section' : 'soma',
                'segment' : 0.5,
                'section_num': 0,
                'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'],
                'current_names' : ['Ih','SKv3_1','Na16 WT','Na16 WT','Na12','Na12 MUT','pas'],
                'ionic_concentrations' :["cai", "ki", "nai"]
                }




root_path_out = './Plots/' ##path for saving your plots
if not os.path.exists(root_path_out): ##make directory if it doens't exist
        os.makedirs(root_path_out)


config_dict3={"sim_config_soma": sim_config_soma}

for config_name, config in config_dict3.items():
  path = f'WT_spikes_dvdt_FI_curves_ExamplePlots_021626_2'


## Run the model and obtain raw data Voltages and corresponding times. Can also get currents if desired.
# Commented params are defaults in Na12HHModel_TF.py. Can change here if you want to tune or alter things (such as reduce nav to 0 to mimic TTX).  
simwt = tf.Na12Model_TF(#ais_nav12_fac=5.76,
                        # ais_nav16_fac=1.08*0.6,
                        # nav12=1.1*1.1,
                        # nav16=1.43*1.2, 
                        # somaK=0.022, 
                        # KP=3.9375, 
                        # KT=5,
                        # ais_ca = 43*0.5,
                        # ais_Kca = 0.25,
                        # soma_na16=0.8,
                        # soma_na12=2.56,
                        # node_na = 1,
                        # dend_nav12=1,
                        na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'], ## Can change mut_name to na12_mut.txt to make het
                        na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                        plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
wt_Vm1,_,wt_t1,_ = simwt.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config)


####---------------------------------####
######### Plotting and analysis #########
####---------------------------------####

## Plot Stim and dvdt in single plot
fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
simwt.plot_stim(axs = axs[0],stim_amp = 0.5,dt=0.005,stim_dur = 500, clr='cadetblue')
plot_dvdt_from_volts(simwt.volt_soma, simwt.dt, axs[1],clr='cadetblue')
fig_volts.savefig(f'{simwt.plot_folder}/WT_validate.pdf') #Change output file path here 

## Plot the FI curve from stim raw data y-axis label hard-coded in NrnHelper.py line 62
# wt_fi=simwt.plot_fi_curve_2line(wt_data=None,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'WT_FI-stim0.5dur500')