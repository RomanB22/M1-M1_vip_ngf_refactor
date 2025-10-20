"""
init.py

Starting script to run NetPyNE-based M1 model.

Usage:
    python init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py

Contributors: salvadordura@gmail.com
"""

#import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers

#from neuron import h,gui
from netpyne import sim
#from cfg_cell import cfg
#from netParams_cell import netParams

cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg_cell.py', netParamsDefault='netParams_cell.py')
sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node


#sim.allSimData['EPSPratio'] = sim.analysis.plotEPSPAmp(include= [0,1], trace='V_soma', start=300, interval=1000.0/cfg.groupRate, number=3, amp= 'ratio', saveFig=True, showFig=False)[0].tolist()
#sim.allSimData['EPSPamp'] = sim.analysis.plotEPSPAmp(include= [0], trace='V_soma', start=300, interval=50, number=3, amp= 'relative', polarity='exc', saveFig=False, showFig=False)[0].tolist()
#sim.allSimData['EPSPpeak'] = sim.analysis.plotEPSPAmp(include= [0,1], trace='V_soma', start=300, interval=1000.0/cfg.groupRate, number=1, amp= 'absolute', saveFig=True, showFig=False)[0].tolist()
# sim.allSimData['EPSPpeak_ZD'] = sim.analysis.plotEPSPAmp(include= [1], trace='V_soma', start=300, interval=1000.0/cfg.groupRate, number=1, amp= 'absolute', saveFig=True, showFig=False)[0].tolist()

# print  sim.allSimData['EPSPamp']

# RMP_ih = sim.allSimData['V_soma']['cell_2'][4990]
# RMP_zd = sim.allSimData['V_soma']['cell_3'][4990]
# sub_epsp = sim.allSimData['V_soma']['cell_4'][10000] - RMP_ih
# sub_sagpeak = max(sim.allSimData['V_soma']['cell_4'][5000:6000]) - RMP_ih
# sub_sag = (sub_sagpeak-sub_epsp)/sub_epsp *100

# print '\n--------------------------------------------'
# print 'EPSP peak (zd - ih): %.2f - %.2f = %.2f'%(sim.allSimData['EPSPpeak_ZD'][0][0], sim.allSimData['EPSPpeak'][0][0], sim.allSimData['EPSPpeak_ZD'][0][0] - sim.allSimData['EPSPpeak'][0][0])

# print 'fI 0.3-0.4-0.6nA : %.2f - %.2f - %.2f'%(sim.allSimData['popRates']['PT5B_fI30']*1.5, sim.allSimData['popRates']['PT5B_fI40']*1.5, sim.allSimData['popRates']['PT5B_fI60']*1.5)
# print 'stim ih-zd   : %.2f - %.2f (%.2f)'%(sim.allSimData['popRates']['PT5B']*1.5, sim.allSimData['popRates']['PT5B_ZD']*1.5, (sim.allSimData['popRates']['PT5B_ZD']*1.5-sim.allSimData['popRates']['PT5B']*1.5))
# print 'RMP ih-zd    : %.2f - %.2f'%(RMP_ih, RMP_zd)
# print 'Peak ih-zd 	: %.2f - %.2f'%(max(sim.allSimData['V_soma']['cell_2']), max(sim.allSimData['V_soma']['cell_3']))
# print 'Sub 0.1nA  	: %.2f - %.2f%% sag'%(sub_epsp, sub_sag) 
# print '--------------------------------------------\n'

sim.analysis.plotData()         			# plot spike raster
sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)


