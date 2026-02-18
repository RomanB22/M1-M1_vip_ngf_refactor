from netpyne.batchtools.search import search
import numpy as np
import json
from cfg import cfg

with open(cfg.cellParamsJson, 'r') as fptr:
    sections = list(json.load(fptr)['secs'].keys())

sections = [sec for sec in sections if sec not in ('axon_0', 'axon_1')]

weights = list(np.arange(0.01, 0.2, 0.01)/100.0)
# Create parameter grid for search
params = {
    'sec'   : sections,
    'weight': weights,
}

# use batch_sge_config if running on a
sge_config = {
    'command': 'python test.py'}

result_grid = search(job_type = 'sh',
       comm_type       = "socket",
       params          = params,
       run_config      = sge_config,
       label           = "grid_search",
       output_path     = "./grid_batch",
       checkpoint_path = "./ray",
       num_samples     = 1,
       metric          = 'epsp',
       mode            = 'min',
       algorithm       = "variant_generator",
       max_concurrent  = 9)
