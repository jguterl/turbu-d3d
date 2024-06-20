#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:26:52 2022

@author: guterlj
"""

from UEDGEToolBox.Launcher import UBoxLauncher
from UEDGEToolBox.Launcher import *
from UEDGEToolBox.ParallelLauncher import slurm_support 
import subprocess

def setup_slurm_array_scripts(slurm_options,directory, casename, Nsim, n_jobs):
    slurm_runners = [] 
    slurmscript_directory = os.path.join(directory,'sbatch_array_scripts')
    try:
        os.mkdir(slurmscript_directory)
    except:
        pass
    
    n_files = int(np.ceil(Nsim/n_jobs))
    k = 1
    for i in range(n_files):
        script_name = '{}.sbatch'.format(casename + '_' + str(i))
        slurm_options["J"] = casename + '_' + str(i)
        slurm_options["o"] = os.path.join(slurmscript_directory,"{}.o".format(slurm_options["J"]))
        slurm_options["e"] = os.path.join(slurmscript_directory,"{}.e".format(slurm_options["J"]))  
        slurm_options["array"] = str(0) + '-' + str(n_jobs-1)
        logpath= os.path.join(slurmscript_directory,"{}.log".format(slurm_options["J"]))
        j = int(np.floor(i*n_jobs/450))
        if i==0:
            command = 'sbatch ' + directory + '/sbatch_scripts/' + casename +'_$SLURM_ARRAY_TASK_ID.sbatch'
        else:
            command = 'sbatch ' + directory + '/sbatch_scripts/' + casename +'_' + str(i) +'$SLURM_ARRAY_TASK_ID.sbatch'
        slurm_runner = slurm_support.SlurmSbatch(command+' >> {}'.format(logpath), **slurm_options, script_dir = slurmscript_directory, script_name = script_name, pyslurm=True)
        slurm_runners.append(slurm_runner)
        slurm_runner.write_job_file()
        
        
        print ('Submitting with process sbatch file: {}'.format(script_name))
        res = subprocess.check_output(['sbatch ' + slurmscript_directory + '/' + script_name], shell=True)
        
        if j==k:
            k = k+1
            while os.path.isfile(directory+'/sbatch_scripts/'+casename + '_' + str(i) + str(int(n_jobs-1)) +'.log')==False:
                []
            # print(directory+'/sbatch_scripts/'+casename + '_' + str(j) + str(int(n_jobs-1)) +'.log')
            # break
def create_log_file(params, project, casename, inputfile):
    data2 = {}
    #project = "slab_1D_scan_test_array"
    data2['inputfile'] = '/fusion/projects/boundary/peretm/slab/input_1D_fixed_fraction.py'
    #casename="runtest_array_sim"
    data2['directory'] = os.path.join('/fusion/projects/boundary/peretm/simulations/', project)

    command = 'python ' + data2['inputfile'] 
    
    n_val = 1
    data2["params"] = {}
    for i in range(len(list(params.keys()))):
        n_val = n_val * len(params[list(params.keys())[i]])
     
        data2['params'][list(params.keys())[i]] = params[list(params.keys())[i]]
        
    data2['nsim'] = n_val
   

    # data2['params']['ncore'] = params["ncore"]
    # data2['params']['pcore'] = params["pcore"]
    # data2['params']['zdiv'] = params["zdiv"]

    data2['project'] = project
    data2['sims'] = {}
    #np.linspace(0,n_val**3-1,n_val**3).astype(int).tolist()
    param_array = np.zeros((n_val,len(list(params.keys()))))
    l = 0

    for i in range(len(params["ncore"])):
        for j in range(len(params["pcore"])):
            for k in range(len(params["zdiv"])):
                param_array[l,0] = params["ncore"][i]
                param_array[l,1] = params["pcore"][j]
                param_array[l,2] = params["zdiv"][k]
                data2['sims'][l] = {}
                data2['sims'][l]['command'] = command +' --ncore=' +str(params["ncore"][i]) +' --pcore=' + str(params["pcore"][j]) + ' --zdiv=' + str(params['zdiv'][k]) +' --casename=' +casename + '_' + str(l) + ' --project=' + project
                data2['sims'][l]['casename'] = casename + '_' + str(l)
                data2['sims'][l]['inputfile'] = data2['inputfile']
                data2['sims'][l]['logpath'] = data2['directory'] + '/sbatch_scripts/' + casename + '_'+ str(l) + '.log'
                data2['sims'][l]['params'] = {}
                data2['sims'][l]['params']['ncore'] = params["ncore"][i]
                data2['sims'][l]['params']['pcore'] = params["pcore"][j]
                data2['sims'][l]['params']['zdiv'] = params["zdiv"][k]
                data2['sims'][l]['project'] = project
                
                l=l+1
                
    data2['sims_param_array'] = param_array

    np.save(data2['directory']+ '/log.npy', data2, allow_pickle=True)    
    # for (i,sim) in self.sim_setup['sims'].items():
    #     slurmscript_directory = os.path.join(self.sim_setup['directory'],'sbatch_array_scripts')
    #     try: 
    #         os.mkdir(slurmscript_directory)
    #     except:
    #         pass
    #     script_name = '{}.sbatch'.format(sim['casename'])
    #     slurm_options["J"] = sim['casename']
    #     slurm_options["o"] = os.path.join(slurmscript_directory,"{}.o".format(sim["casename"]))
    #     slurm_options["e"] = os.path.join(slurmscript_directory,"{}.e".format(sim["casename"]))
    #     logpath= os.path.join(slurmscript_directory,"{}.log".format(sim["casename"]))
    #     sim['logpath'] = logpath
    #     slurm_runner = slurm_support.SlurmSbatch(sim['command']+' >> {}'.format(logpath), **slurm_options, script_dir = self.slurmscript_directory, script_name = script_name, pyslurm=True)
    #     slurm_runners.append(slurm_runner)
    #     slurm_runner.write_job_file()
        
        
        
project = "slab_1D_scan_fixed_resolution"
UBox.CreateProject(project,force=True, overwrite=True)
inputfile = '/fusion/projects/boundary/peretm/slab/input_1D_fixed_resolution.py'
params = {}

n_val = 30
n_jobs = 10
params["ncore"] = np.logspace(18,21,n_val)
params["pcore"] = np.logspace(6,8,n_val)/2
params["zdiv"] = np.linspace(5,100,n_val) + 2.61


create_log_file(params, project, "runtest_array_fixed_resolution", inputfile)
#params["fc"] = 0.001
# params["fc"] = [0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15, 0.175 ,0.2]

#params["ncore"] = np.logspace(18,21,1)
#params["pcore"] = np.logspace(6,8,1)
#params["fc"] = [0.01]

UBox.setup_array_runs(params,inputfile,casename="runtest_array_fixed_resolution")

slurm_options = {}
slurm_options['p'] = 'short'
#slurm_options['qos'] = 'debug'
#slurm_options['account'] = 'm3938'
#slurm_options['constraint'] = 'haswell'
slurm_options['J'] = ''
slurm_options['t'] = '00-00:15:00'
#slurm_options['t'] = '00-00:10:00'
slurm_options['o'] = '%j.o'
slurm_options['e'] = '%j.e'
slurm_options['ntasks-per-node'] = 1
slurm_options['N'] = 1

UBox.setup_slurm_scripts(slurm_options)

slurm_array_options = {}
slurm_array_options['p'] = 'preemptable'
#slurm_options['qos'] = 'debug'
#slurm_options['account'] = 'm3938'
#slurm_options['constraint'] = 'haswell'
slurm_array_options['J'] = ''
slurm_array_options['t'] = '00-05:00:00'
#slurm_options['t'] = '00-00:10:00'
slurm_array_options['o'] = '%j.o'
slurm_array_options['e'] = '%j.e'
slurm_array_options['ntasks-per-node'] = 1

directory = os.path.join('/fusion/projects/boundary/peretm/simulations/', "slab_1D_scan_fixed_resolution")
setup_slurm_array_scripts(slurm_array_options,directory, "runtest_array_fixed_resolution",,n_jobs)


#UBox.sbatch(slurm_options)
# UBoxtest=UBoxLauncher()
# project = "slab_1D_scan_extended_test"
# UBoxtest.Launch()
# UBoxtest.CreateProject(project,force=True, overwrite=True)
# params_sampling={}
# params_sampling['ncore'] = 'log'
# params_sampling['pcore'] = 'log'
# params_sampling['fc'] = 'linear'
# UBoxtest.setup_random_runs(params,inputfile,casename="runtest", nruns=200, params_sampling=params_sampling)
# UBoxtest.sbatch(slurm_options)


