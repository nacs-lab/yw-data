#!/usr/bin/python3

"""
submit batch jobs to the queue
"""

from subprocess import check_output
from functools import reduce
from itertools import product, repeat
from copy import deepcopy
from time import sleep
from os import path,mkdir
from sys import argv,exit

OUTPUT_DIR = '/n/home03/bye/tenpy_squeezing/workfolder_yu/output_cluster/'
#print('output subdirectory name: ')
#temp = raw_input()
OUTPUT_DIR = path.join(OUTPUT_DIR, input("output subdirectory name: "))

if '-dry-run' not in argv:
    if not path.isdir(OUTPUT_DIR):
        mkdir(OUTPUT_DIR)
    else:
        #print("directory exists. overwrite? (y/n)")
        #ans = raw_input()
        if input("directory exists. overwrite? (y/n)") not in ['y','Y']:
            exit()

def get_lists(keys_so_far, d):
    sub_lists = []
    for key, val in d.items():
        if isinstance(val, list):
            sub_lists.append( [keys_so_far+[key],val] )
        elif isinstance(val, dict):
            sub_lists += get_lists(keys_so_far+[key],val)
        else:
            sub_lists.append( [keys_so_far+[key],[val]] )
    return sub_lists

def make_product( d ):
    '''
    Take a dictionary and return the product of options in the lists.
    I know that didn't make much sense. it does in my head.
    '''
    lsts = get_lists([],d)

    return product(*[zip(repeat(x[0]),x[1]) for x in lsts])

def get_nested_dict( d, key_list ):
    return reduce(dict.__getitem__,key_list,d)


options = {
    'batch_options' : {
        '-J':'squeezing_tenpy',
        '-p':'serial_requeue',
        '-n':'1',
        '-c':'4',
        '-N':'1',
        '-t':'2-23:00',
        #'--priority':'TOP',
        '--mem':'8G',
        '--account':'yao_lab',
        '--contiguous':'',
        '--open-mode=appen': '',
    },

    'run_options' : {
        'dirc': '/n/home03/bye/tenpy_squeezing/workfolder_yu/output_cluster/',
        'L' : ['20','24','28','32'],
        'Omega' : '1.5',
        'U' : '15.',
        'chi': '128',
        'N_steps': '1',
        'Rabi_ramp_time': '2',
        'Rabi_ramp_dt': '0.1',
        'T_DiLILA':'10.',
        'Gap_init':'5.26',
        'Gap_crit':'0.767',
        'Gap_final':'1.63',
        'Delta_ini': '-5.',
        'Delta_crit': '1.3',
        'Delta_fin': '3.',
        'tinflect': '7.',
        'auto_tinflect': '1.',
        'dt': '0.052',
    }
}


# output these options so that we remember what they were
if '-dry-run' not in argv:
    with open(path.join(OUTPUT_DIR,'run_options'),'w') as f:
        f.write('batch_options:\n')
        for opt,val in options['batch_options'].items():
            f.write('   '+str(opt)+':'+str(val)+'\n')
        f.write('run_options:\n')
        for opt,val in options['run_options'].items():
            f.write('   '+str(opt)+':'+str(val)+'\n')
        
run_lst = []
for n,x in enumerate(make_product(options)):
    tmp_d = {'batch_options':{},'run_options':{}}
    for key_list,value in x:
        if isinstance(value,dict): # don't clobber other stuff in a dict there
            get_nested_dict(tmp_d,key_list[:-1])[key_list[-1]].update(value)
        else: # this is just a regular value
            get_nested_dict(tmp_d,key_list[:-1])[key_list[-1]] = value

    # give the job a unique name and places to output stuff
    tmp_d['batch_options']['-o'] = path.join(OUTPUT_DIR, '%a_'+tmp_d['batch_options']['-J']+'.out')
    tmp_d['batch_options']['-e'] = path.join(OUTPUT_DIR, '%a_'+tmp_d['batch_options']['-J']+'.err')
    tmp_d['batch_options']['-J'] = OUTPUT_DIR.split('/')[-1]+'_'+tmp_d['batch_options']['-J']

    run_lst.append(tmp_d)

for job_n,d in enumerate(run_lst):

    s = ''
    for opt,val in d['run_options'].items():
        s += ','.join([opt,val])+'\n'

    if not '-dry-run' in argv:
        with open(path.join(OUTPUT_DIR,str(job_n))+'.opts','w') as f:
            f.write(s)
    else:
        print(job_n)
        print(s)
        print()

batch_script = '#!/bin/bash'

for opt,val in d['batch_options'].items():
    batch_script += ' '.join(['\n#SBATCH',opt,val])

batch_script += '\nsource /n/home03/bye/tenpy_squeezing/tenpy_bash'
batch_script += '\nexport OMP_NUM_THREADS='+str(d['batch_options']['-c'])+' '
batch_script += '\nsrun --mpi=pmi2 -n '+str(d['batch_options']['-n'])
batch_script += ' python -O /n/home03/bye/tenpy_squeezing/workfolder_yu/rydberg-1D_DiLILA_with_Rabi_ramp.py '
batch_script += path.join(OUTPUT_DIR,'${SLURM_ARRAY_TASK_ID}.opts')

if '-dry-run' in argv:
    print(batch_script)
else:
    # write out the batch script to keep track of what we did
    with open(path.join(OUTPUT_DIR,d['batch_options']['-J'])+'.batch','w') as f:
        f.write(batch_script)
        
    print(check_output(['sbatch','--array=0-'+str(len(run_lst)-1)],
                       input=batch_script,
                       universal_newlines=True),
          '('+str(len(run_lst))+')')

