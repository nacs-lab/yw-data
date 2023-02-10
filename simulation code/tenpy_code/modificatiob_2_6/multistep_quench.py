#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import os
from os import path,mkdir
from sys import argv
import tenpy
from tenpy.tools import hdf5_io
import h5py
import datetime
import random
from tenpy.algorithms import dmrg
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite, SpinSite, set_common_charges
from tenpy.algorithms.tebd import RandomUnitaryEvolution
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import _parse_sites
from tenpy.models.lattice import get_lattice, Lattice, HelicalLattice, IrregularLattice
#tenpy.tools.misc.setup_logging(to_stdout="INFO")
import warnings
starttime = datetime.datetime.now()
warnings.filterwarnings("ignore")
class Rydberg1D(CouplingMPOModel):
    def init_sites(self, model_params):
        s = SpinHalfSite(conserve=None)
        s.add_op("P0", [[1, 0], [0, 0]])  # projector onto up spin
        s.add_op("P1", [[0, 0], [0, 1]])  # projector onto down spin
        return s

    def init_terms(self, model_params):
        # Omega
        time = model_params.get('time', None)
        if time is None:
            Omega = model_params.get('Omega', Omega)
        else:
            Omegafunc = model_params['Omega_function']
            Omega = Omegafunc(time)
            
        model_params['Omega'] = Omega  # allows to "measure" Omega
        if Omega != 0:
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(Omega, u, 'Sx', category="X")
                #self.add_onsite(Omega+np.random.random()*dOmega, u, 'Sx', category="X")
        #print('omega={}'.format(omega), flush=True)
        # delta
        time = model_params.get('time', None)
        if time is None:
            delta = model_params.get('Delta', delta)
        else:
            deltafunc = model_params['Delta_function']
            delta = deltafunc(time)

        model_params['Delta'] = delta  # allows to "measure" delta
        if delta != 0:
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(-delta, u, 'P1', category="P")
        #print('delta={}'.format(delta), flush=True)

        perimeter = model_params.get("L", 10)
        add_shift = model_params.get('add_shift',True)  # allows to "measure" delta
        if add_shift == True:
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(-100./perimeter, u, 'Id', category="shift")
        #print('delta={}'.format(delta), flush=True)


        # Use U instead of Rb because we might ramp Omega
        if 'U' not in model_params and 'Rb' in model_params:
            Rb = model_params['Rb']
            # V(x) = U/x^6 is interaction
            # Rb is defined such that Omega = V(Rb) = U/Rb^6
            U = Rb**6 * Omega
        else:
            U = model_params.get("U", 1.)
        #print('U={}'.format(U), flush=True)
#         E_shift = model_params.get("E_shift", 0.)

#         for u in range(len(self.lat.unit_cell)):
#             self.add_onsite(E_shift, u, 'Id', category="offset")

        distance_cutoff = model_params.get("distance_cutoff", 2.01)
        radius = 0.5/np.sin(np.pi/perimeter)
        if model_params.get("auto_cutoff", False):
            distance_cutoff = self.lat.Ls[0]-0.01
  #          print("distance_cutoff"+str(distance_cutoff))
        #print(self.lat.pairs.values)
#         for pairs in self.lat.pairs.values():
#             print(pairs)
#             for u1, u2, dx in pairs:
#                 distance = self.lat.distance(u1, u2, dx)
#                 print("distance"+str(distance))
#                 if distance > distance_cutoff:
#                     continue
#                 cord_distance = 2*radius*np.sin(2*np.pi/perimeter*distance/2)
#                 print("cord_distance"+str(cord_distance))
#                 strength = U / cord_distance**6
#                 self.add_coupling(strength, u1, "P1", u2, "P1", dx, category="PP")
        for i in range(perimeter):
            for j in range(i+1,perimeter):
                distance = float(abs(i-j))
                if distance > distance_cutoff and L-distance>distance_cutoff:
                    continue
                cord_distance = 2*radius*np.sin(2*np.pi/perimeter*distance/2)
 #               print("cord_distance"+str(cord_distance))
                strength = U / cord_distance**6
                self.add_coupling_term(strength,i,j,"P1","P1",category="PP")
                

fil_name = sys.argv[1]
fil = open(fil_name, 'r')
params = {}
for line in fil:
    vals = line.split(",")
    if vals[0] in ["chi","L","N_steps","N_times"]:
        params[vals[0]] = int(vals[1])
    elif vals[0] in ["haha"]:
        params[vals[0]] = []
        for i in range(1,len(vals)):
            params[vals[0]] += [int(vals[i])]
    elif vals[0] in ["dirc"]:
        params[vals[0]] = vals[1][:-1]
#     elif vals[0] in [""]:
#         params[vals[0]] = bool(vals[1])
    else:
        params[vals[0]] = float(vals[1])

dirc = params['dirc']
if not path.isdir(dirc):
    mkdir(dirc)
            
# Lattice params
L = params['L']
Omega = 2*np.pi*params['Omega']
# the 1/2 factor in the hamitonian is cancelled by converting the sx to sigmax 
U =2*np.pi*params['U']
chi = params['chi']
N_steps = params['N_steps'] # save the data after N_step 
Rabi_ramp_time = params['Rabi_ramp_time']
Rabi_ramp_dt = params['Rabi_ramp_dt']
# Dilila ramp parameter

T_UniLILA = params['T_UniLILA']
gapinit = 2*np.pi* params['Gap_init']
gapcrit = 2*np.pi* params['Gap_crit']
dcrit = 2*np.pi* params['Delta_crit']
dinit = 2*np.pi*params['Delta_ini']
dt = params['dt'] # detuning scanning dt
twait = params['twait']
dt_wait = params['dt_wait']
T_total = (T_UniLILA +  Rabi_ramp_time)*2+twait 
#
def Omegafunc(time):
    if 0 <= time < Rabi_ramp_time:
        return Omega*time/Rabi_ramp_time
    if Rabi_ramp_time<= time < T_total -Rabi_ramp_time:
        return Omega
    if T_total -Rabi_ramp_time <= time<= T_total:
        return Omega*(T_total-time)/Rabi_ramp_time
    else:
        print('Omega t out of range',str(t))
        return

def DILILA(t,t_len, dinit,gapinit,gapcrit,gapfin,dcrit,dfin,tinflect):
    numstep1 = gapinit* dcrit* t + gapcrit* dinit*(tinflect -t)
    denomsteps1 = gapinit* t + gapcrit * (tinflect - t)
    numstep2 = gapfin* dcrit*(t_len - t ) + gapcrit* dfin*(t-tinflect)
    denomsteps2 = gapfin*(t_len -t) +gapcrit*(t-tinflect)
    if t>t_len:
        out = dfin
    elif t>tinflect:
        out = numstep2/denomsteps2
    else:
        out = numstep1/denomsteps1
    return out
def UniLILA(t,t_len, dinit,gapinit,gapcrit,dcrit):
    numstep1 = gapinit* dcrit* t + gapcrit* dinit*(t_len -t)
    denomsteps1 = gapinit* t + gapcrit * (t_len - t)
    if t>t_len:
        out = dcrit
    else:
        out = numstep1/denomsteps1
    return out
def Deltafunc(time):
    if 0<=time<Rabi_ramp_time:
        return dinit
    if Rabi_ramp_time<= time <Rabi_ramp_time+T_UniLILA_stop:
        return UniLILA(time-Rabi_ramp_time,T_UniLILA,dinit,gapinit, gapcrit,dcrit)
    if Rabi_ramp_time+T_UniLILA_stop <= time < T_UniLILA_stop + Rabi_ramp_time+ t_wait:
        return UniLILA(T_UniLILA_stop,T_UniLILA,dinit,gapinit, gapcrit,dcrit)
    if  T_UniLILA_stop + Rabi_ramp_time+ t_wait <= time <  2*T_UniLILA_stop + Rabi_ramp_time+ t_wait:
        temp_t = T_total-time
        return UniLILA(temp_t-Rabi_ramp_time,T_UniLILA,dinit,gapinit, gapcrit,dcrit)
    if 2*T_UniLILA_stop + Rabi_ramp_time+ t_wait<= time <= T_total:
        return dinit
    else: 
        print('detuning t out of range ', str(t))
        return  
    
model_params = {'L':L,
              'bc_MPS': 'finite',
              'Omega_function': lambda time: Omegafunc(time),
              'Delta_function': lambda time: Deltafunc(time),
              #'Rb': Rb,
              'U': U,
              'add_shift': True,
              #'bc': 'periodic',
              #'bc_coupling': 'periodic',
              'distance_cutoff': 5.1,
              'auto_cutoff': False,
              'time': 0.0,
               }

M = Rydberg1D(model_params)
#print('Here:')
#print(M.coupling_terms['PP'])
dmrg_params = {
         'trunc_params': {
            'svd_min': 1.e-8,
            'chi_max': chi,
        },
        #'max_E_err': ,
        'max_sweeps': 50,
        #'max_sweeps': 100,
        'max_hours': 23,
        'mixer': True,
        #'mixer_params': {
        #    'amplitude': 0.1,
        #},
        'lanczos_params': {
            'N_max': 5,
        },
        'update_env': 0,
        'start_env': 1,
        'max_hours': 2,
        #'chi_list': {0:32, 20:64, 40:chi_max}
    }

tdvp_params_rabi = {
        'trunc_params': {
            'svd_min': 1.e-8,
            'chi_max': chi,
        },
        'dt': Rabi_ramp_dt/N_steps,
        'N_steps': N_steps,
        'start_time': 0.0,
        'max_sweeps': 2,
        'compression_method': 'variational'
    }
tdvp_params_UniLILA = {
        'trunc_params': {
            'svd_min': 1.e-8,
            'chi_max': chi,
        },
        'dt': dt/N_steps,
        'N_steps': N_steps,
        'start_time': Rabi_ramp_time,
        'max_sweeps': 2,
        'compression_method': 'variational'
    }
tdvp_params_wait = {
        'trunc_params': {
            'svd_min': 1.e-8,
            'chi_max': chi,
        },
        'dt': dt_wait/N_steps,
        'N_steps': N_steps,
        'start_time': Rabi_ramp_time+T_UniLILA_stop,
        'max_sweeps': 2,
        'compression_method': 'variational'
    }
tdvp_params_UniLILA_inverse = {
        'trunc_params': {
            'svd_min': 1.e-8,
            'chi_max': chi,
        },
        'dt': dt/N_steps,
        'N_steps': N_steps,
        'start_time': Rabi_ramp_time+T_UniLILA_stop+twait,
        'max_sweeps': 2,
        'compression_method': 'variational'
    }
tdvp_params_rabi_inverse = {
        'trunc_params': {
            'svd_min': 1.e-8,
            'chi_max': chi,
        },
        'dt': Rabi_ramp_dt/N_steps,
        'N_steps': N_steps,
        'start_time': Rabi_ramp_time+2*T_UniLILA_stop+twait,
        'max_sweeps': 2,
        'compression_method': 'variational'
    }

product_state_afm = [["up"],["down"]]*int(L/2)
product_state = [["up"]]
psi_afm = MPS.from_lat_product_state(M.lat, product_state_afm) # the afm state, which we take the overlap with
psi_fm = MPS.from_lat_product_state(M.lat, product_state) # the initial state, which we take the overlap with

psi = MPS.from_lat_product_state(M.lat, product_state)



psi_gs = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
psi_ex1 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
psi_ex2 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)


def find_gs():
    eng_dmrg_gs = dmrg.TwoSiteDMRGEngine(psi_gs, eng_tdvp.model, dmrg_params)
    E_gs,_ = eng_dmrg_gs.run()


    #print(dmrg_params)
    #print(dict(**dmrg_params,orthogonal_to=[psi_gs]))
    eng_dmrg_ex1 = dmrg.TwoSiteDMRGEngine(psi_ex1, eng_tdvp.model, dict(**dmrg_params,orthogonal_to=[psi_gs]))
    E_ex1,_ = eng_dmrg_ex1.run()

    eng_dmrg_ex2 = dmrg.TwoSiteDMRGEngine(psi_ex2, eng_tdvp.model, dict(**dmrg_params,orthogonal_to=[psi_gs,psi_ex1]))
    E_ex2,_ = eng_dmrg_ex2.run()
    
    return np.array([E_gs,E_ex1,E_ex2])

file_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_name +='_'+str(random.randint(0,1000))
file_name  += "L%d"%L
file_name +="multistep"

rerun = False
if rerun == True:
    data = np.load(dirc+file_name+".npy",allow_pickle=True).tolist()
    nT = data['nT']
    raise ValueError
else:
    data = {'z':[],'zz':[],'zg':[],'zzg':[],'overlap':[],'E':[],'nT':[],'params':params}
    nT = 0

def measure():
    data['z'].append(psi.expectation_value('P1'))
    data['zz'].append(psi.correlation_function('P1','P1'))
    data['zg'].append(psi_gs.expectation_value('P1'))
    data['zzg'].append(psi_gs.correlation_function('P1','P1'))
    data['overlap'].append(np.array([float(abs(psi.overlap(psi_fm))**2),float(abs(psi.overlap(psi_afm))**2),float(abs(psi.overlap(psi_gs))**2),float(abs(psi.overlap(psi_ex1))**2),float(abs(psi.overlap(psi_ex2))**2)]))
    data['E'].append(E)
    #print(eng_tdvp.model.options)
    data['nT'].append([n,eng_tdvp.model.options['time'],eng_tdvp.model.options['Omega']/2/np.pi,eng_tdvp.model.options['Delta']/2/np.pi])
    
N_times_rabi = int(Rabi_ramp_time/Rabi_ramp_dt)
N_times_UniLILA = int(T_UniLILA_stop/dt)
N_times_wait = int(twait/dt_wait)

eng_tdvp = tdvp.TimeDependentTwoSiteTDVP(psi, M, tdvp_params_rabi)
N_initial = 0
for n in range(N_initial,N_initial+N_times_rabi):
    psi_gs = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex1 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex2 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)

    E = find_gs()
    measure()
    np.save(dirc+file_name+".npy",data)
    print('Rabi scan, Run step: '+str(n)+'total '+ str(N_times_rabi))
    eng_tdvp.run()

eng_tdvp = tdvp.TimeDependentTwoSiteTDVP(psi, M, tdvp_params_UniLILA)
N_initial = N_times_rabi
for n in range(N_initial,N_times_UniLILA+N_initial):
    psi_gs = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex1 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex2 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    E = find_gs()
    measure()
    np.save(dirc+file_name+".npy",data)
    print('Uni Ramp, Run step: '+str(n-N_initial)+'total '+ str(N_times_UniLILA))
    eng_tdvp.run()
    
N_initial = N_times_UniLILA+N_times_rabi    
eng_tdvp = tdvp.TimeDependentTwoSiteTDVP(psi, M, tdvp_params_wait)
for n in range(N_initial,N_initial+N_times_wait):
    psi_gs = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex1 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex2 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    E = find_gs()
    measure()
    np.save(dirc+file_name+".npy",data)
    print('Wait, Run step: '+str(n-N_initial)+'total '+ str(N_times_wait))
    eng_tdvp.run()

eng_tdvp = tdvp.TimeDependentTwoSiteTDVP(psi, M, tdvp_params_UniLILA_inverse)
N_initial = N_times_UniLILA+N_times_rabi +N_times_wait
for n in range(N_initial,N_times_UniLILA+N_initial):
    psi_gs = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex1 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex2 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    E = find_gs()
    measure()
    np.save(dirc+file_name+".npy",data)
    print('Inverse Uni Ramp, Run step: '+str(n-N_initial)+'total '+ str(N_times_UniLILA))
    eng_tdvp.run()
    
eng_tdvp = tdvp.TimeDependentTwoSiteTDVP(psi, M, tdvp_params_rabi_inverse)
N_initial = 2*N_times_UniLILA+N_times_rabi +N_times_wait
for n in range(N_initial,N_times_rabi+N_initial):
    psi_gs = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex1 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex2 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    E = find_gs()
    measure()
    np.save(dirc+file_name+".npy",data)
    print('Inverse Rabi Ramp, Run step: '+str(n-N_initial)+'total '+ str(N_times_rabi))
    eng_tdvp.run()
    
    
E = find_gs()
measure()
np.save(dirc+file_name+".npy",data)
#print(data)
print(file_name)
endtime = datetime.datetime.now()
print((endtime-starttime).seconds)   

