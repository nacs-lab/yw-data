import sys
#sys.path.append('/n/holystore01/LABS/yao_lab/Everyone/Greg/dynamite/')
import os
from os import path,mkdir
from sys import argv

import numpy as np
from numpy import logspace

from dynamite import config
from dynamite.subspaces import Parity, Auto, SpinConserve
from dynamite.operators import sigmax, sigmay, sigmaz, index_sum, index_product, op_sum, identity
from dynamite.states import State
from dynamite.computations import entanglement_entropy
from dynamite.computations import reduced_density_matrix

from dynamite.msc_tools import dnm_int_t
import datetime
#sys.path.insert(0, '/n/lukin_lab2/Users/nyao/Greg/gpu_shell/.venv-opt/lib64/python3.4/site-packages')
#sys.path.insert(0, '/n/lukin_lab2/Users/nyao/Greg/dynamite/')

fil_name = sys.argv[1]
fil = open(fil_name, 'r')
params = {}
for line in fil:
    vals = line.split(",")
    if vals[0] in [""]:
        params[vals[0]] = int(vals[1])
    elif vals[0] in ["dirc","symmetry_full"]:
        params[vals[0]] = vals[1][:-1]
    elif vals[0] in ["L","No"]:
        #params[vals[0]] = [int(vals[1]),int(vals[2])]
        params[vals[0]] = []
        for i in range(1,len(vals)):
            params[vals[0]] += [int(vals[i])]
    elif vals[0] in ["ramp_time_list","Crit_delta_list"]:
        #params[vals[0]] = [int(vals[1]),int(vals[2])]
        params[vals[0]] = []
        for i in range(1,len(vals)):
            params[vals[0]] += [float(vals[i])]
    else:
        params[vals[0]] = float(vals[1])

L1 = params["L"][0]
L2 = params["L"][1]
#N = L1*L2
hole_den = params["hole_den"]
#pulse_err = params["pulse_err"]
#pos_err = params["pos_err"]
No_list = params["No"]
dt = params["dt"]
shapefactor = params["shapefactor"]


#C6 = 102000/3.8**6
#C6 = 25.8
C6 = 26*2*np.pi/(shapefactor**6)
Omega = 2*np.pi*params["Omega"]
Delta_ini = 2*np.pi*params["Delta_ini"]
Delta_fin = 2*np.pi*params["Delta_fin"]
ramp_time = params["ramp_time"]
T_extra = params["T_extra"]
tau1 = params["tau1"]
tau2 = params["tau2"]
Crit_t = params["Crit_t"]
Crit_delta_list = params["Crit_delta_list"]



dirc = params['dirc']
if not path.isdir(dirc):
    mkdir(dirc)

#config.initialize(gpu=True, slepc_args=['-mfn_ncv', '30'])

pos_occupied = np.random.random(size=L1*L2)>hole_den

config.shell = False
N = np.sum(pos_occupied)
config.L = N
print(N)
config.initialize(gpu=True)

from petsc4py.PETSc import Viewer
from petsc4py.PETSc import Vec

def generate_pos(L1,L2,pos_err=0):
    theta_temp = 2*np.pi*np.random.random(L1*L2)
    r_temp = np.random.normal(scale=pos_err,size = L1*L2)
    return np.array([np.kron(np.arange(L1),np.ones(L2))+r_temp*np.cos(theta_temp),np.kron(np.ones(L1),np.arange(L2))+r_temp*np.sin(theta_temp)]).transpose()

def func_V(distance):
    return C6*distance**-6.

def build_hamiltonian_interaction(pos_lst):
    h = 0*sigmaz(0)
    length = len(pos_lst)
    #print(length)
    for site1 in range(0,length):
        temp = 0.
        for site2 in range(site1+1,length):
            distance = np.sum((pos_lst[site1]-pos_lst[site2])**2)**0.5
            #print('site%d_%d_%.3f_%.3f:'%(site1,site2,func_v(distance),distance))
            temp += func_V(distance)
            h += func_V(distance)*(sigmaz(site1)+identity())*(sigmaz(site2)+identity())/4
        print('site%d:%.3f'%(site1,temp))
    return h
#def build_hamiltonian_interaction(pos_lst):
#    h = 0*sigmaz(0)
#    length = len(pos_lst)
#    #print(length)
#    for site1 in range(0,length):
#        temp = 0.
#        for site2 in range(0,length):
#            if site2 == site1:
#                continue
#            distance = np.sum((pos_lst[site1]-pos_lst[site2])**2)**0.5
#            #print('site%d_%d_%.3f_%.3f:'%(site1,site2,func_v(distance),distance))
#            temp += func_V(distance)
#            h += func_V(distance)*(sigmaz(site1)+identity())*(sigmaz(site2)+identity())/4
#        print('site%d:%.3f'%(site1,temp))
#    return h
                                    
def build_hamiltonian_field(Delta=1):
    H = 0*sigmaz(0)
    for site in range(N):
        H += -Delta*(sigmaz(site)+identity())/2
    return H

def build_hamiltonian_drive(Omega=1):
    H = 0*sigmax(0)
    for site in range(N):
        H += Omega*sigmax(site)
    return H

def measure_z(state):
    z_list = []
    for site in range(N):
        z_list.append(state.dot(sigmaz(site)*state).real)
    return z_list
#Y = index_sum(sigmay(0), size=N)

def measure_zz(state):
    zz_list = np.ones((N,N))
    for site1 in range(N):
        for site2 in range(N):
            zz_list[site1,site2] = state.dot((sigmaz(site1)*sigmaz(site2))*state).real
    return np.reshape(zz_list,N**2)

data_all = {}
data_all['z'] = []
data_all['zz'] = []
for i_Crit_delta,Crit_delta in enumerate(Crit_delta_list):
    T = ramp_time+T_extra
    def func_Omega(t):
        return Omega
    
    def func_Delta(t):
        if t<ramp_time:
            return Delta_ini+t/ramp_time*(Delta_fin-Delta_ini)
        else:
            return Delta_fin
    def expramp(t,final1,tau1,t1,final2,tau2,Lt,old):
    #old initial detuning, final2: final detuning, final1: critical point
    # Lt total sweep time, t1, sweep time from old to the initial detuning
    # tau1 decay rate in the first step, tau2 decay right in the second step, the small tau corresponds to adiabetic sweep through the critical point
        exp1 = np.exp(t1 / tau1)        
        exp1m = exp1-1
        part1 = -np.exp(-t/tau1)*exp1*(final1-old)/exp1m+(exp1*final1-old)/exp1m
        exp2m = np.exp((Lt-t1)/tau2)-1
        exp2 = np.exp(Lt/tau2)
        exp2a = np.exp(t1/tau2)
        part2 = (final2 - final1)/exp2m*np.exp((t-t1)/tau2)+(-exp2*final1+exp2a*final2)/(exp2a-exp2)
        out = old
        if t<t1:
            out = part1
        elif t<Lt:
            out = part2
        else:
            out = final2
        return out
    starttime = datetime.datetime.now()
    for No in No_list:
        pos_lst = generate_pos(L1,L2)
        print(pos_lst)
        H_V = build_hamiltonian_interaction(pos_lst[pos_occupied])
        H_D = build_hamiltonian_field()
        H_O = build_hamiltonian_drive()
    
        state_evolve = State(state='D'*int(N))
        steps = int(T/dt)
        data = {}
        data['z'] = []
        data['zz'] = []
        
        for i in range(steps):
            data['z'].append(measure_z(state_evolve))
            data['zz'].append(measure_zz(state_evolve))
            print(expramp(i*dt,Crit_delta,tau1,Crit_t,Delta_fin/2./np.pi,tau2,ramp_time,Delta_ini/2./np.pi))
            state_evolve = (H_V+func_Omega(i*dt)*H_O+2*np.pi*expramp(i*dt,Crit_delta,tau1,Crit_t,Delta_fin/2./np.pi,tau2,ramp_time,Delta_ini/2./np.pi)*H_D).evolve(state_evolve, t=dt)

        data['z'].append(measure_z(state_evolve))
        data['zz'].append(measure_zz(state_evolve))
        data_all['z'].append(data['z'][-1])
        data_all['zz'].append(data['zz'][-1])
        #np.save(dirc+'/alpha%.2f_Jz%.2f_L%d'%(alpha,Jz,L), data)
        #np.save(path.join(dirc,'alpha%.2f_Jz%.2f_L%d'%(alpha,Jz,L)), data, allow_pickle=True, fix_imports=True)
        #print(data['yz'])
        
        fil_name = '_L%d_%d_den%.3f_delta%.1f_%.1f_omega%.3f_crit_delta%.3f_dt%.3f_No%d.csv'%(L1,L2,hole_den,Delta_ini/(2*np.pi),Delta_fin/(2*np.pi),Omega/(2*np.pi),Crit_delta,dt,No)
        np.savetxt(path.join(dirc,'pos_occ_hole'+fil_name), pos_occupied, delimiter=',')
        for ob in ['z','zz']:
            print(data[ob])
            np.savetxt(path.join(dirc,ob+fil_name), data[ob], delimiter=',')
    fil_name = '_L%d_%d_den%.3f_delta%.1f_%.1f_omega%.3f_dt%.3f_No%dtau1_%3f.csv'%(L1,L2,hole_den,Delta_ini/(2*np.pi),Delta_fin/(2*np.pi),Omega/(2*np.pi),dt,No,tau1)
    np.savetxt(path.join(dirc,'pos_occ_hole'+fil_name), pos_occupied, delimiter=',')
    np.savetxt(path.join(dirc,'Crit_delta_list'+fil_name), Crit_delta_list, delimiter=',')
    for ob in ['z','zz']:
        print(data[ob])
        np.savetxt(path.join(dirc,'sweepcrit_delta_'+ob+fil_name), data_all[ob], delimiter=',')

print('done')

endtime = datetime.datetime.now()
print((endtime-starttime).seconds)
