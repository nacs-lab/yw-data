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


#C6 = 102000/3.8**6
#C6 = 25.8
C6 = 2*np.pi*params["V"]
Omega = 2*np.pi*params["Omega"]
Delta_ini = 2*np.pi*params["Delta_ini"]
Delta_fin = 2*np.pi*params["Delta_fin"]
ramp_time = params["ramp_time"]
T = params["T"]

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
       # print('site%d:%.3f'%(site1,temp))
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
#    returh
                                    
def build_hamiltonian_field(Delta=1):
    H = 0*sigmaz(0)
    for site in range(N):
        H += -Delta*(sigmaz(site)+identity())/2
    return H

def build_hamiltonian_drive(Omega=1):
    H = 0*sigmax(0)
    for site in range(N):
        H += Omega/2*sigmax(site)
    return H

def func_Omega(t):
    return Omega

def func_Delta(t):
    if t<ramp_time:
        return Delta_ini+t/ramp_time*(Delta_fin-Delta_ini)
    else:
        return Delta_fin

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



starttime = datetime.datetime.now()
for No in No_list:
    pos_lst = generate_pos(L1,L2)
    print(pos_lst)
    H_V = build_hamiltonian_interaction(pos_lst[pos_occupied])
    H_D = build_hamiltonian_field()
    H_O = build_hamiltonian_drive()

    #state_evolve = State(state='D'*int(N))
    steps = int(T/dt)
    print('Total step:',steps)
    data = {}
    data['t'] = []
    data['Detuning'] = []
    data['E0']=[]
    data['E1']=[]
    data['overlap_g']=[]
    data['overlap_afm']=[]
    data['z0']=[]
    data['zz0']=[]
    data['z1']=[]
    data['zz1']=[]
    data['z2']=[]
    data['zz2']=[]
    product_state = State(state='D'*int(L1*L2))
    AFM_state = State(state =( 'UD'*int(L1*L2/2+1))[:L1*L2])
    for i in range(steps):
       # data['z'].append(measure_z(state_evolve))
        #data['zz'].append(measure_zz(state_evolve))

        H = (H_V+func_Omega((i+0.5)*dt)*H_O+func_Delta((i+0.5)*dt)*H_D)
        data['t'].append((i+0.5)*dt)
        data['Detuning'].append(func_Delta((i+0.5)*dt)/2/np.pi)
        eigvals,eigvecs = H.eigsolve(getvecs=True,nev = 8)
     #   print(eigvals)
        data['E0'].append(eigvals[0])
        data['E1'].append(eigvals[1])
        data['overlap_g'].append(np.abs(eigvecs[0].dot(product_state))**2)
        data['overlap_afm'].append(np.abs(eigvecs[0].dot(AFM_state))**2)
        print(i)
        data['z0'].append(measure_z(eigvecs[0]))
        data['zz0'].append(measure_zz(eigvecs[0]).reshape([-1]))

        data['z1'].append(measure_z(eigvecs[1]))
        data['zz1'].append(measure_zz(eigvecs[1]).reshape([-1]))


        data['z2'].append(measure_z(eigvecs[2]))
        data['zz2'].append(measure_zz(eigvecs[2]).reshape([-1]))
    #np.save(dirc+'/alpha%.2f_Jz%.2f_L%d'%(alpha,Jz,L), data)
    #np.save(path.join(dirc,'alpha%.2f_Jz%.2f_L%d'%(alpha,Jz,L)), data, allow_pickle=True, fix_imports=True)
    #print(data['yz'])
    
    fil_name = '_L%d_%d_den%.3f_delta%.1f_%.1f_omega%.3f_ramptime%.3f_dt%.3f_No%d.csv'%(L1,L2,hole_den,Delta_ini/(2*np.pi),Delta_fin/(2*np.pi),Omega/(2*np.pi),ramp_time,dt,No)
   # np.savetxt(path.join(dirc,'pos_occ_hole'+fil_name), pos_occupied, delimiter=',')
    op_list1 = ['t','Detuning','E0','E1','overlap_g','overlap_afm']
    op_list2 = ['z0','zz0','z1','zz1','z2','zz2']
    data_temp = np.zeros([len(op_list1),len(data['t'])])
    for i_ob,ob in enumerate(op_list1):
        data_temp[i_ob] = data[ob]
   #     print(data_temp)
        np.savetxt(path.join(dirc,'AFM_Energy'+fil_name), data_temp,delimiter=',')
    for ob in op_list2:
        #print()
        np.savetxt(path.join(dirc,ob+fil_name), data[ob], delimiter=',')
print('done')
print(fil_name)
endtime = datetime.datetime.now()
print((endtime-starttime).seconds)
