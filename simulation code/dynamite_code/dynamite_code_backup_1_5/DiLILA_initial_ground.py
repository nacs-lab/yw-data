import sys
#sys.path.append('/n/holystore01/LABS/yao_lab/Everyone/Greg/dynamite/')
import os
from os import path,mkdir
from sys import argv

import shutil
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


fil_save_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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
C6 = 20*2*np.pi
Omega = 2*np.pi*params["Omega"]
dinit = 2*np.pi*params["dinit"]
gapinit = 2*np.pi*params['gapinit']
gapcrit = 2*np.pi*params['gapcrit']
gapfin = 2*np.pi*params['gapfin']
dcrit = 2*np.pi*params['dcrit']
dfin = 2*np.pi*params['dfin']
tinflect = params['tinflect']
T = params["T"]

dirc = params['dirc']
if not path.isdir(dirc):
    mkdir(dirc)
fil_save_name += '_L%d_%d'%(L1,L2)
shutil.copyfile(fil_name, path.join(dirc,'param_'+fil_save_name))
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
        #temp = 0.
        for site2 in range(site1+1,length):
            distance = np.sum((pos_lst[site1]-pos_lst[site2])**2)**0.5
            #print('site%d_%d_%.3f_%.3f:'%(site1,site2,func_v(distance),distance))
            #temp += func_V(distance)
            h += func_V(distance)*(sigmaz(site1)+identity())*(sigmaz(site2)+identity())/4
        #print('site%d:%.3f'%(site1,temp))
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

    state_evolve0 = State(state='D'*int(N))
    steps = int(T/dt)
    print(steps)
    data = {}
    data['z0'] = []
    data['zz0'] = []
    data['t'] = []
    data['Detuning'] = []
    data['overlap_g']=[]
    data['overlap_afm']=[]
    data['overlap_0']=[]
   
    data['z1'] = []
    data['zz1'] = []
    data['overlap_g1']=[]
    data['overlap_afm1']=[]
    data['overlap_01']=[]
   
    #data['overlap_2']=[]
 
   # data['overlap_3']=[]
    product_state = State(state='D'*int(L1*L2))
    product_state = State(state='D'*int(L1*L2))
    AFM_state = State(state =( 'UD'*int(L1*L2/2+1))[:L1*L2])
 
    H = H_V+func_Omega((0))*H_O+DILILA(0,T,dinit,gapinit,gapcrit,gapfin,dcrit,dfin,tinflect)*H_D   
    eigvals,eigvecs = H.eigsolve(getvecs=True,nev = 4)
    
    state_evolve1 = eigvecs[0] 
   
    for i in range(steps):
        print(i)
        H = H_V+func_Omega((i*dt))*H_O+DILILA(i*dt,T,dinit,gapinit,gapcrit,gapfin,dcrit,dfin,tinflect)*H_D
        eigvals,eigvecs = H.eigsolve(getvecs=True,nev = 4)
        data['t'].append(i*dt)
        data['Detuning'].append(DILILA(i*dt,T,dinit,gapinit,gapcrit,gapfin,dcrit,dfin,tinflect)/2/np.pi)
        data['z0'].append(measure_z(state_evolve0))
        data['zz0'].append(measure_zz(state_evolve0))
        data['overlap_g'].append(np.abs(state_evolve0.dot(product_state))**2)
        data['overlap_afm'].append(np.abs(state_evolve0.dot(AFM_state))**2)
        data['overlap_0'].append(np.abs(state_evolve0.dot(eigvecs[0]))**2)
       
        data['z1'].append(measure_z(state_evolve1))
        data['zz1'].append(measure_zz(state_evolve1))
        data['overlap_g1'].append(np.abs(state_evolve1.dot(product_state))**2)
        data['overlap_afm1'].append(np.abs(state_evolve1.dot(AFM_state))**2)
        data['overlap_01'].append(np.abs(state_evolve1.dot(eigvecs[0]))**2)
       
        H = H_V+func_Omega(((i+0.5)*dt))*H_O+DILILA((i+0.5)*dt,T,dinit,gapinit,gapcrit,gapfin,dcrit,dfin,tinflect)*H_D
        state_evolve0 = (H).evolve(state_evolve0, t=dt)
        state_evolve1 = (H).evolve(state_evolve1, t=dt)
    
    
    H = H_V+func_Omega(((steps)*dt))*H_O+DILILA((steps)*dt,T,dinit,gapinit,gapcrit,gapfin,dcrit,dfin,tinflect)*H_D
    eigvals,eigvecs = H.eigsolve(getvecs=True,nev = 4)
    data['t'].append((steps)*dt)
    data['Detuning'].append(DILILA(steps*dt,T,dinit,gapinit,gapcrit,gapfin,dcrit,dfin,tinflect)/2/np.pi)
    data['z0'].append(measure_z(state_evolve0))
    data['zz0'].append(measure_zz(state_evolve0))
    data['overlap_g'].append(np.abs(state_evolve0.dot(product_state))**2)
    data['overlap_afm'].append(np.abs(state_evolve0.dot(AFM_state))**2)
    data['overlap_0'].append(np.abs(state_evolve0.dot(eigvecs[0]))**2)
   # data['overlap_1'].append(np.abs(state_evolve.dot(eigvecs[1]))**2)
   # data['overlap_2'].append(np.abs(state_evolve.dot(eigvecs[2]))**2)
   # data['overlap_3'].append(np.abs(state_evolve.dot(eigvecs[3]))**2)
 
    data['z1'].append(measure_z(state_evolve1))
    data['zz1'].append(measure_zz(state_evolve1))
    data['overlap_g1'].append(np.abs(state_evolve1.dot(product_state))**2)
    data['overlap_afm1'].append(np.abs(state_evolve1.dot(AFM_state))**2)
    data['overlap_01'].append(np.abs(state_evolve1.dot(eigvecs[0]))**2)
       
    op_list0 = ['t','Detuning','overlap_g','overlap_afm','overlap_0']
    data_temp0 = np.zeros([len(op_list0),len(data['t'])])
    op_list1 = ['t','Detuning','overlap_g1','overlap_afm1','overlap_01']
    data_temp1 = np.zeros([len(op_list1),len(data['t'])])
    for i_ob,ob in enumerate(op_list0):
        data_temp0[i_ob] = data[ob]
        np.savetxt(path.join(dirc,'Overlap0'+fil_save_name), data_temp0,delimiter=',')
    op_list2 = ['z0','zz0','z1','zz1']
    for i_ob,ob in enumerate(op_list1):
        data_temp1[i_ob] = data[ob]
        np.savetxt(path.join(dirc,'Overlap1'+fil_save_name), data_temp1,delimiter=',')
   
    #fil_name = '_L%d_%d_den%.3f_delta%.1f_%.1f_omega%.3f_ramptime%.3f_dt%.3f_No%dtau1%3f.csv'%(L1,L2,hole_den,Delta_ini/(2*np.pi),Delta_fin/(2*np.pi),Omega/(2*np.pi),ramp_time,dt,No,tau1)
    #np.savetxt(path.join(dirc,'pos_occ_hole'+fil_name), pos_occupied, delimiter=',')
    for ob in op_list2:
        np.savetxt(path.join(dirc,ob+'_'+fil_save_name), data[ob], delimiter=',')
print('done')

endtime = datetime.datetime.now()
print((endtime-starttime).seconds)
