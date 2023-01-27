import numpy as np
import sys
import os
from os import path,mkdir
from sys import argv
import tenpy
from tenpy.tools import hdf5_io
import h5py
import datetime
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
    else:
        params[vals[0]] = float(vals[1])

dirc = params['dirc']
if not path.isdir(dirc):
    mkdir(dirc)
            
# Lattice params
L = params['L']
print(L)
Omega = 2*np.pi*params['Omega']
# the 1/2 factor in the hamitonian is cancelled by converting the sx to sigmax 
Delta_ini =2*np.pi*params['Delta_ini']
Delta_fin =2*np.pi*params['Delta_fin']
U =2*np.pi*params['U']
ramp_time = params['ramp_time']
chi = params['chi']
dt = params['dt']
N_steps = params['N_steps']
T = params['T']
#
def Omegafunc(time):
    return Omega

def Deltafunc(time):
    if time < ramp_time:
        return Delta_ini+time/ramp_time*(Delta_fin-Delta_ini)
    else:
        return Delta_fin

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

tdvp_params = {
        'trunc_params': {
            'svd_min': 1.e-8,
            'chi_max': chi,
        },
        'dt': dt/N_steps,
        'N_steps': N_steps,
        'start_time': 0.0,
        'max_sweeps': 2,
        'compression_method': 'variational'
    }
product_state_afm = [["up"],["down"]]*int(L/2)
product_state = [["up"]]
psi_afm = MPS.from_lat_product_state(M.lat, product_state_afm) # the initial state, which we take the overlap with

psi_fm = MPS.from_lat_product_state(M.lat, product_state) # the initial state, which we take the overlap with

psi = MPS.from_lat_product_state(M.lat, product_state)
eng_tdvp = tdvp.TimeDependentTwoSiteTDVP(psi, M, tdvp_params)
#eng_tdvp = tdvp.TwoSiteTDVPEngine(psi, M, tdvp_params)

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
file_name  += "L%d"%L

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
 # the initial state, which we take the overlap with

    data['E'].append(E+100)
    #print(eng_tdvp.model.options)
    data['nT'].append([n,eng_tdvp.model.options['time'],eng_tdvp.model.options['Omega']/np.pi,eng_tdvp.model.options['Delta']/2/np.pi])
N_times = int(T/dt)


for n in range(nT,N_times):
    psi_gs = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex1 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)
    psi_ex2 = MPS.from_lat_product_state(eng_tdvp.model.lat, product_state)

    E = find_gs()
    measure()
    np.save(dirc+file_name+".npy",data)
    print('Run step: '+str(n))
    eng_tdvp.run()

E = find_gs()
measure()
np.save(dirc+file_name+".npy",data)
#print(data)
print(file_name)
   
