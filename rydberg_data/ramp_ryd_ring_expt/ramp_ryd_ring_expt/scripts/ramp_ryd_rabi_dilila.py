import numpy as np
from rydberg import *

import argparse
import os
import tenpy
from tenpy.algorithms import tdvp
from tenpy.networks.terms import TermList
from tenpy.networks.mps import MPS
from tenpy.tools.hdf5_io import load, save

# ramp functions
from RampFun import *

# the convention can vary
# Here, Sz = 1/2[[1, 0], 
#                [0, -1]]
# where [1, 0] = ground, [0, 1] = rydberg

def get_sigma_1_point(i, Szs):
    n_exp = 1/2 - np.mean(Szs)
    
    return (-1)**i*(1/2 - Szs[i] - n_exp)
    
def get_eps_1_point(j, Szs):
    n_exp = 1/2 - np.mean(Szs)

    return 0.5*(1 - 2*Szs[j]) + 0.5*(1 - 2*Szs[j+1]) - 2*n_exp

# <n_in_j> helper function
def get_nn(i, j, Szs, SzSzs):
    
    return 1/4*(1 - 2*Szs[i] - 2*Szs[j] + 4*SzSzs[i, j])

def get_sigma_2_point(i, j, Szs, SzSzs):
    n_exp = 1/2 - np.mean(Szs)
    
    ninj = 1/4 - Szs[i]/2 - Szs[j]/2 + SzSzs[i, j]
    
    ni = 1/2 - Szs[i]
    nj = 1/2 - Szs[j]
    
    return (-1)**(i+j)*(ninj - ni*n_exp - nj*n_exp + n_exp**2)

# this is a disconnected correlator
def get_eps_2_point(i, j, Szs, SzSzs):
    n_exp = 1/2 - np.mean(Szs)
    return get_nn(i, j, Szs, SzSzs) + get_nn(i, j+1, Szs, SzSzs)\
           + get_nn(i+1, j, Szs, SzSzs) + get_nn(i+1, j+1, Szs, SzSzs)\
           - 2*n_exp*(1/2 - Szs[i] + 1/2 - Szs[i+1] + 1/2 - Szs[j] + 1/2 - Szs[j+1])\
           + 4*n_exp**2

# delta0 varies, delta is fixed, ring
def main():
    parser = argparse.ArgumentParser(description="Calculates rydberg correlations (PBC) over time.")

    parser.add_argument('--dcrit_index', dest='dcrit_index', default=0, type=int)

    parser.add_argument('--chi_max', dest='chi_max', default=64, type=int)
    parser.add_argument('--L', dest='L', default=24, type=int)

    args = parser.parse_args()
    
    # initial state to load
    
    chi_max = args.chi_max
    chi_max_init = 64
    cutoff = 8.01
    
    # initialize the rydberg model 
    real_omega = 1.55 # MHz
    L = args.L
    Omega = 1
    U = 9/real_omega
    Rb = U**(1/6) #1.32
    
    delta_t = 0.1
    
    # rabi ramp parameters
    omega_ramp_time = 2*real_omega # (1/omega)
    omega_ramp_end_idx = int(2*real_omega/delta_t)
    
    # dilila parameters
    gapinit = 3.384
    gapcrit = 0.24 #0.9276 
    gapfin = 1.95
    
    dini = -5/real_omega
    #dcrit [0.5, 0.9, 1.1, 1.5]
    
    dcrits = [0.4, 1.34, 1.54, 1.74, 2.4] #np.linspace(0.5, 1.5, 20)
    dcrits = np.array(dcrits)/real_omega
    
    dcrit = dcrits[args.dcrit_index] #1.1 #0.5 # vary this
    dfin = 4.45/real_omega

    pulselen = 6.5*real_omega
    tinflect = (dcrit-dini)/(dfin-dini)*pulselen

    total_ramp_time = pulselen
    total_evolution_time = total_ramp_time
    
    num_ts = int(total_evolution_time/delta_t) + omega_ramp_end_idx

    # run TDVP 
    
    # requeueing: if psi has already been saved, load it
   
    save_filename = '../results/ramp_rydberg_ring_L_%d_Omega_%.2f_Rb_%.2f_dini_%.3f_dcrit_%.3f_dfin_%.3f_T_%.3f_chi_%04d_cutoff%.2f'%(L, Omega, Rb, dini, dcrit, dfin, pulselen, chi_max, cutoff)
    ''' 
    if os.path.isfile(save_filename + '.npz'):
        
        # load psi
        psi = load(save_filename + '.h5')
        
        # load measurements
        test = np.load(save_filename + '.npz')
        
        start_t_idx = test['t_idx'] + 1 # start after last saved index

        evals = test['evals'] # energy
        svals = test['svals'] # entropy
            
        sigma_1_points = test['sigma_1_points']
        sigma_2_points = test['sigma_2_points']
        
        sz_1_points = test['sz_1_points']
        sz_2_points = test['sz_2_points']
    '''
    if True:
        # load the saved psi with h0 
        #psi_file = '../../gs_ryd_ring_expt/results/gs_rydberg_ring_L_%d_Omega_%.2f_Rb_%.2f_delta_%.3f_chi_%04d_cutoff%.2f.h5'%(L, Omega, Rb, dini, chi_max_init, cutoff)
        
        #psi = load(psi_file)
        product_state = [[["up"]]] # no rydbergs
            
        model_params = {'N':L,
          'bc_MPS': 'finite',
          'Omega': 0,
          'delta': dini,
          'U': U,
          'terms_type': 'powerlaw',
          'lattice': 'Circle',
          'distance_cutoff': cutoff
           }
        
        M = Rydberg(model_params) #InfiniteChain(model_params)#
        psi = MPS.from_lat_product_state(M.lat, product_state)
        
        start_t_idx = 0

        evals = np.zeros([num_ts], dtype=complex) # energy
        svals = np.zeros([num_ts, L-1], dtype=complex) # entropy

        sigma_1_points = np.zeros([num_ts], dtype=complex)
        sigma_2_points = np.zeros([num_ts, L//2-1], dtype=complex)
        
        sz_1_points = np.zeros([num_ts, L], dtype=complex)
        sz_2_points = np.zeros([num_ts, L], dtype=complex)

    tdvp_params = {
        'trunc_params': {
            'svd_min': 1.e-8,
            'chi_max': chi_max,
        },
        'dt': 0.01,
        'N_steps': 10, # measure every dt = 0.05 instead of 0.01
        'start_time': 0.0,
        'max_sweeps': 2,
        'compression_method': 'variational'
    }
    
    
    for t_idx in range(start_t_idx, num_ts):
        
        print("On time index %d"%(t_idx), flush=True)

        # omega ramp 
        if t_idx < omega_ramp_end_idx: 
            # linear rabi ramp, initial delta0
            Omega_t = Omega*t_idx/omega_ramp_end_idx
            model_params = {'N':L,
              'bc_MPS': 'finite',
              'Omega': Omega_t,
              'delta': dini,
              'U': U,
              'terms_type': 'powerlaw',
              'lattice': 'Circle',
              'distance_cutoff': cutoff
              }
        else:
            t_cur = delta_t*(t_idx - omega_ramp_end_idx)

            delta_cur = getDiLILADetune(gapinit, gapcrit, gapfin, dcrit, dfin, dini, tinflect, pulselen, t_cur)

            model_params = {'N':L,
                      'bc_MPS': 'finite',
                      'Omega': Omega,
                      'delta': delta_cur,
                      'Rb': Rb,
                      'terms_type': 'powerlaw',
                      'lattice': 'Circle',
                      'distance_cutoff': cutoff
                        }

        M = Rydberg(model_params) 
        eng = tdvp.TwoSiteTDVPEngine(psi, M, tdvp_params)

        # make measurements
        # measurements we want
        evals[t_idx] = M.H_MPO.expectation_value(psi)
        svals[t_idx] = psi.entanglement_entropy()
        
        sz_1_points[t_idx, :] = psi.expectation_value('Sz')
        SzSzs = psi.correlation_function(['Sz'], ['Sz'], autoJW=False)
        sz_2_points[t_idx, :] = SzSzs[0, :]
        
        sigma_1_points[t_idx] = get_sigma_1_point(L//2, sz_1_points[t_idx, :])
        
        for shift in range(0, L//2-1):
            sigma_2_points[t_idx, shift] = get_sigma_2_point(L//2, L//2+shift, 
                    sz_1_points[t_idx, :], 
                                                                   SzSzs)  

        eng.run()
        
        if t_idx % 5 == 0:
            np.savez(save_filename,
             t_idx=t_idx,
             evals=evals, 
             svals=svals,
             sigma_1_points=sigma_1_points, 
             sigma_2_points=sigma_2_points,
             sz_1_points=sz_1_points, 
             sz_2_points=sz_2_points,
            )
            
            save(psi, save_filename+'.h5')
                
    np.savez(save_filename,
             t_idx=t_idx,
             evals=evals, 
             svals=svals,
             sigma_1_points=sigma_1_points, 
             sigma_2_points=sigma_2_points,
             sz_1_points=sz_1_points, 
             sz_2_points=sz_2_points,
            )
    save(psi, save_filename+'.h5')
                     
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
