"""Model definition and simulation setup for Rydberg atoms on the Kagome lattice.
"""
import numpy as np
import tenpy
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite, SpinSite
from tenpy.networks.mps import InitialStateBuilder
from tenpy.simulations.simulation import * # run_simulation, run_seq_simulations etc
from tenpy.models.lattice import get_lattice, Lattice, HelicalLattice, IrregularLattice
from scipy.interpolate import interp1d
from tenpy.tools.misc import to_array
from tenpy.tools.fit import fit_with_sum_of_exp

from custom_lattices import RubyXC_rho, RubyXC_blockade, RubyXC_group_rho, Circle

tenpy.models.TiltedLattice = HelicalLattice  # backwards compatibility: allow loading old files

# Use this!
# Note: time evolution is hacked together and very inelegant: should rewrite
class Rydberg(CouplingMPOModel):
    def init_lattice(self, model_params):
        use_helix = model_params.get('use_helix', False)
        if use_helix:
            return self.init_helix_lattice(model_params)

        lattice = model_params.get('lattice', "Kagome2")
        if lattice=='RubyXC_rho' or lattice=='RubyXC_group_rho':
            # Create the regular lattice
            LatticeClass = get_lattice(lattice_name=lattice)
            order = model_params.get('order', 'default')
            sites = self.init_sites(model_params)
            bc_MPS = model_params.get('bc_mps', 'infinite')
            bc_y = model_params.get('bc_y', 'cylinder')
            if bc_y == 'cylinder':
                bc_y = 'periodic'
            bc = ['periodic', bc_y]
            Lx = model_params.get('Lx', 2)
            Ly = model_params.get('Ly', 4)
            rho = model_params.get('rho', 1.)  # Special for RubyXC_rho
            lat = LatticeClass(Lx, Ly, sites, rho=rho, order=order, bc=bc, bc_MPS=bc_MPS)
        elif lattice=='Circle':
            sites = self.init_sites(model_params)
            # N: number of sites in circle
            N = model_params.get('N', 2)
            lat = Circle(Lx=1, Ly=1, sites=sites, N=N, bc=['open', 'open'], bc_MPS='finite')
        else:
            lat = super().init_lattice(model_params)

        remove_sites = model_params.get('remove_sites', False)
        if remove_sites:
            lat = IrregularLattice(lat, remove=remove_sites)

        order_override = model_params.get('order_override', None)
        if order_override is not None:
            lat.order = order_override

        return lat

    def init_helix_lattice(self, model_params):
        # create regular lattice
        lat = model_params.get('lattice', self.default_lattice)
        if isinstance(lat, str):
            LatticeClass = get_lattice(lattice_name=lat)
            assert LatticeClass.dim == 2
            order = model_params.get('order', 'Cstyle')
            sites = self.init_sites(model_params)
            bc_MPS = 'infinite'
            bc = ['periodic', -1]
            Lx = model_params.get('Lx', 1)
            Ly = model_params.get('Ly', 4)
            lat = LatticeClass(Lx, Ly, sites, order=order, bc=["periodic", -1], bc_MPS=bc_MPS)
        # else: a lattice was already provided
        assert isinstance(lat, Lattice)
        N_unit_cells = model_params.get('N_unit_cells', 1 + lat.Lu%2)
        helix_lat = HelicalLattice(lat, N_unit_cells)
        return helix_lat

    def init_sites(self, model_params):
        s = SpinHalfSite(conserve=None)
        s.add_op("P0", [[1, 0], [0, 0]])  # projector onto up spin
        s.add_op("P1", [[0, 0], [0, 1]])  # projector onto down spin
        return s

    def init_terms(self, model_params):
        Lx, Ly = self.lat.Ls
        Lu = len(self.lat.unit_cell)

        # Omega
        time = model_params.get('time', None)
        if time is None:
            Omega = 0.
            if 'Omega_initial' in model_params:
                Omega = model_params['Omega_initial']
            if 'Omegapoints' in model_params:
                tpoints = model_params['tpoints']
                Omegapoints = model_params['Omegapoints']
                Omegat = interp1d(tpoints, Omegapoints)
                Omega = Omegat(0.0)
            Omega = model_params.get('Omega', Omega)
        else:
            # time evolution
            Omega_ramp_type = model_params.get('Omega_ramp_type', 'constant')
            if Omega_ramp_type == 'constant':
                Omega = 0.
                if 'Omega_initial' in model_params:
                    Omega = model_params['Omega_initial']
                Omega = model_params.get('Omega', Omega)
            elif Omega_ramp_type == 'linear':
                Omega = model_params['Omega_a'] + model_params['Omega_b']*time
            elif Omega_ramp_type == 'linear_down':
                ramp_final = model_params['ramp_final_time']
                factor = max(0., 1. - time / ramp_final)
                Omega = model_params['Omega_initial'] * factor
            elif Omega_ramp_type == 'exp_decay':
                tau = model_params['Omega_ramp_tau']
                factor = np.exp(-time/tau)
                Omega = model_params['Omega_initial'] * factor
            elif Omega_ramp_type == 'exp':
                tau = model_params['Omega_ramp_tau']
                factor = np.exp(-time/tau)
                Omega_final = model_params['Omega_final']
                Omega = Omega_final + (model_params['Omega_initial']-Omega_final) * factor
            elif Omega_ramp_type == 'custom_function':
                Omegafunc = model_params['Omega_function']
                Omega = Omegafunc(time)
            elif Omega_ramp_type == 'points':
                tpoints_Omega = model_params['tpoints_Omega']
                Omegapoints = model_params['Omegapoints']
                Omegat = interp1d(tpoints_Omega, Omegapoints)
                Omega = Omegat(time)
            model_params['Omega'] = Omega  # allows to "measure" Omega
        if Omega != 0:
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(Omega, u, 'Sx', category="X")
        print('Omega={}'.format(Omega), flush=True)

        # delta
        time = model_params.get('time', None)
        if time is None:
            delta = 0.
            if 'delta_initial' in model_params:
                delta = 0.
                if 'delta_initial' in model_params:
                    delta = model_params['delta_initial']
                delta = model_params.get('delta', delta)
            if 'deltapoints' in model_params:
                tpoints = model_params['tpoints']
                deltapoints = model_params['deltapoints']
                deltat = interp1d(tpoints, deltapoints)
                delta = deltat(0.0)
            delta = model_params.get('delta', delta)
        else:
            # time evolution
            delta_ramp_type = model_params.get('delta_ramp_type', 'constant')
            if delta_ramp_type == 'constant':
                delta = model_params['delta_initial']
            elif delta_ramp_type == 'linear':
                delta = model_params['delta_a'] + model_params['delta_b']*time
            elif delta_ramp_type == 'linear_down':
                ramp_final = model_params['ramp_final_time']
                factor = max(0., 1. - time / ramp_final)
                delta = model_params['delta_initial'] * factor
            elif delta_ramp_type == 'exp_decay':
                tau = model_params['delta_ramp_tau']
                factor = np.exp(-time/tau)
                delta = model_params['delta_initial'] * factor
            elif delta_ramp_type == 'exp':
                tau = model_params['delta_ramp_tau']
                factor = np.exp(-time/tau)
                delta_final = model_params['delta_final']
                delta = delta_final + (model_params['delta_initial']-delta_final) * factor
            elif delta_ramp_type == 'custom_function':
                deltafunc = model_params['delta_function']
                delta = deltafunc(time)
            elif delta_ramp_type == 'points':
                tpoints_delta = model_params['tpoints_delta']
                deltapoints = model_params['deltapoints']
                deltat = interp1d(tpoints_delta, deltapoints)
                delta = deltat(time)
            model_params['delta'] = delta  # allows to "measure" delta
        if delta != 0:
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(-delta, u, 'P1', category="P")
        print('delta={}'.format(delta), flush=True)

        # Use U instead of Rb because we might ramp Omega
        if 'U' not in model_params and 'Rb' in model_params:
            Rb = model_params['Rb']
            # V(x) = U/x^6 is interaction
            # Rb is defined such that Omega = V(Rb) = U/Rb^6
            U = Rb**6 * Omega
        else:
            U = model_params.get("U", 1.)
        print('U={}'.format(U), flush=True)
        E_shift = model_params.get("E_shift", 0.)
        distance_cutoff = model_params.get("distance_cutoff", 2.01)
        cap = model_params.get("cap", -1)
        if model_params.get("auto_cutoff_cylinder", False):
            if model_params.get("use_helix", False):
                # r_mod is the vector "modded out" from the plane to form the cylinder
                r_mod = (self.lat.Ls[1] * self.lat.basis[1]) - self.lat.basis[0] 
                distance_cutoff = 0.49 * np.linalg.norm(r_mod)
            else:
                r_mod = self.lat.Ls[1] * self.lat.basis[1]
                distance_cutoff = 0.49 * np.linalg.norm(r_mod)
            print("Using max cylinder cutoff: %.2f"%distance_cutoff)
        if model_params.get("auto_cutoff_open", False):
            # all-to-all interactions for square
            distance_cutoff = 1.01 * np.sqrt( (Lx-1)**2 + (Ly-1)**2) 

        for u in range(len(self.lat.unit_cell)):
            # self.add_onsite(Omega, u, 'Sx', category="X")
            # self.add_onsite(-delta, u, 'P1', category="P")
            self.add_onsite(E_shift, u, 'Id', category="offset")

        add_position_disorder(self, model_params)
        
        for pairs in self.lat.pairs.values():
            for u1, u2, dx in pairs:
                distance = self.lat.distance(u1, u2, dx)
                if distance > distance_cutoff:
                    continue
                if cap == -1:
                    strength = U / distance**6
                else:
                    strength = min(U / distance**6, cap)
                self.add_coupling(strength, u1, "P1", u2, "P1", dx, category="PP")

# 1D with sum of exponentials
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
            Omega = 0.
            if 'Omega_initial' in model_params:
                Omega = model_params['Omega_initial']
            if 'Omegapoints' in model_params:
                tpoints = model_params['tpoints']
                Omegapoints = model_params['Omegapoints']
                Omegat = interp1d(tpoints, Omegapoints)
                Omega = Omegat(0.0)
            Omega = model_params.get('Omega', Omega)
        else:
            # time evolution
            Omega_ramp_type = model_params.get('Omega_ramp_type', 'constant')
            if Omega_ramp_type == 'constant':
                Omega = 0.
                if 'Omega_initial' in model_params:
                    Omega = model_params['Omega_initial']
                Omega = model_params.get('Omega', Omega)
            elif Omega_ramp_type == 'linear':
                Omega = model_params['Omega_a'] + model_params['Omega_b']*time
            elif Omega_ramp_type == 'linear_down':
                ramp_final = model_params['ramp_final_time']
                factor = max(0., 1. - time / ramp_final)
                Omega = model_params['Omega_initial'] * factor
            elif Omega_ramp_type == 'exp_decay':
                tau = model_params['Omega_ramp_tau']
                factor = np.exp(-time/tau)
                Omega = model_params['Omega_initial'] * factor
            elif Omega_ramp_type == 'exp':
                tau = model_params['Omega_ramp_tau']
                factor = np.exp(-time/tau)
                Omega_final = model_params['Omega_final']
                Omega = Omega_final + (model_params['Omega_initial']-Omega_final) * factor
            elif Omega_ramp_type == 'custom_function':
                Omegafunc = model_params['Omega_function']
                Omega = Omegafunc(time)
            elif Omega_ramp_type == 'points':
                tpoints_Omega = model_params['tpoints_Omega']
                Omegapoints = model_params['Omegapoints']
                Omegat = interp1d(tpoints_Omega, Omegapoints)
                Omega = Omegat(time)
        model_params['Omega'] = Omega  # allows to "measure" Omega
        if Omega != 0:
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(Omega, u, 'Sx', category="X")
        print('Omega={}'.format(Omega), flush=True)

        # delta
        time = model_params.get('time', None)
        if time is None:
            delta = 0.
            if 'delta_initial' in model_params:
                delta = 0.
                if 'delta_initial' in model_params:
                    delta = model_params['delta_initial']
                delta = model_params.get('delta', delta)
            if 'deltapoints' in model_params:
                tpoints = model_params['tpoints']
                deltapoints = model_params['deltapoints']
                deltat = interp1d(tpoints, deltapoints)
                delta = deltat(0.0)
            delta = model_params.get('delta', delta)
        else:
            # time evolution
            delta_ramp_type = model_params.get('delta_ramp_type', 'constant')
            if delta_ramp_type == 'constant':
                delta = model_params['delta_initial']
            elif delta_ramp_type == 'linear':
                delta = model_params['delta_a'] + model_params['delta_b']*time
            elif delta_ramp_type == 'linear_down':
                ramp_final = model_params['ramp_final_time']
                factor = max(0., 1. - time / ramp_final)
                delta = model_params['delta_initial'] * factor
            elif delta_ramp_type == 'exp_decay':
                tau = model_params['delta_ramp_tau']
                factor = np.exp(-time/tau)
                delta = model_params['delta_initial'] * factor
            elif delta_ramp_type == 'exp':
                tau = model_params['delta_ramp_tau']
                factor = np.exp(-time/tau)
                delta_final = model_params['delta_final']
                delta = delta_final + (model_params['delta_initial']-delta_final) * factor
            elif delta_ramp_type == 'custom_function':
                deltafunc = model_params['delta_function']
                delta = deltafunc(time)
            elif delta_ramp_type == 'points':
                tpoints_delta = model_params['tpoints_delta']
                deltapoints = model_params['deltapoints']
                deltat = interp1d(tpoints_delta, deltapoints)
                delta = deltat(time)
        model_params['delta'] = delta  # allows to "measure" delta
        if delta != 0:
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(-delta, u, 'P1', category="P")
        print('delta={}'.format(delta), flush=True)

        # Use U instead of Rb because we might ramp Omega
        if 'U' not in model_params and 'Rb' in model_params:
            Rb = model_params['Rb']
            # V(x) = U/x^6 is interaction
            # Rb is defined such that Omega = V(Rb) = U/Rb^6
            U = Rb**6 * Omega
        else:
            U = model_params.get("U", 1.)
        print('U={}'.format(U), flush=True)
        E_shift = model_params.get("E_shift", 0.)

        for u in range(len(self.lat.unit_cell)):
            # self.add_onsite(Omega, u, 'Sx', category="X")
            # self.add_onsite(-delta, u, 'P1', category="P")
            self.add_onsite(E_shift, u, 'Id', category="offset")

        terms_type = model_params.get("terms_type", "exp")
        if terms_type == 'exp':
            def powerlaw(x):
                return np.power(x, -6.)
            n_exp = model_params.get('n_exp', 5)
            fit_range = model_params.get('fit_range', 50)
            lam, pref = fit_with_sum_of_exp(powerlaw, n_exp, fit_range)
            for pr, la in zip(pref, lam):
                self.add_exponentially_decaying_coupling(pr*U, la, 'P1', 'P1')
        else:
            distance_cutoff = model_params.get("distance_cutoff", 2.01)
            cap = model_params.get("cap", -1)
            if model_params.get("auto_cutoff", False):
                distance_cutoff = self.lat.Ls[0]-0.01
            for pairs in self.lat.pairs.values():
                for u1, u2, dx in pairs:
                    distance = self.lat.distance(u1, u2, dx)
                    if distance > distance_cutoff:
                        continue
                    if cap == -1:
                        strength = U / distance**6
                    else:
                        strength = min(U / distance**6, cap)
                    self.add_coupling(strength, u1, "P1", u2, "P1", dx, category="PP")
            

def to_blockade(u):
    ufinal = 0 if u<3 else 1
    p = "P" + str(int(u%3+1))
    return (ufinal, p)

# rubyxc blockade Only
class RydbergHilbert(CouplingMPOModel):
    def init_lattice(self, model_params):
        order = model_params.get('order', 'default')
        sites = self.init_sites(model_params)
        bc_MPS = model_params.get('bc_mps', 'infinite')
        bc_y = model_params.get('bc_y', 'cylinder')
        if bc_y == 'cylinder':
            bc_y = 'periodic'
        bc = ['periodic', bc_y]
        Lx = model_params.get('Lx', 2)
        Ly = model_params.get('Ly', 4)
        rho = model_params.get('rho', 1.)
        lat = RubyXC_blockade(Lx, Ly, sites, rho=rho, order=order, bc=bc, bc_MPS=bc_MPS)
        return lat
    
    def init_sites(self, model_params):
        s = SpinSite(S=1.5, conserve=None)
        s.add_op("P0", [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  # projector onto 0 state
        s.add_op("P1", [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  # projector onto 1 state
        s.add_op("P2", [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])  # projector onto 2 state
        s.add_op("P3", [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])  # projector onto 3 state
        s.add_op("Sigmax1", [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        s.add_op("Sigmax2", [[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        s.add_op("Sigmax3", [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        s.add_op("Sx1", [[0, 0.5, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        s.add_op("Sx2", [[0, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0]])
        s.add_op("Sx3", [[0, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0]])
        s.add_op("Sigmay1", [[0, -1.j, 0, 0], [1.j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        s.add_op("Sigmay2", [[0, 0, -1.j, 0], [0, 0, 0, 0], [1.j, 0, 0, 0], [0, 0, 0, 0]])
        s.add_op("Sigmay3", [[0, 0, 0, -1.j], [0, 0, 0, 0], [0, 0, 0, 0], [1.j, 0, 0, 0]])
        s.add_op("Sy1", [[0, -0.5j, 0, 0], [0.5j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        s.add_op("Sy2", [[0, 0, -0.5j, 0], [0, 0, 0, 0], [0.5j, 0, 0, 0], [0, 0, 0, 0]])
        s.add_op("Sy3", [[0, 0, 0, -0.5j], [0, 0, 0, 0], [0, 0, 0, 0], [0.5j, 0, 0, 0]])
        s.add_op("Sigmaz1", [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        s.add_op("Sigmaz2", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        s.add_op("Sigmaz3", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        s.add_op("Sz1", [[0.5, 0, 0, 0], [0, -0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]])
        s.add_op("Sz2", [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, -0.5, 0], [0, 0, 0, 0.5]])
        s.add_op("Sz3", [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, -0.5]])
        return s

    def init_terms(self, model_params):
        Omega = model_params.get("Omega", 1.)
        delta = model_params.get("delta", 0.)
        Rb = model_params.get("Rb", 1.)
        E_shift = model_params.get("E_shift", 0.)
        distance_cutoff = model_params.get("distance_cutoff", 2.01)
        cap = model_params.get("cap", -1)
        # V(x) = U/x^6 is interaction
        # Rb is defined such that Omega = V(Rb) = U/Rb^6
        U = Rb**6 * Omega

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(Omega, u, 'Sx1', category="X")
            self.add_onsite(Omega, u, 'Sx2', category="X")
            self.add_onsite(Omega, u, 'Sx3', category="X")
            self.add_onsite(-delta, u, 'P1', category="P")
            self.add_onsite(-delta, u, 'P2', category="P")
            self.add_onsite(-delta, u, 'P3', category="P")
            self.add_onsite(3*E_shift, u, 'Id', category="offset")

        Lx = model_params.get('Lx', 2)
        Ly = model_params.get('Ly', 4)
        rho = model_params.get('rho', 1.)
        lattice = RubyXC_rho(Lx, Ly, sites=None, rho=rho, bc='periodic')

        for pairs in lattice.pairs.values():
            for u1, u2, dx in pairs:
                distance = lattice.distance(u1, u2, dx)
                if distance > distance_cutoff:
                    continue
                if cap == -1:
                    strength = U / distance**6
                else:
                    strength = min(U / distance**6, cap)
                if distance > 1.05:
                    self.add_coupling(strength, to_blockade(u1)[0], to_blockade(u1)[1], to_blockade(u2)[0], to_blockade(u2)[1], dx, category="PP")

def checkEnvironments(engine):
    lp = engine.env.get_LP(0)
    rp = engine.env.get_RP(-1)
    h = engine.env.H
    idl = h.get_IdL(0)
    idr = h.get_IdR(-1)
    lppart = lp.take_slice(idl, axes='wR')
    rppart = rp.take_slice(idr, axes='wL')
    lnorm = np.linalg.norm(lppart.to_ndarray() - np.eye(lppart.shape[0]))
    rnorm = np.linalg.norm(rppart.to_ndarray() - np.eye(rppart.shape[0]))
    print("Check environment")
    print(lnorm, rnorm)

def add_position_disorder(model, model_params):
    """Add lattice.position_disorder to a model.

    Reads out model params `position_disorder_sigma` and `position_disorder_z_factor`.
    Reasonable values are position_disorder_sigma = 100nm/11.5um.

    Disorder/shot averaging can be done with different RNG seeds (simulation_params['seed']).

    For a ramp-down, we also check model parameters `time` and `position_disorder_velocity`.
    """
    sigma = model_params.get("position_disorder_sigma", 0.)
    z_factor = model_params.get("position_disorder_z_factor", 1.)
    speed = model_params.get("position_disorder_velocity", 0.)
    time = model_params.get('time', None)
    lat = model.lat
    if sigma and lat.position_disorder is None:
        # note: only use model.rng to ensure reporduciability for time-dependent H
        if z_factor:  # non-zero / not None
            # first extend lattice basis/unit_cell_position to include z
            lat.basis = np.pad(lat.basis, ((0, 0), (0, 1)), 'constant') # pad value defaults to 0
            lat.unit_cell_positions = np.pad(lat.unit_cell_positions, ((0, 0), (0, 1)), 'constant')
            # and then add 3D position_disorder
            lat.position_disorder = model.rng.normal(scale=sigma, size=lat.shape + (3,))
            lat.position_disorder[..., 2] *= z_factor
        else:
            lat.position_disorder = model.rng.normal(scale=sigma, size=lat.shape + (2,))
    if speed and time:
        if not hasattr(lat, 'position_disorder_velocity'):
            # again, only draw the random numbers once at the beginning!
            lat.position_disorder_time0 = lat.position_disorder.copy()
            lat.position_disorder_velocity = model.rng.normal(scale=speed,
                                                              size=lat.position_disorder.shape)
            if z_factor:
                lat.position_disorder_velocity[..., 2] *= z_factor
        lat.position_disorder = lat.position_disorder_time0 + time * lat.position_disorder_velocity

    type_  = model_params.get("squeeze_to_VBS_type", None)
    squeeze_strength = model_params.get("squeeze_to_VBS_strength", 0.)
    if type_ is None:
        return  # don't squeeze anything
    squeeze_types = {
        'glider_42': np.array([(0, 3, 1, 0), (1, 3, 0, 3), (0, 4, 1, 0), (1, 4, 0, 3),
                               (1, 2, 0, 1), (1, 2, 2, 4), (1, 3, 2, 5), (1, 3, 1, 2),
                               (1, 4, 2, 5), (1, 4, 1, 2), (1, 2, 1, 0), (2, 2, 0, 3),
                               (2, 1, 0, 1), (2, 1, 2, 4), (2, 2, 2, 5), (2, 2, 1, 2),
                               (2, 3, 0, 0), (2, 3, 1, 3), (2, 3, 2, 1), (2, 4, 0, 4),
                               (2, 4, 2, 5), (2, 4, 1, 2), (2, 1, 1, 5), (3, 0, 2, 2),
                               (3, 1, 0, 1), (3, 1, 2, 4), (3, 2, 0, 1), (3, 2, 2, 4),
                               (3, 3, 0, 0), (3, 3, 1, 3), (3, 3, 2, 1), (3, 4, 0, 4),
                               (4, 0, 2, 2), (3, 1, 1, 5), (3, 2, 1, 5), (4, 1, 2, 2),
                               (4, 1, 0, 0), (4, 1, 1, 3), (4, 2, 0, 0), (4, 2, 1, 3),
                               (4, 2, 2, 1), (4, 3, 0, 4),
                               ]),
        'random1': np.array([(0, 3, 1, 0), (1, 3, 0, 3), (0, 4, 1, 5), (1, 3, 2, 2),
                             (1, 2, 0, 1), (1, 2, 2, 4), (1, 4, 0, 1), (1, 4, 2, 4),
                             (1, 2, 1, 0), (2, 2, 0, 3), (1, 3, 1, 0), (2, 3, 0, 3),
                             (1, 4, 1, 5), (2, 3, 2, 2), (2, 1, 0, 1), (2, 1, 2, 4),
                             (2, 2, 1, 2), (2, 2, 2, 5), (2, 4, 0, 1), (2, 4, 2, 4),
                             (2, 1, 1, 5), (3, 0, 2, 2), (2, 3, 1, 5), (3, 2, 2, 2),
                             (2, 4, 1, 0), (3, 4, 0, 3), (3, 1, 2, 1), (3, 2, 0, 4),
                             (3, 3, 0, 1), (3, 3, 2, 4), (3, 2, 1, 0), (4, 2, 0, 3),
                             (3, 1, 0, 0), (3, 1, 1, 3), (3, 3, 1, 0), (4, 3, 0, 3),
                             (4, 0, 2, 1), (4, 1, 0, 4), (4, 1, 2, 5), (4, 1, 1, 2),
                             (4, 2, 2, 5), (4, 2, 1, 2)]),
        'random1_12x4': np.array([(11, 3, 1, 2),
                        (11, 3, 2, 5),
                        (10, 3, 1, 0),
                        (11, 3, 0, 3),
                        (11, 2, 1, 2),
                        (11, 2, 2, 5),
                        (11, 1, 0, 0),
                        (11, 1, 1, 3),
                        (11, 0, 1, 2),
                        (11, 0, 2, 5),
                        (10, 0, 1, 0),
                        (11, 0, 0, 3),
                        (10, 1, 0, 0),
                        (10, 1, 1, 3),
                        (11, 1, 2, 1),
                        (11, 2, 0, 4),
                        (10, 2, 1, 2),
                        (10, 2, 2, 5),
                        (9, 3, 1, 0),
                        (10, 3, 0, 3),
                        (9, 1, 1, 5),
                        (10, 0, 2, 2),
                        (10, 3, 2, 1),
                        (10, 0, 0, 4),
                        (9, 0, 0, 0),
                        (9, 0, 1, 3),
                        (10, 1, 2, 1),
                        (10, 2, 0, 4),
                        (9, 3, 0, 1),
                        (9, 3, 2, 4),
                        (8, 1, 1, 5),
                        (9, 0, 2, 2),
                        (9, 1, 0, 1),
                        (9, 1, 2, 4),
                        (8, 0, 1, 2),
                        (8, 0, 2, 5),
                        (8, 1, 0, 1),
                        (8, 1, 2, 4),
                        (7, 1, 1, 2),
                        (7, 1, 2, 5),
                        (9, 2, 1, 2),
                        (9, 2, 2, 5),
                        (8, 2, 1, 0),
                        (9, 2, 0, 3),
                        (8, 3, 2, 1),
                        (8, 0, 0, 4),
                        (8, 3, 0, 0),
                        (8, 3, 1, 3),
                        (7, 2, 1, 0),
                        (8, 2, 0, 3),
                        (7, 3, 1, 5),
                        (8, 2, 2, 2),
                        (7, 0, 2, 1),
                        (7, 1, 0, 4),
                        (7, 0, 0, 0),
                        (7, 0, 1, 3),
                        (6, 1, 1, 2),
                        (6, 1, 2, 5),
                        (6, 0, 2, 1),
                        (6, 1, 0, 4),
                        (5, 1, 1, 2),
                        (5, 1, 2, 5),
                        (5, 0, 2, 1),
                        (5, 1, 0, 4),
                        (6, 0, 1, 5),
                        (7, 3, 2, 2),
                        (7, 2, 0, 1),
                        (7, 2, 2, 4),
                        (6, 3, 1, 0),
                        (7, 3, 0, 3),
                        (6, 2, 0, 0),
                        (6, 2, 1, 3),
                        (4, 1, 1, 2),
                        (4, 1, 2, 5),
                        (5, 0, 1, 0),
                        (6, 0, 0, 3),
                        (6, 3, 0, 1),
                        (6, 3, 2, 4),
                        (5, 3, 1, 5),
                        (6, 2, 2, 2),
                        (4, 0, 2, 1),
                        (4, 1, 0, 4),
                        (5, 2, 1, 2),
                        (5, 2, 2, 5),
                        (4, 2, 1, 0),
                        (5, 2, 0, 3),
                        (3, 1, 0, 0),
                        (3, 1, 1, 3),
                        (5, 3, 0, 1),
                        (5, 3, 2, 4),
                        (4, 0, 1, 0),
                        (5, 0, 0, 3),
                        (4, 3, 0, 0),
                        (4, 3, 1, 3),
                        (4, 3, 2, 1),
                        (4, 0, 0, 4),
                        (4, 2, 0, 1),
                        (4, 2, 2, 4),
                        (3, 2, 0, 0),
                        (3, 2, 1, 3),
                        (2, 2, 1, 5),
                        (3, 1, 2, 2),
                        (3, 3, 1, 2),
                        (3, 3, 2, 5),
                        (3, 2, 2, 1),
                        (3, 3, 0, 4),
                        (2, 3, 0, 0),
                        (2, 3, 1, 3),
                        (3, 0, 0, 0),
                        (3, 0, 1, 3),
                        (2, 1, 1, 5),
                        (3, 0, 2, 2),
                        (2, 0, 1, 2),
                        (2, 0, 2, 5),
                        (2, 3, 2, 1),
                        (2, 0, 0, 4),
                        (1, 1, 1, 0),
                        (2, 1, 0, 3),
                        (2, 1, 2, 1),
                        (2, 2, 0, 4),
                        (1, 3, 1, 5),
                        (2, 2, 2, 2),
                        (1, 0, 1, 2),
                        (1, 0, 2, 5),
                        (0, 0, 1, 0),
                        (1, 0, 0, 3),
                        (1, 3, 0, 1),
                        (1, 3, 2, 4),
                        (1, 2, 0, 0),
                        (1, 2, 1, 3),
                        (0, 3, 1, 5),
                        (1, 2, 2, 2),
                        (0, 3, 2, 1),
                        (0, 0, 0, 4),
                        (0, 2, 2, 1),
                        (0, 3, 0, 4),
                        (0, 0, 2, 1),
                        (0, 1, 0, 4),
                        (0, 1, 1, 0),
                        (1, 1, 0, 3),
                        (0, 2, 1, 5),
                        (1, 1, 2, 2),
                        (0, 1, 2, 1),
                        (0, 2, 0, 4)])
    }  # yapf: disable
    (Lx, Ly) = lat.Ls
    if Lx%2==0 and Ly%2==0:
        glidercyl_0 = []
        glidercyl_1 = []
        for i in range(int(Lx/2)):
            for j in range(int(Ly/2)):
                glidercyl_0 = glidercyl_0 + [(2*i+1, 2*j, 1, 5), (2*i, 2*j+1, 2, 2), (2*i+1, 2*j+1, 1, 2), (2*i+1, 2*j+1, 2, 5)]
                glidercyl_1 = glidercyl_1 + [(2*i+1, 2*j, 1, 5), (2*i, 2*j+1, 2, 2), (2*i+1, 2*j+1, 1, 2), (2*i+1, 2*j+1, 2, 5)]
                glidercyl_1 = glidercyl_1 + [(2*i, 2*j, 0, 0), (2*i, 2*j, 1, 3), (2*i, 2*j, 2, 1), (2*i, 2*j+1, 0, 4), (2*i, 2*j+1, 1, 0), (2*i+1, 2*j, 0, 1), (2*i+1, 2*j, 2, 4), (2*i+1, 2*j+1, 0, 3)]
        squeeze_types['glidercyl_0'] = np.array(glidercyl_0)
        squeeze_types['glidercyl_1'] = np.array(glidercyl_1)
    squeeze_glider_42 = squeeze_types['glider_42']
    squeeze_glider_42_ref = squeeze_glider_42.copy()
    squeeze_glider_42_ref[:, 0] = 7 - squeeze_glider_42[:, 0] - squeeze_glider_42[:, 1] # mirror x -> 7-x - y
    squeeze_glider_42_ref[:, 2][squeeze_glider_42[:, 2] == 1] = 0  # switch u=0 and u=1
    squeeze_glider_42_ref[:, 2][squeeze_glider_42[:, 2] == 0] = 1
    squeeze_glider_42_ref[:, 3] = np.mod(3 - squeeze_glider_42[:, 3], 6)
    squeeze_types['glider_42_ref'] = squeeze_glider_42_ref

    # assert model.lat.N_sites == 42 , "squeezing only for 42-site clusters"
    s_type = squeeze_types[type_]

    displacements = 0.5*np.array([lat.basis[0],
                                lat.basis[1],
                                lat.basis[1] - lat.basis[0],
                                -lat.basis[0],
                                -lat.basis[1],
                                -(lat.basis[1] -lat.basis[0])])

    squeeze = np.zeros(lat.shape + (displacements.shape[1], ))
    squeeze[s_type[:, 0], s_type[:, 1], s_type[:, 2], :] = displacements[s_type[:, 3]]
    if lat.position_disorder is None:
        lat.position_disorder = squeeze * squeeze_strength
    else:
        lat.position_disorder +=  squeeze * squeeze_strength
    # done

# Simulation functions

def trunc_err(results, psi, simulation, key='trunc_err_eps'):
    results[key] = simulation.engine.trunc_err.eps

# Time evolution
def run_simulation_time(**simulation_params):
    return run_simulation(simulation_class_name='RealTimeEvolution', simulation_class_kwargs=None, **simulation_params)

# Orthogonal Excitations
def run_simulation_ortho(**simulation_params):
    return run_simulation(simulation_class_name='OrthogonalExcitations', simulation_class_kwargs=None, **simulation_params)


def run_OrthogonalExcitations(**simulation_params):
    # Pass the most_params  = simulation_params dict to OrthogonalExcitations class
    SimClass = tenpy.tools.misc.find_subclass(tenpy.simulations.simulation.Simulation,'OrthogonalExcitations')
    if SimClass is None:
        raise ValueError("can't find simulation class called " + repr( 'OrthogonalExcitations'))

    sim = SimClass(simulation_params)
    results = sim.run()

    return results
