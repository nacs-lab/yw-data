"""Complex lattices for tweezer array experiments, with long-range couplings
"""
import tenpy
from tenpy.models.lattice import Kagome, Square, Triangular
from tenpy.models.lattice import IrregularLattice
import numpy as np

from tenpy.models.lattice import Lattice
from tenpy.models.lattice import _parse_sites


class Durer(Lattice):
    """A lattice with pentagonal motifs, a la Albrecht Durer
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 5

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 5)  
        #     4   
        #    /  \
        #  2      3
        #   \    /
        #   0---1
 
        pos = np.array( [[0, 0],  
                        [1, 0],
                        [(1-np.sqrt(5))/4, np.sqrt(5+np.sqrt(5))/np.sqrt(8)],
                        [(3+np.sqrt(5))/4, np.sqrt(5+np.sqrt(5))/np.sqrt(8)],
                        [0.5, np.sqrt(5+2*np.sqrt(5))/2]])

        #basis = [[(2+np.sqrt(5))/2, np.sqrt(5+2*np.sqrt(5))/2], [0, np.sqrt(5+2*np.sqrt(5))]]
        basis = [[(2+np.sqrt(5))/2, np.sqrt(5+2*np.sqrt(5))/2], [-(2+np.sqrt(5))/2, np.sqrt(5+2*np.sqrt(5))/2]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 3, np.array([0,0])), (2, 4, np.array([0,0])), (2, 1, np.array([0,1])), (3, 4, np.array([0,0])), (3, 0, np.array([1,0]))],
            # d = 1.618034 ; d^3 = 4.24 
            n1_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (1, 2, np.array([0,0])), (1, 4, np.array([0,0])), (2, 3, np.array([0,0])), (2, 3, np.array([0,1])), (3, 2, np.array([1,0])), (4, 1, np.array([0,1])), (4, 3, np.array([0,1])), (4, 0, np.array([1,0])), (4, 2, np.array([1,0])), (4, 0, np.array([1,1])), (4, 1, np.array([1,1]))],
            # d = 1.902113 ; d^3 = 6.88 
            n2_nearest_neighbors = [(0, 1, np.array([0,1])), (1, 0, np.array([1,0])), (2, 0, np.array([0,1])), (3, 1, np.array([1,0]))],
            # d = 2.148961 ; d^3 = 9.92 
            n3_nearest_neighbors = [(2, 0, np.array([1,1])), (3, 1, np.array([1,1]))],
            # d = 2.497212 ; d^3 = 15.57 
            n4_nearest_neighbors = [(2, 4, np.array([0,1])), (2, 0, np.array([1,0])), (2, 1, np.array([1,1])), (3, 1, np.array([0,1])), (3, 4, np.array([1,0])), (3, 0, np.array([1,1]))],
            # d = 2.618034 ; d^3 = 17.94 
            n5_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 3, np.array([0,1])), (0, 0, np.array([1,0])), (1, 1, np.array([0,1])), (1, 1, np.array([1,0])), (1, 2, np.array([1,0])), (2, 2, np.array([0,1])), (2, 2, np.array([1,0])), (3, 3, np.array([0,1])), (3, 2, np.array([1,-1])), (3, 3, np.array([1,0])), (4, 0, np.array([0,1])), (4, 4, np.array([0,1])), (4, 1, np.array([1,0])), (4, 4, np.array([1,0])), (4, 2, np.array([1,1])), (4, 3, np.array([1,1]))],
            # d = 3.077684 ; d^3 = 29.15 
            n6_nearest_neighbors = [(0, 2, np.array([1,0])), (0, 0, np.array([1,1])), (1, 3, np.array([0,1])), (1, 2, np.array([1,-1])), (1, 1, np.array([1,1])), (2, 2, np.array([1,1])), (3, 0, np.array([1,-1])), (3, 3, np.array([1,1])), (4, 2, np.array([0,1])), (4, 3, np.array([1,0])), (4, 4, np.array([1,1]))],
            # d = 3.236068 ; d^3 = 33.89 
            n7_nearest_neighbors = [(0, 1, np.array([1,1])), (1, 0, np.array([1,-1])), (1, 0, np.array([1,1]))],
            # d = 3.477092 ; d^3 = 42.04 
            n8_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 4, np.array([0,1])), (0, 1, np.array([1,0])), (1, 0, np.array([0,1])), (1, 3, np.array([1,0])), (1, 4, np.array([1,0])), (2, 1, np.array([1,0])), (2, 3, np.array([1,1])), (3, 0, np.array([0,1])), (3, 4, np.array([1,-1])), (3, 2, np.array([1,1])), (4, 2, np.array([1,-1])), (4, 1, np.array([1,2])), (4, 0, np.array([2,1]))],
            # d = 3.618034 ; d^3 = 47.36 
            n9_nearest_neighbors = [(2, 1, np.array([0,2])), (2, 4, np.array([1,0])), (3, 4, np.array([0,1])), (3, 0, np.array([2,0]))],
            # d = 3.753688 ; d^3 = 52.89 
            n10_nearest_neighbors = [(2, 4, np.array([1,1])), (2, 1, np.array([1,2])), (3, 4, np.array([1,1])), (3, 0, np.array([2,1]))],
            # d = 4.040574 ; d^3 = 65.97 
            n11_nearest_neighbors = [(0, 2, np.array([1,-1])), (0, 4, np.array([1,0])), (0, 2, np.array([1,1])), (1, 4, np.array([0,1])), (1, 4, np.array([1,-1])), (1, 3, np.array([1,1])), (2, 3, np.array([0,2])), (2, 3, np.array([1,0])), (3, 2, np.array([0,1])), (3, 1, np.array([1,-1])), (3, 2, np.array([2,0])), (4, 1, np.array([0,2])), (4, 0, np.array([1,-1])), (4, 0, np.array([1,2])), (4, 0, np.array([2,0])), (4, 1, np.array([2,1]))],
            # d = 4.087567 ; d^3 = 68.30 
            n12_nearest_neighbors = [(2, 0, np.array([1,2])), (3, 1, np.array([2,1]))],
            # d = 4.236068 ; d^3 = 76.01 
            n13_nearest_neighbors = [(0, 0, np.array([1,-1])), (0, 3, np.array([1,0])), (0, 3, np.array([1,1])), (1, 2, np.array([0,1])), (1, 1, np.array([1,-1])), (1, 2, np.array([1,1])), (2, 2, np.array([1,-1])), (3, 3, np.array([1,-1])), (4, 3, np.array([0,2])), (4, 4, np.array([1,-1])), (4, 3, np.array([1,2])), (4, 2, np.array([2,0])), (4, 2, np.array([2,1]))],
            # d = 4.396162 ; d^3 = 84.96 
            n14_nearest_neighbors = [(2, 0, np.array([2,1])), (3, 1, np.array([1,2]))],
            # d = 4.465901 ; d^3 = 89.07 
            n15_nearest_neighbors = [(0, 1, np.array([0,2])), (1, 0, np.array([2,0])), (2, 0, np.array([0,2])), (3, 1, np.array([2,0]))],
            # d = 4.643523 ; d^3 = 100.13 
            n16_nearest_neighbors = [(0, 4, np.array([1,1])), (1, 3, np.array([1,-1])), (1, 4, np.array([1,1])), (2, 0, np.array([1,-1])), (2, 3, np.array([1,2])), (3, 2, np.array([2,1])), (4, 0, np.array([2,2])), (4, 1, np.array([2,2]))],
            # d = 4.749980 ; d^3 = 107.17 
            n17_nearest_neighbors = [(0, 1, np.array([1,2])), (1, 0, np.array([2,1]))],
            # d = 4.979797 ; d^3 = 123.49 
            n18_nearest_neighbors = [(0, 3, np.array([0,2])), (0, 4, np.array([1,-1])), (1, 2, np.array([2,0])), (3, 2, np.array([1,-2])), (3, 2, np.array([2,-1])), (4, 0, np.array([0,2])), (4, 1, np.array([1,-1])), (4, 2, np.array([1,2])), (4, 1, np.array([2,0])), (4, 3, np.array([2,1]))],
            # d = 5.018002 ; d^3 = 126.35 
            n19_nearest_neighbors = [(2, 4, np.array([0,2])), (2, 0, np.array([2,0])), (2, 1, np.array([2,1])), (3, 1, np.array([0,2])), (3, 0, np.array([1,2])), (3, 4, np.array([2,0]))],
            # d = 5.079210 ; d^3 = 131.04 
            n20_nearest_neighbors = [(0, 0, np.array([1,2])), (0, 0, np.array([2,1])), (1, 2, np.array([1,-2])), (1, 1, np.array([1,2])), (1, 1, np.array([2,1])), (2, 4, np.array([1,-1])), (2, 2, np.array([1,2])), (2, 2, np.array([2,1])), (3, 3, np.array([1,2])), (3, 0, np.array([2,-1])), (3, 3, np.array([2,1])), (4, 3, np.array([1,-1])), (4, 4, np.array([1,2])), (4, 4, np.array([2,1]))],
            # d = 5.213477 ; d^3 = 141.70 
            n21_nearest_neighbors = [(2, 0, np.array([2,2])), (3, 1, np.array([2,2]))],
            # d = 5.236068 ; d^3 = 143.55 
            n22_nearest_neighbors = [(0, 0, np.array([0,2])), (0, 1, np.array([1,-1])), (0, 0, np.array([2,0])), (1, 1, np.array([0,2])), (1, 1, np.array([2,0])), (2, 2, np.array([0,2])), (2, 2, np.array([2,0])), (3, 3, np.array([0,2])), (3, 3, np.array([2,0])), (4, 4, np.array([0,2])), (4, 4, np.array([2,0]))],
            # d = 5.366412 ; d^3 = 154.54 
            n23_nearest_neighbors = [(2, 4, np.array([1,2])), (2, 1, np.array([2,2])), (3, 4, np.array([2,1])), (3, 0, np.array([2,2]))],
            # d = 5.570856 ; d^3 = 172.89 
            n24_nearest_neighbors = [(0, 1, np.array([2,1])), (1, 0, np.array([1,-2])), (1, 0, np.array([1,2])), (1, 0, np.array([2,-1]))],
            # d = 5.626053 ; d^3 = 178.08 
            n25_nearest_neighbors = [(0, 3, np.array([1,-1])), (0, 3, np.array([1,2])), (0, 2, np.array([2,0])), (1, 3, np.array([0,2])), (1, 2, np.array([2,-1])), (1, 2, np.array([2,1])), (2, 1, np.array([1,-1])), (3, 0, np.array([1,-2])), (3, 4, np.array([1,-2])), (4, 2, np.array([0,2])), (4, 2, np.array([2,-1])), (4, 3, np.array([2,0])), (4, 2, np.array([2,2])), (4, 3, np.array([2,2]))],
            # d = 5.854102 ; d^3 = 200.62 
            n26_nearest_neighbors = [(0, 2, np.array([2,1])), (1, 4, np.array([1,-2])), (1, 3, np.array([1,2])), (2, 3, np.array([1,-1])), (4, 0, np.array([2,-1]))],
            # d = 5.938898 ; d^3 = 209.47 
            n27_nearest_neighbors = [(0, 4, np.array([0,2])), (1, 4, np.array([2,0])), (2, 1, np.array([2,0])), (2, 3, np.array([2,1])), (3, 0, np.array([0,2])), (3, 2, np.array([1,2])), (3, 4, np.array([2,-1])), (4, 2, np.array([1,-2])), (4, 1, np.array([1,3])), (4, 0, np.array([3,1]))],
            # d = 5.970969 ; d^3 = 212.88 
            n28_nearest_neighbors = [(2, 1, np.array([1,3])), (2, 4, np.array([2,1])), (3, 4, np.array([1,2])), (3, 0, np.array([3,1]))],
            # d = 6.073594 ; d^3 = 224.05 
            n29_nearest_neighbors = [(0, 2, np.array([0,2])), (0, 2, np.array([1,-2])), (0, 2, np.array([1,2])), (0, 1, np.array([2,0])), (1, 0, np.array([0,2])), (1, 3, np.array([2,0])), (1, 3, np.array([2,1])), (3, 1, np.array([2,-1]))],
            # d = 6.155367 ; d^3 = 233.22 
            n30_nearest_neighbors = [(0, 0, np.array([2,2])), (1, 1, np.array([2,2])), (2, 2, np.array([2,2])), (3, 3, np.array([2,2])), (4, 4, np.array([2,2]))],
            # d = 6.236068 ; d^3 = 242.51 
            n31_nearest_neighbors = [(0, 1, np.array([2,2])), (1, 0, np.array([2,2])), (2, 1, np.array([0,3])), (2, 4, np.array([2,0])), (3, 4, np.array([0,2])), (3, 0, np.array([3,0]))],
            # d = 6.364478 ; d^3 = 257.80 
            n32_nearest_neighbors = [(0, 4, np.array([1,2])), (1, 4, np.array([2,1])), (2, 3, np.array([2,2])), (3, 2, np.array([2,2])), (4, 1, np.array([2,3])), (4, 0, np.array([3,2]))],
            # d = 6.519707 ; d^3 = 277.13 
            n33_nearest_neighbors = [(2, 0, np.array([1,3])), (3, 1, np.array([3,1]))]
       )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class TruncatedSquare(Lattice):
    """The vertices of the truncated square (Mediterranean / octagonal) tiling
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 4

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 4) 
        #     | 
        #     3
        #   /   \
        #--1  x  2--
        #   \   /
        #     0
        #     |

        pos = np.array([[0,-1/(2**0.5)], [-1/(2**0.5), 0],
                        [1/(2**0.5), 0], [0,  1/(2**0.5)]])

        basis = [[1 + 2**0.5, 0], [0, 1 + 2**0.5]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 3, np.array([0,0])), (2, 3, np.array([0,0])), (2, 1, np.array([1,0])), (3, 0, np.array([0,1]))],
            # d = 1.414214 ; d^3 = 2.83 
            n1_nearest_neighbors = [(0, 3, np.array([0,0])), (1, 2, np.array([0,0]))],
            # d = 1.847759 ; d^3 = 6.31 
            n2_nearest_neighbors = [(0, 1, np.array([1,0])), (1, 0, np.array([0,1])), (2, 0, np.array([0,1])), (2, 0, np.array([1,0])), (2, 3, np.array([1,0])), (3, 1, np.array([0,1])), (3, 2, np.array([0,1])), (3, 1, np.array([1,0]))],
            # d = 2.414214 ; d^3 = 14.07 
            n3_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 1, np.array([1,-1])), (0, 0, np.array([1,0])), (1, 1, np.array([0,1])), (1, 1, np.array([1,0])), (2, 2, np.array([0,1])), (2, 3, np.array([1,-1])), (2, 2, np.array([1,0])), (2, 0, np.array([1,1])), (3, 3, np.array([0,1])), (3, 3, np.array([1,0])), (3, 1, np.array([1,1]))],
            # d = 2.613126 ; d^3 = 17.84 
            n4_nearest_neighbors = [(0, 3, np.array([1,-1])), (2, 1, np.array([1,-1])), (2, 1, np.array([1,1])), (3, 0, np.array([1,1]))],
            # d = 2.797933 ; d^3 = 21.90 
            n5_nearest_neighbors = [(0, 3, np.array([1,0])), (1, 2, np.array([0,1])), (2, 1, np.array([0,1])), (3, 0, np.array([1,0]))],
            # d = 3.200413 ; d^3 = 32.78 
            n6_nearest_neighbors = [(0, 1, np.array([0,1])), (0, 2, np.array([0,1])), (0, 2, np.array([1,0])), (1, 3, np.array([0,1])), (1, 0, np.array([1,0])), (1, 3, np.array([1,0])), (2, 3, np.array([0,1])), (3, 2, np.array([1,0]))],
            # d = 3.414214 ; d^3 = 39.80 
            n7_nearest_neighbors = [(0, 0, np.array([1,-1])), (0, 0, np.array([1,1])), (1, 1, np.array([1,-1])), (1, 1, np.array([1,1])), (2, 2, np.array([1,-1])), (2, 2, np.array([1,1])), (2, 1, np.array([2,0])), (3, 0, np.array([0,2])), (3, 3, np.array([1,-1])), (3, 3, np.array([1,1]))],
            # d = 3.557647 ; d^3 = 45.03 
            n8_nearest_neighbors = [(0, 2, np.array([1,-1])), (0, 1, np.array([1,1])), (1, 3, np.array([1,-1])), (1, 0, np.array([1,1])), (2, 0, np.array([1,-1])), (2, 3, np.array([1,1])), (3, 1, np.array([1,-1])), (3, 2, np.array([1,1]))],
            # d = 3.828427 ; d^3 = 56.11 
            n9_nearest_neighbors = [(0, 3, np.array([0,1])), (1, 2, np.array([1,0]))],
            # d = 4.181541 ; d^3 = 73.12 
            n10_nearest_neighbors = [(0, 3, np.array([1,-2])), (0, 1, np.array([2,0])), (1, 0, np.array([0,2])), (2, 0, np.array([0,2])), (2, 1, np.array([2,-1])), (2, 0, np.array([2,0])), (2, 3, np.array([2,0])), (2, 1, np.array([2,1])), (3, 1, np.array([0,2])), (3, 2, np.array([0,2])), (3, 0, np.array([1,2])), (3, 1, np.array([2,0]))],
            # d = 4.414214 ; d^3 = 86.01 
            n11_nearest_neighbors = [(0, 2, np.array([1,1])), (1, 0, np.array([1,-1])), (1, 3, np.array([1,1])), (3, 2, np.array([1,-1]))],
            # d = 4.460885 ; d^3 = 88.77 
            n12_nearest_neighbors = [(0, 1, np.array([1,-2])), (0, 1, np.array([2,-1])), (2, 3, np.array([1,-2])), (2, 0, np.array([1,2])), (2, 3, np.array([2,-1])), (2, 0, np.array([2,1])), (3, 1, np.array([1,2])), (3, 1, np.array([2,1]))],
            # d = 4.526067 ; d^3 = 92.72 
            n13_nearest_neighbors = [(0, 3, np.array([1,1])), (1, 2, np.array([1,-1])), (1, 2, np.array([1,1])), (3, 0, np.array([1,-1]))]
        )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class SnubSquare(Lattice):
    """The vertices of the snub square tiling
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 4

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 4) 
        #         3 
        #       // \
        #     2     \
        #      \     1
        #       \  //
        #        0

        pos = np.array([[0, 0], [(1+3**0.5)/(8**0.5), (-1+3**0.5)/(8**0.5)],
                        [(1-3**0.5)/(8**0.5), (1+3**0.5)/(8**0.5)], [0.5**0.5, 1.5**0.5]])

        basis = [[(1+3**0.5)/(2**0.5), 0], [0, (1+3**0.5)/(2**0.5)]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
                # d = 1.000000 ; d^3 = 1.00 
                nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 3, np.array([0,0])), (1, 0, np.array([1,0])), (1, 2, np.array([1,0])), (2, 3, np.array([0,0])), (2, 0, np.array([0,1])), (3, 0, np.array([0,1])), (3, 1, np.array([0,1])), (3, 2, np.array([1,0]))],
                # d = 1.414214 ; d^3 = 2.83 
                n1_nearest_neighbors = [(0, 3, np.array([0,0])), (1, 2, np.array([0,0])), (1, 2, np.array([1,-1])), (3, 0, np.array([1,1]))],
                # d = 1.732051 ; d^3 = 5.20 
                n2_nearest_neighbors = [(2, 1, np.array([0,1])), (3, 0, np.array([1,0]))],
                # d = 1.931852 ; d^3 = 7.21 
                n3_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 2, np.array([1,-1])), (0, 0, np.array([1,0])), (0, 2, np.array([1,0])), (1, 0, np.array([0,1])), (1, 1, np.array([0,1])), (1, 3, np.array([1,-1])), (1, 1, np.array([1,0])), (1, 3, np.array([1,0])), (1, 0, np.array([1,1])), (2, 2, np.array([0,1])), (2, 2, np.array([1,0])), (3, 2, np.array([0,1])), (3, 3, np.array([0,1])), (3, 3, np.array([1,0])), (3, 2, np.array([1,1]))],
                # d = 2.394170 ; d^3 = 13.72 
                n4_nearest_neighbors = [(0, 1, np.array([0,1])), (1, 0, np.array([1,-1])), (2, 3, np.array([0,1])), (2, 0, np.array([1,0])), (2, 0, np.array([1,1])), (3, 2, np.array([1,-1])), (3, 1, np.array([1,0])), (3, 1, np.array([1,1]))],
                # d = 2.732051 ; d^3 = 20.39 
                n5_nearest_neighbors = [(0, 0, np.array([1,-1])), (0, 3, np.array([1,-1])), (0, 0, np.array([1,1])), (1, 1, np.array([1,-1])), (1, 1, np.array([1,1])), (1, 2, np.array([1,1])), (1, 2, np.array([2,0])), (2, 2, np.array([1,-1])), (2, 2, np.array([1,1])), (3, 0, np.array([0,2])), (3, 3, np.array([1,-1])), (3, 3, np.array([1,1]))],
                # d = 2.909313 ; d^3 = 24.62 
                n6_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 1, np.array([1,0])), (0, 3, np.array([1,0])), (1, 2, np.array([0,1])), (1, 3, np.array([0,1])), (1, 2, np.array([2,-1])), (1, 0, np.array([2,0])), (2, 0, np.array([0,2])), (2, 3, np.array([1,0])), (3, 1, np.array([0,2])), (3, 0, np.array([1,2])), (3, 2, np.array([2,0]))],
                # d = 3.234826 ; d^3 = 33.85 
                n7_nearest_neighbors = [(0, 3, np.array([0,1])), (1, 2, np.array([1,-2])), (2, 1, np.array([1,0])), (3, 0, np.array([2,1]))],
                # d = 3.346065 ; d^3 = 37.46 
                n8_nearest_neighbors = [(0, 2, np.array([1,-2])), (0, 1, np.array([1,-1])), (0, 2, np.array([1,1])), (1, 3, np.array([1,-2])), (1, 3, np.array([1,1])), (1, 0, np.array([2,1])), (2, 3, np.array([1,-1])), (3, 2, np.array([2,1]))],
                # d = 3.385868 ; d^3 = 38.82 
                n9_nearest_neighbors = [(2, 1, np.array([0,2])), (2, 1, np.array([1,1])), (3, 0, np.array([1,-1])), (3, 0, np.array([2,0]))],
                # d = 3.632651 ; d^3 = 47.94 
                n10_nearest_neighbors = [(0, 1, np.array([1,1])), (1, 0, np.array([2,-1])), (2, 0, np.array([1,-1])), (2, 3, np.array([1,1])), (2, 0, np.array([1,2])), (3, 1, np.array([1,-1])), (3, 1, np.array([1,2])), (3, 2, np.array([2,-1]))],
                # d = 3.732051 ; d^3 = 51.98 
                n11_nearest_neighbors = [(0, 3, np.array([1,-2])), (0, 2, np.array([2,-1])), (0, 2, np.array([2,0])), (1, 0, np.array([0,2])), (1, 0, np.array([1,2])), (1, 3, np.array([2,-1])), (1, 3, np.array([2,0])), (1, 2, np.array([2,1])), (3, 2, np.array([0,2])), (3, 2, np.array([1,2]))],
                # d = 3.863703 ; d^3 = 57.68 
                n12_nearest_neighbors = [(0, 0, np.array([0,2])), (0, 0, np.array([2,0])), (1, 1, np.array([0,2])), (1, 1, np.array([2,0])), (2, 2, np.array([0,2])), (2, 2, np.array([2,0])), (3, 3, np.array([0,2])), (3, 3, np.array([2,0]))],
                # d = 4.114390 ; d^3 = 69.65 
                n13_nearest_neighbors = [(0, 3, np.array([1,1])), (1, 2, np.array([2,-2])), (2, 1, np.array([1,-1])), (3, 0, np.array([2,2]))]
        )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Cairo(Lattice):
    """A Cairo  lattice.
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 6)  
        #              5
        #    - -3--   /
        #    /    \  / 
        #   /      4
        #  2      /
        #   \    /
        #   0---1
 
        isq7 = 0.3779644730092272 # 1 / sqrt(7)
        pos = np.array([[0, 0], 
                        [0.96592583, 0.25881905], [-0.25881905, 0.96592583],
                        [0.25881905, (6**0.5)-0.96592583], [(6**0.5)/2,(6**0.5)/2],
                        [(6**0.5)/2+0.25881905, (6**0.5)/2+0.96592583]])

        basis = [[6**0.5, 0], [0, 6**0.5]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        # Natural edges, just for plotting, or when distances aren't important
        NN = [(0, 1, np.array([0, 0])), (0, 2, np.array([0, 0])), (2, 3, np.array([0, 0])),
              (3, 4, np.array([0, 0])), (4, 1, np.array([0, 0])), (4, 5, np.array([0, 0])),
              (3, 0, np.array([0, 1])), (5, 0, np.array([1, 1])),  (5, 1, np.array([0, 1])),
              (4, 2, np.array([1,0]))]
        # Short edges - d = 0.7320508 ; d^6 = 0.1539030821872433
        sNN = [(2, 3, np.array([0,0])), (5, 1, np.array([0,1]))]
        # Normal edges - d = 1 ; d^6 = 1
        n0NN = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 4, np.array([0,0])),
                (3, 4, np.array([0,0])), (3, 0, np.array([0,1])), (4, 5, np.array([0,0])), 
                (4, 2, np.array([1,0])), (5, 0, np.array([1,1]))]
        # d = sqrt(2) ; d^6 = 8
        n1NN = [(1, 2, np.array([0,0])), (1, 3, np.array([0,0])), (1, 2, np.array([1,0])), 
                (3, 5, np.array([0,0])), (3, 1, np.array([0,1])), (5, 2, np.array([1,0])), 
                (5, 3, np.array([1,0])), (5, 2, np.array([1,1]))]
        # d = 1.5059712 ; d^6 = 11.665409783014345
        n2NN =[(0, 3, np.array([0,0])), (1, 0, np.array([1,0])), (2, 4, np.array([0,0])), 
               (2, 0, np.array([0,1])), (4, 1, np.array([0,1])), (4, 3, np.array([1,0])), 
               (5, 0, np.array([0,1])), (5, 4, np.array([0,1]))]
        # d = sqrt(3) ; d^6 = 27
        n3NN = [(0, 4, np.array([0,0])), (4, 0, np.array([0,1])), (4, 0, np.array([1,0])),
                (4, 0, np.array([1,1]))]
        # d = 2 ; d^6 = 64
        n4NN = [(1, 5, np.array([0,0])), (3, 2, np.array([0,1])), (3, 2, np.array([1,0])), 
                (5, 1, np.array([1,1]))]
        # d = 2.1297649 ; d^6 = 93.32327942531995
        n5NN = [(1, 2, np.array([1,-1])), (1, 3, np.array([1,-1])), (1, 3, np.array([1,0])), 
                (2, 5, np.array([0,0])), (2, 1, np.array([0,1])), (5, 2, np.array([0,1])), 
                (5, 3, np.array([0,1])), (5, 3, np.array([1,1]))]
        # d = 2.3941702 ; d^6 = 188.33460488705205
        n6NN = [(0, 2, np.array([1,0])), (1, 0, np.array([0,1])), (3, 4, np.array([0,1])), 
                (3, 0, np.array([1,1])), (4, 1, np.array([1,0])), (4, 2, np.array([1,1])), 
                (5, 0, np.array([1,0])), (5, 4, np.array([1,0]))]
        # d = 2.45 ; d^6 = 216
        n7NN = [(0, 0, np.array([0,1])), (0, 0, np.array([1,0])), (1, 1, np.array([0,1])), 
                (1, 1, np.array([1,0])), (2, 2, np.array([0,1])), (2, 2, np.array([1,0])), 
                (3, 3, np.array([0,1])), (3, 3, np.array([1,0])), (4, 4, np.array([0,1])), 
                (4, 4, np.array([1,0])), (5, 5, np.array([0,1])), (5, 5, np.array([1,0]))]
        # d = 2.65 ; d^6 = 343
        n8NN = [(0, 5, np.array([0,0])), (0, 2, np.array([1,-1])), (1, 0, np.array([1,1])), 
                (3, 0, np.array([1,0])), (4, 2, np.array([0,1])), (4, 3, np.array([1,-1])), 
                (4, 1, np.array([1,1])), (5, 4, np.array([1,1]))]
        # d= (1+sqrt(3)) ; d^6 = 415.84609
        n9NN = [(3, 2, np.array([1,1])), (5, 1, np.array([1,0]))]
        # d= 2.8754042 ; d^6 = 565.1886301099738
        n10NN = [(0, 1, np.array([0,1])), (0, 3, np.array([1,-1])), (1, 4, np.array([1,0])),
                 (2, 0, np.array([1,0])), (4, 3, np.array([0,1])), (4, 2, np.array([1,-1])),
                 (4, 5, np.array([1,0])), (5, 0, np.array([1,2]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('short_nearest_neighbors', sNN)
        kwargs['pairs'].setdefault('n0_nearest_neighbors', n0NN)
        kwargs['pairs'].setdefault('n1_nearest_neighbors', n1NN)
        kwargs['pairs'].setdefault('n2_nearest_neighbors', n2NN)
        kwargs['pairs'].setdefault('n3_nearest_neighbors', n3NN)
        kwargs['pairs'].setdefault('n4_nearest_neighbors', n4NN)
        kwargs['pairs'].setdefault('n5_nearest_neighbors', n1NN)
        kwargs['pairs'].setdefault('n6_nearest_neighbors', n2NN)
        kwargs['pairs'].setdefault('n7_nearest_neighbors', n3NN)
        kwargs['pairs'].setdefault('n8_nearest_neighbors', n4NN)
        kwargs['pairs'].setdefault('n9_nearest_neighbors', n1NN)
        kwargs['pairs'].setdefault('n10_nearest_neighbors', n2NN)

        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class TruncatedHexagon(Lattice):
    """The vertices of the truncated hexagon (Fisher) lattice
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 6) 
        #        4
        #       |  \
        #       |   5
        #       | /
        #       3
        #      /
        #    2
        #   / |
        # 0   |
        #  \  |
        #    1

        cos30 = 0.5 * (3**0.5)
        pos = np.array([[0, 0], [cos30, -0.5], [cos30, 0.5],
                        [0.5 + cos30, 0.5+cos30], [0.5 + cos30, 1.5 + cos30], [0.5 + 2*cos30, 1 + cos30]])

        basis = [[1.5 + 2*cos30, 1+cos30], [0, 2+2*cos30]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 2, np.array([0,0])), (2, 3, np.array([0,0])), (3, 4, np.array([0,0])), (3, 5, np.array([0,0])), (4, 5, np.array([0,0])), (4, 1, np.array([0,1])), (5, 0, np.array([1,0]))],
            # d = 1.931852 ; d^3 = 7.21 
            n1_nearest_neighbors = [(0, 3, np.array([0,0])), (1, 3, np.array([0,0])), (2, 4, np.array([0,0])), (2, 5, np.array([0,0])), (3, 1, np.array([0,1])), (3, 0, np.array([1,0])), (4, 0, np.array([0,1])), (4, 2, np.array([0,1])), (4, 0, np.array([1,0])), (5, 1, np.array([0,1])), (5, 1, np.array([1,0])), (5, 2, np.array([1,0]))],
            # d = 2.732051 ; d^3 = 20.39 
            n2_nearest_neighbors = [(0, 4, np.array([0,0])), (1, 5, np.array([0,0])), (1, 0, np.array([1,-1])), (2, 1, np.array([0,1])), (2, 0, np.array([1,0])), (3, 0, np.array([0,1])), (3, 1, np.array([1,0])), (4, 3, np.array([0,1])), (4, 2, np.array([1,0])), (5, 2, np.array([0,1])), (5, 4, np.array([1,-1])), (5, 3, np.array([1,0]))],
            # d = 2.909313 ; d^3 = 24.62 
            n3_nearest_neighbors = [(0, 5, np.array([0,0])), (1, 4, np.array([0,0])), (3, 2, np.array([0,1])), (3, 2, np.array([1,0])), (4, 1, np.array([1,0])), (5, 0, np.array([0,1]))],
            # d = 3.346065 ; d^3 = 37.46 
            n4_nearest_neighbors = [(0, 1, np.array([0,1])), (1, 2, np.array([1,-1])), (1, 0, np.array([1,0])), (2, 0, np.array([0,1])), (2, 0, np.array([1,-1])), (2, 1, np.array([1,0])), (3, 4, np.array([1,-1])), (4, 5, np.array([0,1])), (4, 3, np.array([1,0])), (5, 3, np.array([0,1])), (5, 3, np.array([1,-1])), (5, 4, np.array([1,0]))],
            # d = 3.732051 ; d^3 = 51.98 
            n5_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 0, np.array([1,0])), (1, 1, np.array([0,1])), (1, 1, np.array([1,-1])), (1, 3, np.array([1,-1])), (1, 1, np.array([1,0])), (2, 2, np.array([0,1])), (2, 2, np.array([1,-1])), (2, 4, np.array([1,-1])), (2, 2, np.array([1,0])), (3, 3, np.array([0,1])), (3, 0, np.array([1,-1])), (3, 3, np.array([1,-1])), (3, 3, np.array([1,0])), (4, 4, np.array([0,1])), (4, 4, np.array([1,-1])), (4, 4, np.array([1,0])), (4, 0, np.array([1,1])), (5, 5, np.array([0,1])), (5, 2, np.array([1,-1])), (5, 5, np.array([1,-1])), (5, 5, np.array([1,0])), (5, 1, np.array([1,1]))],
            # d = 3.863703 ; d^3 = 57.68 
            n6_nearest_neighbors = [(1, 4, np.array([1,-1])), (2, 3, np.array([1,-1])), (3, 2, np.array([1,-1])), (4, 1, np.array([1,1])), (5, 0, np.array([1,-1])), (5, 0, np.array([1,1]))],
            # d = 4.319752 ; d^3 = 80.61 
            n7_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 2, np.array([1,-1])), (0, 1, np.array([1,0])), (1, 0, np.array([0,1])), (1, 2, np.array([1,0])), (2, 1, np.array([1,-1])), (3, 5, np.array([0,1])), (3, 5, np.array([1,-1])), (3, 4, np.array([1,0])), (4, 3, np.array([1,-1])), (4, 5, np.array([1,0])), (5, 4, np.array([0,1]))],
            # d = 4.625182 ; d^3 = 98.94 
            n8_nearest_neighbors = [(0, 3, np.array([1,-1])), (0, 4, np.array([1,-1])), (1, 4, np.array([1,-2])), (1, 5, np.array([1,-1])), (2, 3, np.array([0,1])), (2, 5, np.array([1,-1])), (2, 3, np.array([1,0])), (3, 1, np.array([1,-1])), (3, 0, np.array([1,1])), (3, 1, np.array([1,1])), (4, 1, np.array([0,2])), (4, 0, np.array([1,-1])), (4, 2, np.array([1,-1])), (4, 2, np.array([1,1])), (5, 1, np.array([1,-1])), (5, 2, np.array([1,1])), (5, 0, np.array([2,-1])), (5, 0, np.array([2,0]))],
            # d = 4.732051 ; d^3 = 105.96 
            n9_nearest_neighbors = [(0, 1, np.array([1,-1])), (0, 2, np.array([1,0])), (1, 2, np.array([0,1])), (3, 4, np.array([0,1])), (3, 5, np.array([1,0])), (4, 5, np.array([1,-1]))],
            # d = 5.277917 ; d^3 = 147.02 
            n10_nearest_neighbors = [(0, 3, np.array([0,1])), (1, 3, np.array([1,-2])), (1, 3, np.array([1,0])), (2, 5, np.array([0,1])), (2, 4, np.array([1,-2])), (2, 4, np.array([1,0])), (3, 0, np.array([2,-1])), (4, 0, np.array([0,2])), (4, 0, np.array([2,0])), (5, 1, np.array([0,2])), (5, 2, np.array([2,-1])), (5, 1, np.array([2,0]))],
            # d = 5.464102 ; d^3 = 163.14 
            n11_nearest_neighbors = [(0, 5, np.array([1,-1])), (3, 2, np.array([1,1])), (4, 1, np.array([1,-1]))],
            # d = 5.620361 ; d^3 = 177.54 
            n12_nearest_neighbors = [(0, 4, np.array([1,-2])), (0, 3, np.array([1,0])), (1, 3, np.array([0,1])), (1, 0, np.array([1,-2])), (1, 2, np.array([1,-2])), (1, 5, np.array([1,-2])), (1, 0, np.array([2,-1])), (2, 4, np.array([0,1])), (2, 5, np.array([1,0])), (2, 0, np.array([1,1])), (2, 1, np.array([1,1])), (2, 0, np.array([2,-1])), (3, 1, np.array([0,2])), (3, 4, np.array([1,-2])), (3, 0, np.array([2,0])), (4, 2, np.array([0,2])), (4, 3, np.array([1,1])), (4, 0, np.array([2,-1])), (5, 4, np.array([1,-2])), (5, 3, np.array([1,1])), (5, 1, np.array([2,-1])), (5, 3, np.array([2,-1])), (5, 4, np.array([2,-1])), (5, 2, np.array([2,0]))],
            # d = 6.026650 ; d^3 = 218.89 
            n13_nearest_neighbors = [(0, 5, np.array([0,1])), (1, 4, np.array([1,0])), (2, 3, np.array([1,-2])), (3, 2, np.array([2,-1])), (4, 1, np.array([2,0])), (5, 0, np.array([0,2]))],
            # d = 6.249205 ; d^3 = 244.05 
            n14_nearest_neighbors = [(0, 4, np.array([0,1])), (0, 3, np.array([1,-2])), (0, 4, np.array([1,0])), (1, 5, np.array([0,1])), (1, 5, np.array([1,0])), (2, 5, np.array([1,-2])), (3, 0, np.array([0,2])), (3, 1, np.array([2,-1])), (3, 1, np.array([2,0])), (4, 2, np.array([2,-1])), (4, 2, np.array([2,0])), (5, 2, np.array([0,2]))],
            # d = 6.464102 ; d^3 = 270.10 
            n15_nearest_neighbors = [(0, 0, np.array([1,-2])), (0, 0, np.array([1,1])), (0, 0, np.array([2,-1])), (1, 1, np.array([1,-2])), (1, 1, np.array([1,1])), (1, 0, np.array([2,-2])), (1, 1, np.array([2,-1])), (2, 1, np.array([0,2])), (2, 2, np.array([1,-2])), (2, 2, np.array([1,1])), (2, 2, np.array([2,-1])), (2, 0, np.array([2,0])), (3, 3, np.array([1,-2])), (3, 3, np.array([1,1])), (3, 3, np.array([2,-1])), (4, 3, np.array([0,2])), (4, 4, np.array([1,-2])), (4, 4, np.array([1,1])), (4, 4, np.array([2,-1])), (5, 5, np.array([1,-2])), (5, 5, np.array([1,1])), (5, 4, np.array([2,-2])), (5, 5, np.array([2,-1])), (5, 3, np.array([2,0]))],
            # d = 6.540995 ; d^3 = 279.85 
            n16_nearest_neighbors = [(0, 2, np.array([1,-2])), (0, 1, np.array([1,1])), (1, 0, np.array([1,1])), (1, 2, np.array([2,-1])), (2, 0, np.array([1,-2])), (2, 1, np.array([2,-1])), (3, 5, np.array([1,-2])), (3, 4, np.array([2,-1])), (4, 5, np.array([1,1])), (4, 3, np.array([2,-1])), (5, 3, np.array([1,-2])), (5, 4, np.array([1,1]))],
            # d = 6.616994 ; d^3 = 289.72 
            n17_nearest_neighbors = [(0, 5, np.array([1,-2])), (0, 5, np.array([1,0])), (1, 4, np.array([0,1])), (3, 2, np.array([0,2])), (3, 2, np.array([2,0])), (4, 1, np.array([2,-1]))]
        )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Tetrille(Lattice):
    """The vertices of the deltoidal trihexagonal tiling
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 6) 
        #    /    |
        #   4     |
        #  /  \\\ |
        # 3       5
        #  \      |
        #   2     |
        #    \    |
        # x   1---0

        cos30 = 0.5 * (3**0.5)
        pos = np.array([[2, 0], [1, 0], [0.5, cos30],
                        [0, 2*cos30], [0.5, 3*cos30], [2,  2*cos30]])

        basis = [[3, 2*cos30], [0, 4*cos30]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 3, np.array([1,-1])), (1, 2, np.array([0,0])), (2, 3, np.array([0,0])), (3, 4, np.array([0,0])), (4, 1, np.array([0,1]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 2, np.array([0,0])), (0, 5, np.array([0,0])), (0, 2, np.array([1,-1])), (0, 4, np.array([1,-1])), (2, 4, np.array([0,0])), (2, 5, np.array([0,0])), (4, 5, np.array([0,0])), (4, 0, np.array([0,1])), (4, 2, np.array([0,1])), (5, 0, np.array([0,1])), (5, 4, np.array([1,-1])), (5, 2, np.array([1,0]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(1, 3, np.array([0,0])), (1, 5, np.array([0,0])), (1, 3, np.array([1,-1])), (3, 5, np.array([0,0])), (3, 1, np.array([0,1])), (5, 1, np.array([0,1])), (5, 3, np.array([1,-1])), (5, 1, np.array([1,0])), (5, 3, np.array([1,0]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 1, np.array([1,-1])), (0, 1, np.array([1,0])), (1, 4, np.array([0,0])), (1, 2, np.array([1,-1])), (1, 4, np.array([1,-1])), (2, 1, np.array([0,1])), (2, 3, np.array([1,-1])), (3, 0, np.array([0,1])), (3, 2, np.array([0,1])), (4, 3, np.array([0,1])), (4, 3, np.array([1,0]))],
            # d = 3.000000 ; d^3 = 27.00 
            n4_nearest_neighbors = [(0, 4, np.array([0,0])), (0, 4, np.array([1,-2])), (0, 5, np.array([1,-1])), (0, 2, np.array([1,0])), (2, 0, np.array([0,1])), (2, 4, np.array([1,-1])), (4, 5, np.array([0,1])), (4, 2, np.array([1,0])), (5, 2, np.array([0,1])), (5, 2, np.array([1,-1])), (5, 0, np.array([1,0])), (5, 4, np.array([1,0]))],
            # d = 3.464102 ; d^3 = 41.57 
            n5_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 0, np.array([1,0])), (1, 1, np.array([0,1])), (1, 1, np.array([1,-1])), (1, 1, np.array([1,0])), (2, 2, np.array([0,1])), (2, 2, np.array([1,-1])), (2, 2, np.array([1,0])), (3, 3, np.array([0,1])), (3, 3, np.array([1,-1])), (3, 3, np.array([1,0])), (4, 4, np.array([0,1])), (4, 4, np.array([1,-1])), (4, 4, np.array([1,0])), (5, 5, np.array([0,1])), (5, 5, np.array([1,-1])), (5, 5, np.array([1,0]))],
            # d = 3.605551 ; d^3 = 46.87 
            n6_nearest_neighbors = [(0, 1, np.array([0,1])), (0, 3, np.array([1,-2])), (0, 3, np.array([1,0])), (1, 0, np.array([0,1])), (1, 4, np.array([1,-2])), (1, 2, np.array([1,0])), (2, 1, np.array([1,0])), (2, 3, np.array([1,0])), (3, 4, np.array([1,-1])), (3, 2, np.array([1,0])), (4, 3, np.array([1,-1])), (4, 1, np.array([1,0]))],
            # d = 4.000000 ; d^3 = 64.00 
            n7_nearest_neighbors = [(1, 3, np.array([1,-2])), (1, 5, np.array([1,-1])), (1, 3, np.array([1,0])), (3, 5, np.array([0,1])), (3, 1, np.array([1,0])), (5, 3, np.array([0,1])), (5, 1, np.array([1,-1])), (5, 1, np.array([1,1])), (5, 3, np.array([2,-1]))],
            # d = 4.358899 ; d^3 = 82.82 
            n8_nearest_neighbors = [(0, 3, np.array([2,-2])), (0, 3, np.array([2,-1])), (1, 2, np.array([0,1])), (1, 0, np.array([1,-1])), (1, 0, np.array([1,0])), (2, 3, np.array([0,1])), (2, 1, np.array([1,-1])), (3, 4, np.array([0,1])), (3, 2, np.array([1,-1])), (3, 4, np.array([1,0])), (4, 1, np.array([0,2])), (4, 1, np.array([1,1]))],
            # d = 4.582576 ; d^3 = 96.23 
            n9_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 2, np.array([1,-2])), (0, 5, np.array([1,-2])), (0, 4, np.array([1,0])), (0, 5, np.array([1,0])), (0, 4, np.array([2,-2])), (0, 2, np.array([2,-1])), (2, 5, np.array([0,1])), (2, 4, np.array([1,-2])), (2, 5, np.array([1,-1])), (2, 0, np.array([1,0])), (2, 4, np.array([1,0])), (4, 0, np.array([0,2])), (4, 2, np.array([1,-1])), (4, 0, np.array([1,0])), (4, 5, np.array([1,0])), (4, 2, np.array([1,1])), (5, 4, np.array([0,1])), (5, 4, np.array([1,-2])), (5, 0, np.array([1,-1])), (5, 0, np.array([1,1])), (5, 2, np.array([1,1])), (5, 2, np.array([2,-1])), (5, 4, np.array([2,-1]))],
            # d = 5.000000 ; d^3 = 125.00 
            n10_nearest_neighbors = [(0, 1, np.array([2,-1])), (1, 2, np.array([1,-2])), (1, 4, np.array([1,0])), (2, 3, np.array([1,-2])), (3, 0, np.array([1,0])), (4, 3, np.array([1,1]))],
            # d = 5.196152 ; d^3 = 140.30 
            n11_nearest_neighbors = [(0, 5, np.array([0,1])), (0, 2, np.array([2,-2])), (0, 4, np.array([2,-1])), (2, 4, np.array([0,1])), (2, 0, np.array([1,-1])), (2, 5, np.array([1,0])), (4, 2, np.array([0,2])), (4, 5, np.array([1,-1])), (4, 0, np.array([1,1])), (5, 0, np.array([0,2])), (5, 4, np.array([2,-2])), (5, 2, np.array([2,0]))],
            # d = 5.291503 ; d^3 = 148.16 
            n12_nearest_neighbors = [(1, 3, np.array([0,1])), (1, 5, np.array([0,1])), (1, 5, np.array([1,-2])), (1, 5, np.array([1,0])), (1, 3, np.array([2,-2])), (1, 3, np.array([2,-1])), (3, 1, np.array([0,2])), (3, 1, np.array([1,-1])), (3, 5, np.array([1,-1])), (3, 5, np.array([1,0])), (3, 1, np.array([1,1])), (5, 1, np.array([0,2])), (5, 3, np.array([1,-2])), (5, 3, np.array([1,1])), (5, 3, np.array([2,-2])), (5, 1, np.array([2,-1])), (5, 1, np.array([2,0])), (5, 3, np.array([2,0]))],
            # d = 5.567764 ; d^3 = 172.60 
            n13_nearest_neighbors = [(0, 3, np.array([0,1])), (0, 1, np.array([1,-2])), (0, 1, np.array([1,1])), (1, 4, np.array([2,-2])), (1, 2, np.array([2,-1])), (2, 1, np.array([1,1])), (2, 3, np.array([2,-1])), (3, 0, np.array([0,2])), (3, 4, np.array([1,-2])), (3, 2, np.array([1,1])), (4, 1, np.array([1,-1])), (4, 3, np.array([2,-1]))]        
        )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class TruncatedKagome(Lattice):
    """The vertices of the truncated trihexagonal tiling
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 12

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 12) 
        #    10---11
        #    /     \
        #   8       9
        #  /         \
        # 6           7
        # |           |
        # 4           5
        #  \         /
        #   2       3
        #    \     /
        #     0---1

        cos30 = 0.5 * (3**0.5)
        pos = np.array([[-0.5, -1-cos30], [0.5, -1-cos30],
                        [-0.5-cos30, -0.5-cos30], [0.5+cos30, -0.5-cos30],
                        [-1-cos30, -0.5], [1+cos30, -0.5],
                        [-1-cos30, 0.5], [1+cos30, 0.5],
                        [-0.5-cos30, 0.5+cos30], [0.5+cos30, 0.5+cos30],
                        [-0.5, 1+cos30], [0.5, 1+cos30]])

        basis = [[1.5 + 3*cos30, 1.5 + cos30], [0, 3 +2*cos30]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 3, np.array([0,0])), (2, 4, np.array([0,0])), (3, 5, np.array([0,0])), (3, 6, np.array([1,-1])), (4, 6, np.array([0,0])), (5, 7, np.array([0,0])), (5, 8, np.array([1,-1])), (6, 8, np.array([0,0])), (7, 9, np.array([0,0])), (7, 2, np.array([1,0])), (8, 10, np.array([0,0])), (9, 11, np.array([0,0])), (9, 4, np.array([1,0])), (10, 11, np.array([0,0])), (10, 0, np.array([0,1])), (11, 1, np.array([0,1]))],
            # d = 1.414214 ; d^3 = 2.83 
            n1_nearest_neighbors = [(3, 8, np.array([1,-1])), (5, 6, np.array([1,-1])), (7, 4, np.array([1,0])), (9, 2, np.array([1,0])), (10, 1, np.array([0,1])), (11, 0, np.array([0,1]))],
            # d = 1.732051 ; d^3 = 5.20 
            n2_nearest_neighbors = [(1, 6, np.array([1,-1])), (3, 4, np.array([1,-1])), (5, 10, np.array([1,-1])), (5, 2, np.array([1,0])), (7, 8, np.array([1,-1])), (7, 0, np.array([1,0])), (8, 0, np.array([0,1])), (9, 1, np.array([0,1])), (9, 6, np.array([1,0])), (10, 2, np.array([0,1])), (11, 3, np.array([0,1])), (11, 4, np.array([1,0]))],
            # d = 1.931852 ; d^3 = 7.21 
            n3_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (1, 2, np.array([0,0])), (1, 5, np.array([0,0])), (2, 6, np.array([0,0])), (3, 7, np.array([0,0])), (4, 8, np.array([0,0])), (5, 9, np.array([0,0])), (6, 10, np.array([0,0])), (7, 11, np.array([0,0])), (8, 11, np.array([0,0])), (9, 10, np.array([0,0]))],
            # d = 2.000000 ; d^3 = 8.00 
            n4_nearest_neighbors = [(1, 4, np.array([1,-1])), (5, 0, np.array([1,0])), (7, 10, np.array([1,-1])), (8, 2, np.array([0,1])), (9, 3, np.array([0,1])), (11, 6, np.array([1,0]))],
            # d = 2.394170 ; d^3 = 13.72 
            n5_nearest_neighbors = [(1, 8, np.array([1,-1])), (3, 10, np.array([1,-1])), (5, 4, np.array([1,-1])), (5, 4, np.array([1,0])), (7, 6, np.array([1,-1])), (7, 6, np.array([1,0])), (8, 1, np.array([0,1])), (9, 0, np.array([0,1])), (9, 0, np.array([1,0])), (10, 3, np.array([0,1])), (11, 2, np.array([0,1])), (11, 2, np.array([1,0]))],
            # d = 2.732051 ; d^3 = 20.39 
            n6_nearest_neighbors = [(0, 5, np.array([0,0])), (0, 6, np.array([0,0])), (0, 6, np.array([1,-1])), (1, 4, np.array([0,0])), (1, 7, np.array([0,0])), (2, 3, np.array([0,0])), (2, 8, np.array([0,0])), (3, 9, np.array([0,0])), (3, 2, np.array([1,-1])), (3, 2, np.array([1,0])), (4, 10, np.array([0,0])), (5, 11, np.array([0,0])), (5, 11, np.array([1,-1])), (6, 11, np.array([0,0])), (6, 0, np.array([0,1])), (7, 10, np.array([0,0])), (7, 1, np.array([0,1])), (7, 1, np.array([1,0])), (8, 9, np.array([0,0])), (9, 8, np.array([1,-1])), (9, 8, np.array([1,0])), (10, 4, np.array([0,1])), (10, 4, np.array([1,0])), (11, 5, np.array([0,1]))],
            # d = 2.909313 ; d^3 = 24.62 
            n7_nearest_neighbors = [(0, 4, np.array([1,-1])), (1, 2, np.array([1,-1])), (3, 0, np.array([1,0])), (5, 1, np.array([1,0])), (6, 2, np.array([0,1])), (7, 3, np.array([0,1])), (7, 11, np.array([1,-1])), (8, 4, np.array([0,1])), (9, 5, np.array([0,1])), (9, 10, np.array([1,-1])), (10, 6, np.array([1,0])), (11, 8, np.array([1,0]))],
            # d = 3.346065 ; d^3 = 37.46 
            n8_nearest_neighbors = [(0, 7, np.array([0,0])), (0, 8, np.array([0,0])), (0, 8, np.array([1,-1])), (1, 6, np.array([0,0])), (1, 9, np.array([0,0])), (2, 5, np.array([0,0])), (2, 10, np.array([0,0])), (3, 4, np.array([0,0])), (3, 11, np.array([0,0])), (3, 11, np.array([1,-1])), (3, 4, np.array([1,0])), (4, 11, np.array([0,0])), (5, 10, np.array([0,0])), (5, 2, np.array([1,-1])), (6, 9, np.array([0,0])), (6, 1, np.array([0,1])), (7, 8, np.array([0,0])), (7, 0, np.array([0,1])), (7, 8, np.array([1,0])), (9, 6, np.array([1,-1])), (9, 1, np.array([1,0])), (10, 5, np.array([0,1])), (10, 2, np.array([1,0])), (11, 4, np.array([0,1]))],
            # d = 3.385868 ; d^3 = 38.82 
            n9_nearest_neighbors = [(1, 10, np.array([1,-1])), (5, 6, np.array([1,0])), (7, 4, np.array([1,-1])), (8, 3, np.array([0,1])), (9, 2, np.array([0,1])), (11, 0, np.array([1,0]))],
            # d = 3.632651 ; d^3 = 47.94 
            n10_nearest_neighbors = [(1, 2, np.array([1,0])), (2, 6, np.array([1,-1])), (3, 0, np.array([1,-1])), (4, 0, np.array([0,1])), (5, 1, np.array([0,1])), (5, 9, np.array([1,-1])), (7, 3, np.array([1,0])), (8, 4, np.array([1,0])), (9, 10, np.array([1,0])), (10, 6, np.array([0,1])), (11, 7, np.array([0,1])), (11, 8, np.array([1,-1]))],
            # d = 3.732051 ; d^3 = 51.98 
            n11_nearest_neighbors = [(0, 9, np.array([0,0])), (0, 10, np.array([0,0])), (0, 2, np.array([1,-1])), (1, 8, np.array([0,0])), (1, 11, np.array([0,0])), (2, 7, np.array([0,0])), (2, 11, np.array([0,0])), (3, 6, np.array([0,0])), (3, 10, np.array([0,0])), (3, 1, np.array([1,0])), (4, 5, np.array([0,0])), (4, 9, np.array([0,0])), (5, 8, np.array([0,0])), (6, 7, np.array([0,0])), (6, 4, np.array([0,1])), (7, 5, np.array([0,1])), (9, 11, np.array([1,-1])), (10, 8, np.array([1,0]))],
            # d = 3.863703 ; d^3 = 57.68 
            n12_nearest_neighbors = [(0, 11, np.array([0,0])), (1, 10, np.array([0,0])), (2, 9, np.array([0,0])), (3, 8, np.array([0,0])), (4, 7, np.array([0,0])), (5, 6, np.array([0,0]))],
            # d = 3.898224 ; d^3 = 59.24 
            n13_nearest_neighbors = [(1, 0, np.array([1,-1])), (1, 0, np.array([1,0])), (2, 4, np.array([1,-1])), (4, 2, np.array([0,1])), (5, 3, np.array([0,1])), (5, 3, np.array([1,0])), (7, 9, np.array([1,-1])), (8, 6, np.array([0,1])), (8, 6, np.array([1,0])), (9, 7, np.array([0,1])), (11, 10, np.array([1,-1])), (11, 10, np.array([1,0]))]      
        )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class SnubHexagon(Lattice):
    """The vertices of the snub hexagonal tiling
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 6)  
        #    4---5
        #   /    \
        #  2      3
        #   \    /
        #   0---1 
        pos = np.array([[0, 0], [1, 0],
                        [-0.5, 0.5 * 3**0.5], [1.5, 0.5 * 3**0.5],
                        [0, 3**0.5], [1, 3**0.5]])
        basis = [[2, 3**0.5], [-0.5, 1.5 * 3**0.5]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
       
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 3, np.array([0,0])), (1, 2, np.array([1,-1])), (2, 4, np.array([0,0])), (3, 5, np.array([0,0])), (3, 2, np.array([1,-1])), (3, 4, np.array([1,-1])), (3, 0, np.array([1,0])), (4, 5, np.array([0,0])), (4, 0, np.array([0,1])), (4, 1, np.array([0,1])), (5, 1, np.array([0,1])), (5, 0, np.array([1,0])), (5, 2, np.array([1,0]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (1, 2, np.array([0,0])), (1, 5, np.array([0,0])), (1, 0, np.array([1,-1])), (1, 4, np.array([1,-1])), (2, 5, np.array([0,0])), (2, 0, np.array([0,1])), (3, 4, np.array([0,0])), (3, 1, np.array([1,0])), (3, 2, np.array([1,0])), (4, 2, np.array([1,0])), (5, 0, np.array([0,1])), (5, 3, np.array([0,1])), (5, 4, np.array([1,-1]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 5, np.array([0,0])), (0, 2, np.array([1,-1])), (1, 4, np.array([0,0])), (1, 0, np.array([1,0])), (2, 3, np.array([0,0])), (2, 1, np.array([0,1])), (3, 1, np.array([0,1])), (3, 0, np.array([1,-1])), (3, 5, np.array([1,-1])), (4, 2, np.array([0,1])), (4, 3, np.array([0,1])), (4, 0, np.array([1,0])), (5, 2, np.array([1,-1])), (5, 1, np.array([1,0])), (5, 4, np.array([1,0]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 1, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 4, np.array([1,-1])), (0, 0, np.array([1,0])), (1, 1, np.array([0,1])), (1, 4, np.array([1,-2])), (1, 1, np.array([1,-1])), (1, 5, np.array([1,-1])), (1, 1, np.array([1,0])), (1, 2, np.array([1,0])), (2, 2, np.array([0,1])), (2, 2, np.array([1,-1])), (2, 0, np.array([1,0])), (2, 2, np.array([1,0])), (3, 0, np.array([0,1])), (3, 3, np.array([0,1])), (3, 1, np.array([1,-1])), (3, 3, np.array([1,-1])), (3, 3, np.array([1,0])), (3, 4, np.array([1,0])), (3, 2, np.array([2,-1])), (4, 4, np.array([0,1])), (4, 5, np.array([0,1])), (4, 2, np.array([1,-1])), (4, 4, np.array([1,-1])), (4, 4, np.array([1,0])), (5, 2, np.array([0,1])), (5, 5, np.array([0,1])), (5, 5, np.array([1,-1])), (5, 3, np.array([1,0])), (5, 5, np.array([1,0])), (5, 0, np.array([1,1]))],
            # d = 3.000000 ; d^3 = 27.00 
            n4_nearest_neighbors = [(0, 2, np.array([1,0])), (1, 0, np.array([0,1])), (1, 2, np.array([1,-2])), (1, 3, np.array([1,-1])), (2, 3, np.array([0,1])), (2, 4, np.array([1,-1])), (3, 4, np.array([1,-2])), (3, 5, np.array([1,0])), (3, 0, np.array([2,-1])), (4, 1, np.array([1,0])), (4, 0, np.array([1,1])), (5, 4, np.array([0,1])), (5, 0, np.array([1,-1])), (5, 1, np.array([1,1])), (5, 2, np.array([2,-1]))],
            # d = 3.464102 ; d^3 = 41.57 
            n5_nearest_neighbors = [(0, 4, np.array([1,-2])), (0, 1, np.array([1,0])), (1, 3, np.array([0,1])), (1, 5, np.array([1,-2])), (1, 2, np.array([2,-1])), (2, 4, np.array([0,1])), (2, 0, np.array([1,-1])), (3, 0, np.array([1,1])), (3, 2, np.array([2,-2])), (3, 4, np.array([2,-1])), (4, 1, np.array([0,2])), (4, 5, np.array([1,0])), (5, 3, np.array([1,-1])), (5, 2, np.array([1,1])), (5, 0, np.array([2,0]))],
            # d = 3.605551 ; d^3 = 46.87 
            n6_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 3, np.array([0,1])), (0, 2, np.array([1,-2])), (0, 1, np.array([1,-1])), (0, 5, np.array([1,-1])), (1, 3, np.array([1,0])), (1, 4, np.array([1,0])), (1, 2, np.array([2,-2])), (1, 0, np.array([2,-1])), (2, 5, np.array([0,1])), (2, 1, np.array([1,0])), (2, 4, np.array([1,0])), (3, 2, np.array([0,1])), (3, 5, np.array([0,1])), (3, 2, np.array([1,-2])), (3, 5, np.array([1,-2])), (3, 1, np.array([1,1])), (3, 4, np.array([2,-2])), (3, 0, np.array([2,0])), (4, 0, np.array([0,2])), (4, 0, np.array([1,-1])), (4, 5, np.array([1,-1])), (4, 3, np.array([1,0])), (4, 1, np.array([1,1])), (4, 2, np.array([1,1])), (5, 1, np.array([0,2])), (5, 1, np.array([1,-1])), (5, 0, np.array([2,-1])), (5, 4, np.array([2,-1])), (5, 2, np.array([2,0]))],
            # d = 4.000000 ; d^3 = 64.00 
            n7_nearest_neighbors = [(0, 3, np.array([1,-1])), (0, 4, np.array([1,0])), (1, 2, np.array([0,1])), (1, 0, np.array([1,-2])), (1, 5, np.array([1,0])), (1, 4, np.array([2,-2])), (2, 5, np.array([1,-1])), (2, 0, np.array([1,1])), (3, 4, np.array([0,1])), (3, 1, np.array([2,-1])), (3, 2, np.array([2,0])), (4, 2, np.array([2,-1])), (5, 0, np.array([0,2])), (5, 4, np.array([1,-2])), (5, 3, np.array([1,1]))],
            # d = 4.358899 ; d^3 = 82.82 
            n8_nearest_neighbors = [(0, 4, np.array([0,1])), (0, 5, np.array([0,1])), (0, 5, np.array([1,-2])), (0, 3, np.array([1,0])), (0, 2, np.array([2,-1])), (1, 5, np.array([0,1])), (1, 3, np.array([1,-2])), (1, 0, np.array([1,1])), (1, 0, np.array([2,-2])), (1, 4, np.array([2,-1])), (2, 0, np.array([0,2])), (2, 1, np.array([0,2])), (2, 4, np.array([1,-2])), (2, 1, np.array([1,-1])), (2, 3, np.array([1,0])), (2, 5, np.array([1,0])), (3, 2, np.array([1,1])), (3, 0, np.array([2,-2])), (3, 5, np.array([2,-1])), (3, 1, np.array([2,0])), (4, 3, np.array([0,2])), (4, 1, np.array([1,-1])), (4, 3, np.array([1,-1])), (4, 0, np.array([2,0])), (4, 2, np.array([2,0])), (5, 3, np.array([0,2])), (5, 4, np.array([1,1])), (5, 2, np.array([2,-2])), (5, 4, np.array([2,-2])), (5, 1, np.array([2,0]))],
            # d = 4.582576 ; d^3 = 96.23 
            n9_nearest_neighbors = [(0, 0, np.array([1,-2])), (0, 5, np.array([1,0])), (0, 0, np.array([1,1])), (0, 2, np.array([2,-2])), (0, 0, np.array([2,-1])), (1, 4, np.array([0,1])), (1, 1, np.array([1,-2])), (1, 1, np.array([1,1])), (1, 1, np.array([2,-1])), (1, 0, np.array([2,0])), (2, 2, np.array([1,-2])), (2, 3, np.array([1,-1])), (2, 1, np.array([1,1])), (2, 2, np.array([1,1])), (2, 2, np.array([2,-1])), (3, 1, np.array([0,2])), (3, 0, np.array([1,-2])), (3, 3, np.array([1,-2])), (3, 3, np.array([1,1])), (3, 5, np.array([2,-2])), (3, 3, np.array([2,-1])), (4, 2, np.array([0,2])), (4, 4, np.array([1,-2])), (4, 3, np.array([1,1])), (4, 4, np.array([1,1])), (4, 0, np.array([2,-1])), (4, 4, np.array([2,-1])), (5, 2, np.array([1,-2])), (5, 5, np.array([1,-2])), (5, 5, np.array([1,1])), (5, 1, np.array([2,-1])), (5, 5, np.array([2,-1])), (5, 4, np.array([2,0]))],
            # d = 5.000000 ; d^3 = 125.00 
            n10_nearest_neighbors = [(0, 1, np.array([1,1])), (0, 4, np.array([2,-2])), (1, 4, np.array([1,-3])), (1, 5, np.array([2,-2])), (1, 2, np.array([2,0])), (2, 0, np.array([2,-1])), (3, 0, np.array([0,2])), (3, 1, np.array([1,-2])), (3, 4, np.array([2,0])), (3, 2, np.array([3,-2])), (4, 2, np.array([1,-2])), (4, 5, np.array([1,1])), (5, 2, np.array([0,2])), (5, 3, np.array([2,-1])), (5, 0, np.array([2,1]))],
            # d = 5.196152 ; d^3 = 140.30 
            n11_nearest_neighbors = [(0, 1, np.array([0,2])), (0, 3, np.array([1,-2])), (0, 4, np.array([2,-1])), (1, 2, np.array([1,1])), (1, 4, np.array([2,-3])), (1, 5, np.array([2,-1])), (2, 5, np.array([1,-2])), (2, 0, np.array([2,0])), (3, 4, np.array([1,1])), (3, 1, np.array([2,-2])), (3, 2, np.array([3,-1])), (4, 5, np.array([0,2])), (4, 2, np.array([2,-2])), (5, 0, np.array([1,2])), (5, 3, np.array([2,0]))],
            # d = 5.291503 ; d^3 = 148.16 
            n12_nearest_neighbors = [(0, 0, np.array([0,2])), (0, 1, np.array([1,-2])), (0, 2, np.array([1,1])), (0, 0, np.array([2,-2])), (0, 0, np.array([2,0])), (1, 1, np.array([0,2])), (1, 2, np.array([2,-3])), (1, 1, np.array([2,-2])), (1, 3, np.array([2,-1])), (1, 1, np.array([2,0])), (2, 2, np.array([0,2])), (2, 3, np.array([0,2])), (2, 2, np.array([2,-2])), (2, 4, np.array([2,-1])), (2, 2, np.array([2,0])), (3, 3, np.array([0,2])), (3, 5, np.array([1,1])), (3, 4, np.array([2,-3])), (3, 3, np.array([2,-2])), (3, 3, np.array([2,0])), (3, 0, np.array([3,-1])), (4, 4, np.array([0,2])), (4, 5, np.array([1,-2])), (4, 0, np.array([1,2])), (4, 4, np.array([2,-2])), (4, 1, np.array([2,0])), (4, 4, np.array([2,0])), (5, 5, np.array([0,2])), (5, 1, np.array([1,2])), (5, 0, np.array([2,-2])), (5, 5, np.array([2,-2])), (5, 5, np.array([2,0])), (5, 2, np.array([3,-1]))],
            # d = 5.567764 ; d^3 = 172.60 
            n13_nearest_neighbors = [(0, 4, np.array([1,-3])), (0, 1, np.array([2,-1])), (0, 2, np.array([2,0])), (1, 0, np.array([0,2])), (1, 2, np.array([1,-3])), (1, 5, np.array([1,-3])), (1, 3, np.array([1,1])), (1, 3, np.array([2,-2])), (1, 2, np.array([3,-2])), (2, 0, np.array([1,-2])), (2, 3, np.array([1,1])), (2, 4, np.array([1,1])), (2, 4, np.array([2,-2])), (3, 4, np.array([1,-3])), (3, 2, np.array([2,-3])), (3, 5, np.array([2,0])), (3, 0, np.array([2,1])), (3, 0, np.array([3,-2])), (3, 4, np.array([3,-2])), (4, 1, np.array([1,2])), (4, 1, np.array([2,-1])), (4, 5, np.array([2,-1])), (4, 0, np.array([2,1])), (5, 4, np.array([0,2])), (5, 0, np.array([1,-2])), (5, 3, np.array([1,-2])), (5, 1, np.array([2,1])), (5, 2, np.array([2,1])), (5, 2, np.array([3,-2])), (5, 0, np.array([3,-1]))],
        )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Floret(Lattice):
    """The vertices of the floret (rosette / 6-fold pentille) pentagonal tiling
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 9

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 9) 
        #     7
        #   /    \
        # 6        \
        # |          8
        # 5        /  |
        #   \    /    |
        #     4       |
        #     |       |
        #     3       1
        #       \   /   \
        # x       2       0

        cos30 = 0.5 * (3**0.5)
        pos = np.array([[4*cos30, 0], [3*cos30, 0.5], [2*cos30, 0],
                        [cos30, 0.5], [cos30, 1.5], [0, 2],
                        [0, 3], [cos30, 3.5], [3*cos30, 2.5]])

        basis = [[4*cos30, 3], [-cos30, 4.5]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 5, np.array([1,-1])), (1, 2, np.array([0,0])), (2, 3, np.array([0,0])), (3, 4, np.array([0,0])), (4, 5, np.array([0,0])), (5, 6, np.array([0,0])), (6, 7, np.array([0,0])), (7, 2, np.array([0,1]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 2, np.array([0,0])), (0, 4, np.array([1,-1])), (0, 6, np.array([1,-1])), (1, 3, np.array([0,0])), (1, 5, np.array([1,-1])), (2, 4, np.array([0,0])), (3, 5, np.array([0,0])), (4, 6, np.array([0,0])), (5, 7, np.array([0,0])), (6, 2, np.array([0,1])), (7, 1, np.array([0,1])), (7, 3, np.array([0,1]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 3, np.array([1,-1])), (1, 4, np.array([0,0])), (1, 8, np.array([0,0])), (1, 6, np.array([1,-1])), (4, 7, np.array([0,0])), (4, 8, np.array([0,0])), (6, 3, np.array([0,1])), (7, 8, np.array([0,0])), (7, 0, np.array([0,1])), (8, 0, np.array([0,1])), (8, 6, np.array([1,-1])), (8, 3, np.array([1,0]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 8, np.array([0,0])), (0, 7, np.array([1,-1])), (1, 4, np.array([1,-1])), (2, 5, np.array([0,0])), (2, 8, np.array([0,0])), (2, 5, np.array([1,-1])), (3, 6, np.array([0,0])), (3, 8, np.array([0,0])), (5, 8, np.array([0,0])), (5, 2, np.array([0,1])), (6, 8, np.array([0,0])), (6, 1, np.array([0,1])), (7, 4, np.array([0,1])), (8, 1, np.array([0,1])), (8, 2, np.array([0,1])), (8, 5, np.array([1,-1])), (8, 7, np.array([1,-1])), (8, 2, np.array([1,0])), (8, 4, np.array([1,0])), (8, 5, np.array([1,0]))],
            # d = 3.000000 ; d^3 = 27.00 
            n4_nearest_neighbors = [(0, 4, np.array([0,0])), (0, 2, np.array([1,-1])), (1, 5, np.array([0,0])), (1, 3, np.array([1,-1])), (1, 7, np.array([1,-1])), (2, 6, np.array([1,-1])), (3, 7, np.array([0,0])), (4, 2, np.array([0,1])), (5, 3, np.array([0,1])), (6, 0, np.array([0,1])), (6, 4, np.array([0,1])), (7, 5, np.array([1,0]))],
            # d = 3.464102 ; d^3 = 41.57 
            n5_nearest_neighbors = [(0, 6, np.array([1,-2])), (0, 2, np.array([1,0])), (1, 7, np.array([0,0])), (1, 3, np.array([1,0])), (2, 6, np.array([0,0])), (2, 4, np.array([1,-1])), (3, 5, np.array([1,-1])), (4, 0, np.array([0,1])), (4, 6, np.array([1,-1])), (5, 1, np.array([0,1])), (7, 5, np.array([0,1])), (7, 3, np.array([1,0]))],
            # d = 3.605551 ; d^3 = 46.87 
            n6_nearest_neighbors = [(0, 7, np.array([1,-2])), (0, 1, np.array([1,-1])), (0, 8, np.array([1,-1])), (0, 3, np.array([1,0])), (1, 6, np.array([0,0])), (1, 2, np.array([1,0])), (2, 7, np.array([0,0])), (2, 3, np.array([1,-1])), (3, 6, np.array([1,-1])), (4, 1, np.array([0,1])), (4, 3, np.array([0,1])), (4, 5, np.array([1,-1])), (5, 0, np.array([0,1])), (6, 5, np.array([0,1])), (7, 8, np.array([0,1])), (7, 4, np.array([1,0])), (7, 6, np.array([1,0])), (8, 3, np.array([0,1])), (8, 4, np.array([1,-1])), (8, 1, np.array([1,0])), (8, 6, np.array([1,0]))],
            # d = 4.000000 ; d^3 = 64.00 
            n7_nearest_neighbors = [(0, 5, np.array([0,0])), (1, 0, np.array([0,1])), (1, 2, np.array([1,-1])), (2, 7, np.array([1,-1])), (3, 2, np.array([0,1])), (4, 3, np.array([1,0])), (5, 4, np.array([0,1])), (6, 5, np.array([1,0])), (7, 6, np.array([1,-1]))],
            # d = 4.358899 ; d^3 = 82.82 
            n8_nearest_neighbors = [(0, 7, np.array([0,0])), (0, 5, np.array([1,-2])), (0, 1, np.array([1,0])), (1, 2, np.array([0,1])), (1, 6, np.array([1,-2])), (1, 8, np.array([1,-1])), (1, 4, np.array([1,0])), (2, 3, np.array([1,0])), (3, 0, np.array([0,1])), (3, 4, np.array([1,-1])), (4, 7, np.array([1,-1])), (4, 5, np.array([1,0])), (5, 6, np.array([1,-1])), (6, 8, np.array([0,1])), (6, 3, np.array([1,0])), (7, 6, np.array([0,1])), (7, 2, np.array([1,0])), (8, 4, np.array([0,1])), (8, 3, np.array([1,-1])), (8, 0, np.array([1,0])), (8, 7, np.array([1,0]))],
            # d = 4.582576 ; d^3 = 96.23 
            n9_nearest_neighbors = [(0, 6, np.array([0,0])), (0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 0, np.array([1,0])), (0, 4, np.array([1,0])), (1, 1, np.array([0,1])), (1, 7, np.array([1,-2])), (1, 1, np.array([1,-1])), (1, 1, np.array([1,0])), (1, 5, np.array([1,0])), (2, 0, np.array([0,1])), (2, 2, np.array([0,1])), (2, 6, np.array([1,-2])), (2, 2, np.array([1,-1])), (2, 2, np.array([1,0])), (3, 1, np.array([0,1])), (3, 3, np.array([0,1])), (3, 3, np.array([1,-1])), (3, 7, np.array([1,-1])), (3, 3, np.array([1,0])), (4, 4, np.array([0,1])), (4, 4, np.array([1,-1])), (4, 2, np.array([1,0])), (4, 4, np.array([1,0])), (5, 5, np.array([0,1])), (5, 5, np.array([1,-1])), (5, 3, np.array([1,0])), (5, 5, np.array([1,0])), (6, 6, np.array([0,1])), (6, 6, np.array([1,-1])), (6, 4, np.array([1,0])), (6, 6, np.array([1,0])), (7, 7, np.array([0,1])), (7, 5, np.array([1,-1])), (7, 7, np.array([1,-1])), (7, 7, np.array([1,0])), (8, 8, np.array([0,1])), (8, 8, np.array([1,-1])), (8, 8, np.array([1,0]))],
            # d = 5.000000 ; d^3 = 125.00 
            n10_nearest_neighbors = [(0, 5, np.array([1,0])), (1, 0, np.array([1,0])), (2, 1, np.array([0,1])), (2, 7, np.array([1,-2])), (3, 2, np.array([1,0])), (4, 3, np.array([1,-1])), (5, 4, np.array([1,0])), (6, 7, np.array([0,1])), (6, 5, np.array([1,-1]))],
            # d = 5.196152 ; d^3 = 140.30 
            n11_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 4, np.array([1,-2])), (0, 6, np.array([2,-2])), (1, 3, np.array([0,1])), (1, 5, np.array([1,-2])), (2, 4, np.array([1,0])), (3, 5, np.array([1,0])), (4, 6, np.array([1,0])), (5, 7, np.array([1,-1])), (6, 2, np.array([1,0])), (7, 1, np.array([1,0])), (7, 3, np.array([1,1]))],
            # d = 5.291503 ; d^3 = 148.16 
            n12_nearest_neighbors = [(0, 1, np.array([0,1])), (0, 5, np.array([2,-2])), (2, 3, np.array([0,1])), (2, 5, np.array([1,-2])), (2, 1, np.array([1,-1])), (2, 8, np.array([1,-1])), (2, 5, np.array([1,0])), (3, 4, np.array([1,0])), (4, 5, np.array([0,1])), (5, 8, np.array([0,1])), (5, 2, np.array([1,0])), (5, 6, np.array([1,0])), (6, 7, np.array([1,-1])), (7, 2, np.array([1,1])), (8, 5, np.array([0,1])), (8, 2, np.array([1,-1])), (8, 2, np.array([1,1])), (8, 5, np.array([2,-1]))],
            # d = 5.567764 ; d^3 = 172.60 
            n13_nearest_neighbors = [(0, 8, np.array([1,-2])), (0, 3, np.array([2,-1])), (0, 5, np.array([2,-1])), (1, 0, np.array([1,-1])), (1, 6, np.array([1,0])), (2, 1, np.array([1,0])), (3, 4, np.array([0,1])), (3, 6, np.array([1,-2])), (3, 2, np.array([1,-1])), (4, 8, np.array([0,1])), (4, 1, np.array([1,0])), (5, 6, np.array([0,1])), (5, 4, np.array([1,-1])), (6, 7, np.array([1,0])), (7, 0, np.array([0,2])), (7, 2, np.array([0,2])), (7, 4, np.array([1,-1])), (7, 8, np.array([1,0])), (8, 1, np.array([1,-1])), (8, 3, np.array([1,1])), (8, 6, np.array([2,-1]))]
        )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Isosnub(Lattice):
    """The vertices of the elongated triangular (isosnub quadrille) tiling
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 2

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites,2) 
        #  /
        # 1---
        # |
        # |
        # 0---
        #  \

        pos = np.array([[0., 0.], [0., 1.]])

        basis = [[1., 0.], [0.5 , 1.0 + 0.5 * 3**0.5]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 1, np.array([1,-1])), (0, 0, np.array([1,0])), (1, 0, np.array([0,1])), (1, 1, np.array([1,0]))],
            # d = 1.414214 ; d^3 = 2.83 
            n1_nearest_neighbors = [(0, 1, np.array([1,0])), (1, 0, np.array([1,0]))],
            # d = 1.732051 ; d^3 = 5.20 
            n2_nearest_neighbors = [(0, 1, np.array([2,-1])), (1, 0, np.array([1,1]))],
            # d = 1.931852 ; d^3 = 7.21 
            n3_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (1, 1, np.array([0,1])), (1, 1, np.array([1,-1]))],
            # d = 2.000000 ; d^3 = 8.00 
            n4_nearest_neighbors = [(0, 0, np.array([2,0])), (1, 1, np.array([2,0]))],
            # d = 2.236068 ; d^3 = 11.18 
            n5_nearest_neighbors = [(0, 1, np.array([2,0])), (1, 0, np.array([2,0]))],
            # d = 2.394170 ; d^3 = 13.72 
            n6_nearest_neighbors = [(0, 0, np.array([1,1])), (0, 0, np.array([2,-1])), (1, 1, np.array([1,1])), (1, 1, np.array([2,-1]))],
            # d = 2.645751 ; d^3 = 18.52 
            n7_nearest_neighbors = [(0, 1, np.array([3,-1])), (1, 0, np.array([2,1]))],
            # d = 2.732051 ; d^3 = 20.39 
            n8_nearest_neighbors = [(0, 1, np.array([1,-2]))],
            # d = 2.909313 ; d^3 = 24.62 
            n9_nearest_neighbors = [(0, 1, np.array([0,1])), (0, 1, np.array([2,-2])), (1, 0, np.array([0,2])), (1, 0, np.array([1,-1]))],
            # d = 3.000000 ; d^3 = 27.00 
            n10_nearest_neighbors = [(0, 0, np.array([3,0])), (1, 1, np.array([3,0]))],
            # d = 3.119623 ; d^3 = 30.36 
            n11_nearest_neighbors = [(0, 0, np.array([2,1])), (0, 0, np.array([3,-1])), (1, 1, np.array([2,1])), (1, 1, np.array([3,-1]))],
            # d = 3.162278 ; d^3 = 31.62 
            n12_nearest_neighbors = [(0, 1, np.array([3,0])), (1, 0, np.array([3,0]))],
            # d = 3.234826 ; d^3 = 33.85 
            n13_nearest_neighbors = [(0, 1, np.array([1,1])), (1, 0, np.array([2,-1]))],
        )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Prismatic(Lattice):
    """The vertices of the prismatic pentagonal tiling
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 3

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites,3) 
        #   /
        # 2
        # |
        # 1---
        # |
        # 0
        #  \

        pos = np.array([[0., 0.], [0., 1.], [0., 2.]])

        basis = [[3**0.5, 0.], [0.5 * 3 **0.5, 2.5]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([1,-1])), (1, 2, np.array([0,0])), (2, 0, np.array([0,1]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 1, np.array([1,-1])), (0, 0, np.array([1,0])), (1, 0, np.array([0,1])), (1, 2, np.array([1,-1])), (1, 1, np.array([1,0])), (2, 1, np.array([0,1])), (2, 2, np.array([1,0]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 2, np.array([0,0])), (0, 1, np.array([1,0])), (1, 0, np.array([1,0])), (1, 2, np.array([1,0])), (2, 1, np.array([1,0]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 2, np.array([1,0])), (0, 2, np.array([2,-1])), (1, 1, np.array([0,1])), (1, 1, np.array([1,-1])), (2, 2, np.array([0,1])), (2, 2, np.array([1,-1])), (2, 0, np.array([1,0])), (2, 0, np.array([1,1]))],
            # d = 3.000000 ; d^3 = 27.00 
            n4_nearest_neighbors = [(0, 2, np.array([1,-2])), (0, 1, np.array([2,-1])), (1, 0, np.array([1,1])), (1, 2, np.array([2,-1])), (2, 1, np.array([1,1]))],
            # d = 3.464102 ; d^3 = 41.57 
            n5_nearest_neighbors = [(0, 2, np.array([2,-2])), (0, 0, np.array([2,0])), (1, 1, np.array([2,0])), (2, 0, np.array([0,2])), (2, 2, np.array([2,0]))],
            # d = 3.605551 ; d^3 = 46.87 
            n6_nearest_neighbors = [(0, 1, np.array([0,1])), (0, 0, np.array([1,1])), (0, 0, np.array([2,-1])), (0, 1, np.array([2,0])), (1, 2, np.array([0,1])), (1, 0, np.array([1,-1])), (1, 1, np.array([1,1])), (1, 1, np.array([2,-1])), (1, 0, np.array([2,0])), (1, 2, np.array([2,0])), (2, 1, np.array([1,-1])), (2, 2, np.array([1,1])), (2, 2, np.array([2,-1])), (2, 1, np.array([2,0]))],
            # d = 4.000000 ; d^3 = 64.00 
            n7_nearest_neighbors = [(0, 1, np.array([1,-2])), (0, 2, np.array([2,0])), (1, 2, np.array([1,-2])), (2, 0, np.array([2,0]))],
            # d = 4.358899 ; d^3 = 82.82 
            n8_nearest_neighbors = [(0, 1, np.array([1,1])), (0, 1, np.array([2,-2])), (0, 2, np.array([3,-1])), (1, 0, np.array([0,2])), (1, 2, np.array([1,1])), (1, 2, np.array([2,-2])), (1, 0, np.array([2,-1])), (2, 1, np.array([0,2])), (2, 1, np.array([2,-1])), (2, 0, np.array([2,1]))],
            # d = 4.582576 ; d^3 = 96.23 
            n9_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 2, np.array([3,-2])), (0, 1, np.array([3,-1])), (1, 0, np.array([2,1])), (1, 2, np.array([3,-1])), (2, 0, np.array([1,-1])), (2, 0, np.array([1,2])), (2, 1, np.array([2,1]))],
            # d = 5.000000 ; d^3 = 125.00 
            n10_nearest_neighbors = [(0, 0, np.array([1,-2])), (0, 0, np.array([2,1])), (0, 0, np.array([3,-1])), (1, 1, np.array([1,-2])), (1, 1, np.array([2,1])), (1, 1, np.array([3,-1])), (2, 2, np.array([1,-2])), (2, 2, np.array([2,1])), (2, 2, np.array([3,-1]))],
            # d = 5.196152 ; d^3 = 140.30 
            n11_nearest_neighbors = [(0, 2, np.array([1,1])), (0, 0, np.array([3,0])), (1, 1, np.array([3,0])), (2, 0, np.array([2,-1])), (2, 2, np.array([3,0]))],
            # d = 5.291503 ; d^3 = 148.16 
            n12_nearest_neighbors = [(0, 0, np.array([0,2])), (0, 0, np.array([2,-2])), (0, 1, np.array([3,-2])), (0, 1, np.array([3,0])), (1, 1, np.array([0,2])), (1, 0, np.array([1,2])), (1, 1, np.array([2,-2])), (1, 2, np.array([3,-2])), (1, 0, np.array([3,0])), (1, 2, np.array([3,0])), (2, 2, np.array([0,2])), (2, 1, np.array([1,2])), (2, 2, np.array([2,-2])), (2, 1, np.array([3,0]))],
            # d = 5.567764 ; d^3 = 172.60 
            n13_nearest_neighbors = [(0, 2, np.array([1,-3])), (0, 2, np.array([2,-3])), (0, 1, np.array([2,1])), (0, 2, np.array([3,0])), (1, 2, np.array([2,1])), (1, 0, np.array([3,-1])), (2, 1, np.array([3,-1])), (2, 0, np.array([3,0]))],
       )
        kwargs['pairs'] = pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class RubyXC(Lattice):
    """A ruby lattice with aspect ratio rho=sqrt(3)  (kagome links)
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 6) 
        #     5        
        #    / \  
        #   3---4
        #   |   |
        #   |   |
        #   1---2
        #   \   / 
        #     0  
      
        cos30 = 0.5 * 3**0.5
        pos = np.array([[0, 0],  [-0.5, cos30],  [0.5, cos30],
                       [-0.5, 3*cos30], [0.5, 3*cos30], [0, 4*cos30]])

        basis = [[2, -4*cos30], [0, 8*cos30]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        
        ruby_pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 2, np.array([0,0])), (3, 4, np.array([0,0])), (3, 5, np.array([0,0])), (4, 5, np.array([0,0]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 3, np.array([1,0])), (1, 3, np.array([0,0])), (2, 4, np.array([0,0])), (2, 5, np.array([1,0])), (4, 0, np.array([1,1])), (5, 1, np.array([1,1]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 5, np.array([1,0])), (1, 4, np.array([0,0])), (2, 3, np.array([0,0])), (2, 3, np.array([1,0])), (4, 1, np.array([1,1])), (5, 0, np.array([1,1]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (0, 4, np.array([1,0])), (1, 5, np.array([0,0])), (1, 3, np.array([1,0])), (1, 5, np.array([1,0])), (2, 5, np.array([0,0])), (2, 4, np.array([1,0])), (3, 0, np.array([1,1])), (3, 1, np.array([1,1])), (4, 2, np.array([1,1])), (5, 2, np.array([1,1]))],
            # d = 3.000000 ; d^3 = 27.00 
            n4_nearest_neighbors = [(0, 1, np.array([1,0])), (2, 0, np.array([1,1])), (2, 1, np.array([2,1])), (4, 5, np.array([1,0])), (4, 3, np.array([2,1])), (5, 3, np.array([1,1]))],
            # d = 3.464102 ; d^3 = 41.57 
            n5_nearest_neighbors = [(0, 5, np.array([0,0])), (1, 4, np.array([1,0])), (2, 3, np.array([2,1])), (3, 2, np.array([1,1])), (4, 1, np.array([2,1])), (5, 0, np.array([0,1]))],
            # d = 3.605551 ; d^3 = 46.87 
            n6_nearest_neighbors = [(0, 2, np.array([1,0])), (0, 1, np.array([2,1])), (1, 0, np.array([1,1])), (2, 1, np.array([1,0])), (2, 1, np.array([1,1])), (2, 0, np.array([2,1])), (3, 5, np.array([1,0])), (4, 3, np.array([1,0])), (4, 3, np.array([1,1])), (4, 5, np.array([2,1])), (5, 4, np.array([1,1])), (5, 3, np.array([2,1]))],
            # d = 4.000000 ; d^3 = 64.00 
            n7_nearest_neighbors = [(0, 0, np.array([1,0])), (0, 0, np.array([1,1])), (0, 0, np.array([2,1])), (1, 1, np.array([1,0])), (1, 1, np.array([1,1])), (1, 1, np.array([2,1])), (2, 2, np.array([1,0])), (2, 2, np.array([1,1])), (2, 2, np.array([2,1])), (3, 3, np.array([1,0])), (3, 3, np.array([1,1])), (3, 3, np.array([2,1])), (4, 4, np.array([1,0])), (4, 4, np.array([1,1])), (4, 4, np.array([2,1])), (5, 5, np.array([1,0])), (5, 5, np.array([1,1])), (5, 5, np.array([2,1]))],
            # d = 4.358899 ; d^3 = 82.82 
            n8_nearest_neighbors = [(0, 3, np.array([2,1])), (1, 3, np.array([2,1])), (2, 4, np.array([2,1])), (2, 5, np.array([2,1])), (3, 0, np.array([0,1])), (3, 1, np.array([2,1])), (4, 0, np.array([0,1])), (4, 0, np.array([2,1])), (4, 2, np.array([2,1])), (5, 1, np.array([0,1])), (5, 2, np.array([0,1])), (5, 1, np.array([2,1]))],
            # d = 4.582576 ; d^3 = 96.23 
            n9_nearest_neighbors = [(0, 1, np.array([1,1])), (0, 2, np.array([2,1])), (1, 2, np.array([1,0])), (1, 2, np.array([1,1])), (1, 0, np.array([2,1])), (2, 0, np.array([1,0])), (3, 4, np.array([1,0])), (3, 4, np.array([1,1])), (3, 5, np.array([2,1])), (4, 5, np.array([1,1])), (5, 3, np.array([1,0])), (5, 4, np.array([2,1]))],
            # d = 5.000000 ; d^3 = 125.00 
            n10_nearest_neighbors = [(0, 2, np.array([1,1])), (1, 0, np.array([1,0])), (1, 2, np.array([2,1])), (3, 5, np.array([1,1])), (3, 4, np.array([2,1])), (5, 4, np.array([1,0]))],
            # d = 5.196152 ; d^3 = 140.30 
            n11_nearest_neighbors = [(0, 4, np.array([2,1])), (1, 5, np.array([2,1])), (3, 1, np.array([0,1])), (3, 0, np.array([2,1])), (4, 2, np.array([0,1])), (5, 2, np.array([2,1]))],
            # d = 5.291503 ; d^3 = 148.16 
            n12_nearest_neighbors = [(0, 5, np.array([2,0])), (0, 5, np.array([2,1])), (1, 4, np.array([2,1])), (2, 3, np.array([1,1])), (2, 3, np.array([3,1])), (3, 2, np.array([0,1])), (3, 2, np.array([2,1])), (4, 1, np.array([0,1])), (4, 1, np.array([1,0])), (4, 1, np.array([3,2])), (5, 0, np.array([2,1])), (5, 0, np.array([2,2]))],
            # d = 5.567764 ; d^3 = 172.60 
            n13_nearest_neighbors = [(0, 3, np.array([2,0])), (0, 3, np.array([3,1])), (1, 3, np.array([1,1])), (2, 4, np.array([1,1])), (2, 5, np.array([2,0])), (2, 5, np.array([3,1])), (3, 1, np.array([1,0])), (4, 2, np.array([1,0])), (4, 0, np.array([2,2])), (4, 0, np.array([3,2])), (5, 1, np.array([2,2])), (5, 1, np.array([3,2]))],
        )
        
        kwargs['pairs'] = ruby_pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class RubyXC_square(Lattice):
    """A ruby lattice with aspect ratio rho=1 (Archimedian)
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, rho= 1, **kwargs):
        sites = _parse_sites(sites, 6) 
        #     5        
        #    / \  
        #   3---4
        #   |   |
        #   |   |
        #   1---2
        #   \   / 
        #     0  
      
        pos = np.array([[0, 0],  [-0.5,  0.5*(3**0.5)],  [0.5, 0.5*(3**0.5)],
                       [-0.5,  0.5*(3**0.5) + 1], [0.5,  0.5*(3**0.5) + 1], [0,  1+ (3**0.5)]])

        basis = [[0.5*(1+(3**0.5)), -0.5*(3+(3**0.5))], [0, (3**0.5)+3]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        
        #TODO: Because rho is general, have to compute the pairs when lattice is initialized


        ruby_pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (0, 3, np.array([1,0])), (1, 2, np.array([0,0])), (1, 3, np.array([0,0])), (2, 4, np.array([0,0])), (2, 5, np.array([1,0])), (3, 4, np.array([0,0])), (3, 5, np.array([0,0])), (4, 5, np.array([0,0])), (4, 0, np.array([1,1])), (5, 1, np.array([1,1]))],
            # d = 1.414214 ; d^3 = 2.83 
            n1_nearest_neighbors = [(0, 5, np.array([1,0])), (1, 4, np.array([0,0])), (2, 3, np.array([0,0])), (2, 3, np.array([1,0])), (4, 1, np.array([1,1])), (5, 0, np.array([1,1]))],
            # d = 1.732051 ; d^3 = 5.20 
            n2_nearest_neighbors = [(0, 1, np.array([1,0])), (2, 0, np.array([1,1])), (2, 1, np.array([2,1])), (4, 5, np.array([1,0])), (4, 3, np.array([2,1])), (5, 3, np.array([1,1]))],
            # d = 1.931852 ; d^3 = 7.21 
            n3_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (0, 4, np.array([1,0])), (1, 5, np.array([0,0])), (1, 3, np.array([1,0])), (1, 5, np.array([1,0])), (2, 5, np.array([0,0])), (2, 4, np.array([1,0])), (3, 0, np.array([1,1])), (3, 1, np.array([1,1])), (4, 2, np.array([1,1])), (5, 2, np.array([1,1]))],
            # d = 2.000000 ; d^3 = 8.00 
            n4_nearest_neighbors = [(2, 3, np.array([2,1])), (4, 1, np.array([2,1])), (5, 0, np.array([0,1]))],
            # d = 2.394170 ; d^3 = 13.72 
            n5_nearest_neighbors = [(0, 2, np.array([1,0])), (0, 1, np.array([2,1])), (1, 0, np.array([1,1])), (2, 1, np.array([1,0])), (2, 1, np.array([1,1])), (2, 0, np.array([2,1])), (3, 5, np.array([1,0])), (4, 3, np.array([1,0])), (4, 3, np.array([1,1])), (4, 5, np.array([2,1])), (5, 4, np.array([1,1])), (5, 3, np.array([2,1]))],
            # d = 2.732051 ; d^3 = 20.39 
            n6_nearest_neighbors = [(0, 5, np.array([0,0])), (0, 0, np.array([1,0])), (0, 0, np.array([1,1])), (0, 0, np.array([2,1])), (1, 1, np.array([1,0])), (1, 4, np.array([1,0])), (1, 1, np.array([1,1])), (1, 1, np.array([2,1])), (2, 2, np.array([1,0])), (2, 2, np.array([1,1])), (2, 2, np.array([2,1])), (3, 3, np.array([1,0])), (3, 2, np.array([1,1])), (3, 3, np.array([1,1])), (3, 3, np.array([2,1])), (4, 4, np.array([1,0])), (4, 4, np.array([1,1])), (4, 4, np.array([2,1])), (5, 5, np.array([1,0])), (5, 5, np.array([1,1])), (5, 5, np.array([2,1]))],
            # d = 2.909313 ; d^3 = 24.62 
            n7_nearest_neighbors = [(0, 3, np.array([2,1])), (1, 3, np.array([2,1])), (2, 4, np.array([2,1])), (2, 5, np.array([2,1])), (3, 0, np.array([0,1])), (3, 1, np.array([2,1])), (4, 0, np.array([0,1])), (4, 0, np.array([2,1])), (4, 2, np.array([2,1])), (5, 1, np.array([0,1])), (5, 2, np.array([0,1])), (5, 1, np.array([2,1]))],
            # d = 3.346065 ; d^3 = 37.46 
            n8_nearest_neighbors = [(0, 1, np.array([1,1])), (0, 2, np.array([2,1])), (1, 2, np.array([1,0])), (1, 2, np.array([1,1])), (1, 0, np.array([2,1])), (2, 0, np.array([1,0])), (3, 4, np.array([1,0])), (3, 4, np.array([1,1])), (3, 5, np.array([2,1])), (4, 5, np.array([1,1])), (5, 3, np.array([1,0])), (5, 4, np.array([2,1]))],
            # d = 3.385868 ; d^3 = 38.82 
            n9_nearest_neighbors = [(0, 5, np.array([2,0])), (2, 3, np.array([1,1])), (2, 3, np.array([3,1])), (4, 1, np.array([1,0])), (4, 1, np.array([3,2])), (5, 0, np.array([2,2]))],
            # d = 3.632651 ; d^3 = 47.94 
            n10_nearest_neighbors = [(0, 3, np.array([2,0])), (0, 3, np.array([3,1])), (1, 3, np.array([1,1])), (2, 4, np.array([1,1])), (2, 5, np.array([2,0])), (2, 5, np.array([3,1])), (3, 1, np.array([1,0])), (4, 2, np.array([1,0])), (4, 0, np.array([2,2])), (4, 0, np.array([3,2])), (5, 1, np.array([2,2])), (5, 1, np.array([3,2]))],
            # d = 3.732051 ; d^3 = 51.98 
            n11_nearest_neighbors = [(0, 2, np.array([1,1])), (0, 4, np.array([2,1])), (1, 0, np.array([1,0])), (1, 2, np.array([2,1])), (1, 5, np.array([2,1])), (3, 1, np.array([0,1])), (3, 5, np.array([1,1])), (3, 0, np.array([2,1])), (3, 4, np.array([2,1])), (4, 2, np.array([0,1])), (5, 4, np.array([1,0])), (5, 2, np.array([2,1]))],
            # d = 3.863703 ; d^3 = 57.68 
            n12_nearest_neighbors = [(0, 5, np.array([2,1])), (1, 4, np.array([2,1])), (3, 2, np.array([0,1])), (3, 2, np.array([2,1])), (4, 1, np.array([0,1])), (5, 0, np.array([2,1]))],
            # d = 3.898224 ; d^3 = 59.24 
            n13_nearest_neighbors = [(0, 1, np.array([3,1])), (1, 0, np.array([0,1])), (2, 0, np.array([0,1])), (2, 1, np.array([3,1])), (2, 0, np.array([3,2])), (2, 1, np.array([3,2])), (4, 3, np.array([3,1])), (4, 5, np.array([3,1])), (4, 3, np.array([3,2])), (5, 3, np.array([0,1])), (5, 4, np.array([0,1])), (5, 3, np.array([3,2]))],
        )

        kwargs['pairs'] = ruby_pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class RubyXC_rhoVertices(Lattice):
    """NOTE: Auxilliary for "RubyXC_rho" only. Contains just the sites/vertices, for finding the pairs.
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    rho : float
        Anisotropy ratio of the rectangles. 
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    def __init__(self, Lx, Ly, sites, rho= 1, **kwargs):
        sites = _parse_sites(sites, 6) 
        #     5        
        #    / \  
        #   3---4
        #   |   |
        #   |   |
        #   1---2
        #   \   / 
        #     0  
      
        pos = np.array([[0, 0],  [-0.5, 0.5*(3**0.5)],  [0.5, 0.5*(3**0.5)],
                       [-0.5, 0.5*(3**0.5)+rho], [0.5, 0.5*(3**0.5)+rho], [0, (3**0.5)+rho]])

        basis = [[0.5*(1+(3**0.5)*rho), -0.5*(3*rho+(3**0.5))], [0, (3**0.5)+3*rho]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)
    

class RubyXC_rho(Lattice):
    """A ruby lattice with a general aspect ratio rho (kwargs)
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    rho : float
        Anisotropy ratio of the rectangles. 
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, rho= 1, **kwargs):
        sites = _parse_sites(sites, 6) 
        #     5        
        #    / \  
        #   3---4
        #   |   |
        #   |   |
        #   1---2
        #   \   / 
        #     0  
      
        pos = np.array([[0, 0],  [-0.5, 0.5*(3**0.5)],  [0.5, 0.5*(3**0.5)],
                       [-0.5, 0.5*(3**0.5)+rho], [0.5, 0.5*(3**0.5)+rho], [0, (3**0.5)+rho]])

        basis = [[0.5*(1+(3**0.5)*rho), -0.5*(3*rho+(3**0.5))], [0, (3**0.5)+3*rho]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        ### Construct the pairs using the RubyXC_rhoVertices class as an auxilliary lattice
        # Create a large lattice patch - size unimportant, just need to call lat.position()
        cutoff = 8
        aux_lat = RubyXC_rhoVertices(Lx=cutoff, Ly=cutoff, sites=None, rho=rho, bc=['open', 'open'])
        # Use the tenpy method to get them
        aux_dict = aux_lat.find_coupling_pairs(cutoff)

        # The aux dictionary keys = coupling distance; we change to "nearest_neighbor" style
        new_dict = {}
        nNN = 0
        for old_key in aux_dict.keys():
            new_key = 'nearest_neighbors' if nNN==0 else 'n%d_nearest_neighbors' %nNN
            new_dict[new_key]=aux_dict[old_key]
            nNN+=1
        
        kwargs['pairs'] = new_dict
        
        #TODO: Because rho is general, have to compute the pairs when lattice is initialized


        # ruby_pairs = dict(
        #     # d = 1.000000 ; d^3 = 1.00 
        #     nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 2, np.array([0,0])), (3, 4, np.array([0,0])), (3, 5, np.array([0,0])), (4, 5, np.array([0,0]))],
        #     # d = 1.732051 ; d^3 = 5.20 
        #     n1_nearest_neighbors = [(0, 3, np.array([1,0])), (1, 3, np.array([0,0])), (2, 4, np.array([0,0])), (2, 5, np.array([1,0])), (4, 0, np.array([1,1])), (5, 1, np.array([1,1]))],
        #     # d = 2.000000 ; d^3 = 8.00 
        #     n2_nearest_neighbors = [(0, 5, np.array([1,0])), (1, 4, np.array([0,0])), (2, 3, np.array([0,0])), (2, 3, np.array([1,0])), (4, 1, np.array([1,1])), (5, 0, np.array([1,1]))],
        #     # d = 2.645751 ; d^3 = 18.52 
        #     n3_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (0, 4, np.array([1,0])), (1, 5, np.array([0,0])), (1, 3, np.array([1,0])), (1, 5, np.array([1,0])), (2, 5, np.array([0,0])), (2, 4, np.array([1,0])), (3, 0, np.array([1,1])), (3, 1, np.array([1,1])), (4, 2, np.array([1,1])), (5, 2, np.array([1,1]))],
        #     # d = 3.000000 ; d^3 = 27.00 
        #     n4_nearest_neighbors = [(0, 1, np.array([1,0])), (2, 0, np.array([1,1])), (2, 1, np.array([2,1])), (4, 5, np.array([1,0])), (4, 3, np.array([2,1])), (5, 3, np.array([1,1]))],
        #     # d = 3.464102 ; d^3 = 41.57 
        #     n5_nearest_neighbors = [(0, 5, np.array([0,0])), (1, 4, np.array([1,0])), (2, 3, np.array([2,1])), (3, 2, np.array([1,1])), (4, 1, np.array([2,1])), (5, 0, np.array([0,1]))],
        #     # d = 3.605551 ; d^3 = 46.87 
        #     n6_nearest_neighbors = [(0, 2, np.array([1,0])), (0, 1, np.array([2,1])), (1, 0, np.array([1,1])), (2, 1, np.array([1,0])), (2, 1, np.array([1,1])), (2, 0, np.array([2,1])), (3, 5, np.array([1,0])), (4, 3, np.array([1,0])), (4, 3, np.array([1,1])), (4, 5, np.array([2,1])), (5, 4, np.array([1,1])), (5, 3, np.array([2,1]))],
        #     # d = 4.000000 ; d^3 = 64.00 
        #     n7_nearest_neighbors = [(0, 0, np.array([1,0])), (0, 0, np.array([1,1])), (0, 0, np.array([2,1])), (1, 1, np.array([1,0])), (1, 1, np.array([1,1])), (1, 1, np.array([2,1])), (2, 2, np.array([1,0])), (2, 2, np.array([1,1])), (2, 2, np.array([2,1])), (3, 3, np.array([1,0])), (3, 3, np.array([1,1])), (3, 3, np.array([2,1])), (4, 4, np.array([1,0])), (4, 4, np.array([1,1])), (4, 4, np.array([2,1])), (5, 5, np.array([1,0])), (5, 5, np.array([1,1])), (5, 5, np.array([2,1]))],
        #     # d = 4.358899 ; d^3 = 82.82 
        #     n8_nearest_neighbors = [(0, 3, np.array([2,1])), (1, 3, np.array([2,1])), (2, 4, np.array([2,1])), (2, 5, np.array([2,1])), (3, 0, np.array([0,1])), (3, 1, np.array([2,1])), (4, 0, np.array([0,1])), (4, 0, np.array([2,1])), (4, 2, np.array([2,1])), (5, 1, np.array([0,1])), (5, 2, np.array([0,1])), (5, 1, np.array([2,1]))],
        #     # d = 4.582576 ; d^3 = 96.23 
        #     n9_nearest_neighbors = [(0, 1, np.array([1,1])), (0, 2, np.array([2,1])), (1, 2, np.array([1,0])), (1, 2, np.array([1,1])), (1, 0, np.array([2,1])), (2, 0, np.array([1,0])), (3, 4, np.array([1,0])), (3, 4, np.array([1,1])), (3, 5, np.array([2,1])), (4, 5, np.array([1,1])), (5, 3, np.array([1,0])), (5, 4, np.array([2,1]))],
        #     # d = 5.000000 ; d^3 = 125.00 
        #     n10_nearest_neighbors = [(0, 2, np.array([1,1])), (1, 0, np.array([1,0])), (1, 2, np.array([2,1])), (3, 5, np.array([1,1])), (3, 4, np.array([2,1])), (5, 4, np.array([1,0]))],
        #     # d = 5.196152 ; d^3 = 140.30 
        #     n11_nearest_neighbors = [(0, 4, np.array([2,1])), (1, 5, np.array([2,1])), (3, 1, np.array([0,1])), (3, 0, np.array([2,1])), (4, 2, np.array([0,1])), (5, 2, np.array([2,1]))],
        #     # d = 5.291503 ; d^3 = 148.16 
        #     n12_nearest_neighbors = [(0, 5, np.array([2,0])), (0, 5, np.array([2,1])), (1, 4, np.array([2,1])), (2, 3, np.array([1,1])), (2, 3, np.array([3,1])), (3, 2, np.array([0,1])), (3, 2, np.array([2,1])), (4, 1, np.array([0,1])), (4, 1, np.array([1,0])), (4, 1, np.array([3,2])), (5, 0, np.array([2,1])), (5, 0, np.array([2,2]))],
        #     # d = 5.567764 ; d^3 = 172.60 
        #     n13_nearest_neighbors = [(0, 3, np.array([2,0])), (0, 3, np.array([3,1])), (1, 3, np.array([1,1])), (2, 4, np.array([1,1])), (2, 5, np.array([2,0])), (2, 5, np.array([3,1])), (3, 1, np.array([1,0])), (4, 2, np.array([1,0])), (4, 0, np.array([2,2])), (4, 0, np.array([3,2])), (5, 1, np.array([2,2])), (5, 1, np.array([3,2]))],
        # )
        
        # kwargs['pairs'] = ruby_pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class RubyXC_three(Lattice):
    """A ruby lattice with a general aspect rho=3 
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    rho : float
        Anisotropy ratio of the rectangles. 
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, rho= 1, **kwargs):
        sites = _parse_sites(sites, 6) 
        #     5        
        #    / \  
        #   3---4
        #   |   |
        #   |   |
        #   1---2
        #   \   / 
        #     0  
        rho=3

        pos = np.array([[0, 0],  [-0.5, 0.5*(3**0.5)],  [0.5, 0.5*(3**0.5)],
                       [-0.5, 0.5*(3**0.5)+rho], [0.5, 0.5*(3**0.5)+rho], [0, (3**0.5)+rho]])

        basis = [[0.5*(1+(3**0.5)*rho), -0.5*(3*rho+(3**0.5))], [0, (3**0.5)+3*rho]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        
        #TODO: Because rho is general, have to compute the pairs when lattice is initialized


        ruby_pairs = dict(
        # d = 1.000000 ; d^3 = 1.00 
        nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 2, np.array([0,0])), (3, 4, np.array([0,0])), (3, 5, np.array([0,0])), (4, 5, np.array([0,0]))],
        # d = 3.000000 ; d^3 = 27.00 
        n1_nearest_neighbors = [(0, 3, np.array([1,0])), (1, 3, np.array([0,0])), (2, 4, np.array([0,0])), (2, 5, np.array([1,0])), (4, 0, np.array([1,1])), (5, 1, np.array([1,1]))],
        # d = 3.162278 ; d^3 = 31.62 
        n2_nearest_neighbors = [(0, 5, np.array([1,0])), (1, 4, np.array([0,0])), (2, 3, np.array([0,0])), (2, 3, np.array([1,0])), (4, 1, np.array([1,1])), (5, 0, np.array([1,1]))],
        # d = 3.898224 ; d^3 = 59.24 
        n3_nearest_neighbors = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (0, 4, np.array([1,0])), (1, 5, np.array([0,0])), (1, 3, np.array([1,0])), (1, 5, np.array([1,0])), (2, 5, np.array([0,0])), (2, 4, np.array([1,0])), (3, 0, np.array([1,1])), (3, 1, np.array([1,1])), (4, 2, np.array([1,1])), (5, 2, np.array([1,1]))],
        # d = 4.732051 ; d^3 = 105.96 
        n4_nearest_neighbors = [(0, 5, np.array([0,0])), (1, 4, np.array([1,0])), (3, 2, np.array([1,1]))],
        # d = 5.196152 ; d^3 = 140.30 
        n5_nearest_neighbors = [(0, 1, np.array([1,0])), (2, 0, np.array([1,1])), (2, 1, np.array([2,1])), (4, 5, np.array([1,0])), (4, 3, np.array([2,1])), (5, 3, np.array([1,1]))],
        # d = 5.761610 ; d^3 = 191.26 
        n6_nearest_neighbors = [(0, 2, np.array([1,0])), (0, 1, np.array([2,1])), (1, 0, np.array([1,1])), (2, 1, np.array([1,0])), (2, 1, np.array([1,1])), (2, 0, np.array([2,1])), (3, 5, np.array([1,0])), (4, 3, np.array([1,0])), (4, 3, np.array([1,1])), (4, 5, np.array([2,1])), (5, 4, np.array([1,1])), (5, 3, np.array([2,1]))],
        # d = 6.000000 ; d^3 = 216.00 
        n7_nearest_neighbors = [(2, 3, np.array([2,1])), (4, 1, np.array([2,1])), (5, 0, np.array([0,1]))],
        # d = 6.196152 ; d^3 = 237.88 
        n8_nearest_neighbors = [(0, 0, np.array([1,0])), (0, 0, np.array([1,1])), (0, 0, np.array([2,1])), (1, 1, np.array([1,0])), (1, 1, np.array([1,1])), (1, 1, np.array([2,1])), (2, 2, np.array([1,0])), (2, 2, np.array([1,1])), (2, 2, np.array([2,1])), (3, 3, np.array([1,0])), (3, 3, np.array([1,1])), (3, 3, np.array([2,1])), (4, 4, np.array([1,0])), (4, 4, np.array([1,1])), (4, 4, np.array([2,1])), (5, 5, np.array([1,0])), (5, 5, np.array([1,1])), (5, 5, np.array([2,1]))],
        # d = 6.751922 ; d^3 = 307.81 
        n9_nearest_neighbors = [(0, 1, np.array([1,1])), (0, 2, np.array([2,1])), (1, 2, np.array([1,0])), (1, 2, np.array([1,1])), (1, 0, np.array([2,1])), (2, 0, np.array([1,0])), (3, 4, np.array([1,0])), (3, 4, np.array([1,1])), (3, 5, np.array([2,1])), (4, 5, np.array([1,1])), (5, 3, np.array([1,0])), (5, 4, np.array([2,1]))],
        # d = 6.884207 ; d^3 = 326.26 
        n10_nearest_neighbors = [(0, 3, np.array([2,1])), (1, 3, np.array([2,1])), (2, 4, np.array([2,1])), (2, 5, np.array([2,1])), (3, 0, np.array([0,1])), (3, 1, np.array([2,1])), (4, 0, np.array([0,1])), (4, 0, np.array([2,1])), (4, 2, np.array([2,1])), (5, 1, np.array([0,1])), (5, 2, np.array([0,1])), (5, 1, np.array([2,1]))],
        # d = 7.196152 ; d^3 = 372.65 
        n11_nearest_neighbors = [(0, 2, np.array([1,1])), (1, 0, np.array([1,0])), (1, 2, np.array([2,1])), (3, 5, np.array([1,1])), (3, 4, np.array([2,1])), (5, 4, np.array([1,0]))],
        # d = 7.732051 ; d^3 = 462.26 
        n12_nearest_neighbors = [(0, 4, np.array([2,1])), (1, 5, np.array([2,1])), (3, 1, np.array([0,1])), (3, 0, np.array([2,1])), (4, 2, np.array([0,1])), (5, 2, np.array([2,1]))],
        # d = 7.796449 ; d^3 = 473.90 
        n13_nearest_neighbors = [(0, 5, np.array([2,1])), (1, 4, np.array([2,1])), (3, 2, np.array([0,1])), (3, 2, np.array([2,1])), (4, 1, np.array([0,1])), (5, 0, np.array([2,1]))],
        # d = 8.625097 ; d^3 = 641.64 
        n14_nearest_neighbors = [(0, 5, np.array([2,0])), (2, 3, np.array([1,1])), (2, 3, np.array([3,1])), (4, 1, np.array([1,0])), (4, 1, np.array([3,2])), (5, 0, np.array([2,2]))],
        # d = 8.921236 ; d^3 = 710.03 
        n15_nearest_neighbors = [(0, 3, np.array([2,0])), (0, 3, np.array([3,1])), (1, 3, np.array([1,1])), (2, 4, np.array([1,1])), (2, 5, np.array([2,0])), (2, 5, np.array([3,1])), (3, 1, np.array([1,0])), (4, 2, np.array([1,0])), (4, 0, np.array([2,2])), (4, 0, np.array([3,2])), (5, 1, np.array([2,2])), (5, 1, np.array([3,2]))],
        # d = 9.315826 ; d^3 = 808.47 
        n16_nearest_neighbors = [(0, 5, np.array([3,1])), (1, 4, np.array([1,1])), (2, 3, np.array([2,0])), (3, 2, np.array([1,0])), (4, 1, np.array([2,2])), (5, 0, np.array([3,2]))],
        # d = 9.590660 ; d^3 = 882.16 
        n17_nearest_neighbors = [(0, 3, np.array([1,1])), (0, 4, np.array([2,0])), (1, 5, np.array([2,0])), (1, 3, np.array([3,1])), (2, 5, np.array([1,1])), (2, 4, np.array([3,1])), (3, 0, np.array([2,2])), (3, 1, np.array([3,2])), (4, 0, np.array([1,0])), (4, 2, np.array([3,2])), (5, 1, np.array([1,0])), (5, 2, np.array([2,2]))],
        # d = 9.878687 ; d^3 = 964.05 
        n18_nearest_neighbors = [(0, 1, np.array([3,1])), (1, 0, np.array([0,1])), (2, 0, np.array([0,1])), (2, 1, np.array([3,1])), (2, 0, np.array([3,2])), (2, 1, np.array([3,2])), (4, 3, np.array([3,1])), (4, 5, np.array([3,1])), (4, 3, np.array([3,2])), (5, 3, np.array([0,1])), (5, 4, np.array([0,1])), (5, 3, np.array([3,2]))],
        )
        
        kwargs['pairs'] = ruby_pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Ruby(Lattice):
    """A ruby lattice with aspect ratio 1 (Archimedean lattice)
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 6

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 6)  
        #    4---5
        #   /    \
        #  2      3
        #   \    /
        #   0---1 
        pos = np.array([[0, 0], [1, 0], [-0.5, 0.5 * 3**0.5], [1.5, 0.5 * 3**0.5], [0, 3**0.5], [1, 3**0.5]])
        basis = [[1.5 + 0.5 * 3**0.5, 0.5 + 0.5 * 3**0.5], [0, 1+ 3**0.5]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        # Old ones by hand
        # # Edges
        # NN = [(0, 1, np.array([0, 0])), (1, 3, np.array([0, 0])), (3, 5, np.array([0, 0])),
        #       (5, 4, np.array([0, 0])), (4, 2, np.array([0, 0])), (2, 0, np.array([0, 0])),
        #       (4, 0, np.array([0, 1])), (5, 1, np.array([0, 1])),
        #       (3, 0, np.array([1, 0])), (5, 2, np.array([1, 0])),
        #       (1, 2, np.array([1,-1])), (3, 4, np.array([1,-1]))]
        
        # # Square diagonals
        # nNN = [(3, 2, np.array([1,0])), (5, 0, np.array([1,0])),
        #        (4, 1, np.array([0,1])), (5, 0, np.array([0,1])),
        #        (1, 4, np.array([1,-1])), (3, 2, np.array([1,-1]))]
        # # Hexagon chords
        # nnNN = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (3, 4, np.array([0,0])),
        #         (1, 2, np.array([0,0])), (1, 5, np.array([0,0])), (2, 5, np.array([0,0]))]
        # # Triangle-square ``rays''  
        # nnnNN = [(0, 1, np.array([-1,0])), (0, 2, np.array([0,-1])),
        #          (1, 3, np.array([0,-1])), (1, 0, np.array([1,-1])),
        #          (2, 0, np.array([-1,1])), (2, 4, np.array([-1,0])),
        #          (3, 1, np.array([1, 0])), (3, 5, np.array([1,-1])),
        #          (4, 2, np.array([0, 1])), (4, 5, np.array([-1,1])),
        #          (5, 3, np.array([0, 1])), (5, 4, np.array([1, 0]))]
        # # Hexagon diagonals
        # nnnnNN = [(0, 4, np.array([0,0])), (1, 5, np.array([0,0])), (2, 3, np.array([0,0]))]
        
        # d = 1 ; d^6 = 1
        NN = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 3, np.array([0,0])),
             (1, 2, np.array([1,-1])), (2, 4, np.array([0,0])), (3, 5, np.array([0,0])),
             (3, 4, np.array([1,-1])), (3, 0, np.array([1,0])), (4, 5, np.array([0,0])),
             (4, 0, np.array([0,1])), (5, 1, np.array([0,1])), (5, 2, np.array([1,0]))]        
        # d = 1.414 ; d^6 = 8
        nNN = [(1, 4, np.array([1,-1])), (3, 2, np.array([1,-1])), (3, 2, np.array([1,0])),
               (4, 1, np.array([0,1])), (5, 0, np.array([0,1])), (5, 0, np.array([1,0]))]
        # d = 1.74 ; d^6 = 27
        nnNN = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (1, 2, np.array([0,0])),
                (1, 5, np.array([0,0])), (2, 5, np.array([0,0])), (3, 4, np.array([0,0]))]
        # d = 1.93 ; d^6 = 52
        n3NN = [(0, 2, np.array([1,-1])), (1, 0, np.array([1,-1])), (1, 0, np.array([1,0])),
                (2, 0, np.array([0,1])), (3, 1, np.array([0,1])), (3, 5, np.array([1,-1])),
                (3, 1, np.array([1,0])), (4, 2, np.array([0,1])), (4, 2, np.array([1,0])), 
                (5, 3, np.array([0,1])), (5, 4, np.array([1,-1])), (5, 4, np.array([1,0]))]
        # d = 2.0 ; d^6 = 64
        n4NN = [(0, 5, np.array([0,0])), (1, 4, np.array([0,0])), (2, 3, np.array([0,0]))]  
        # d = 2.39 ; d^6 = 188.33459
        n5NN = [(0, 4, np.array([1,-1])), (1, 5, np.array([1,-1])), (1, 2, np.array([1,0])),
                (2, 1, np.array([0,1])), (3, 0, np.array([0,1])), (3, 0, np.array([1,-1])),
                (3, 4, np.array([1,0])), (4, 3, np.array([0,1])), (4, 0, np.array([1,0])),
                (5, 2, np.array([0,1])), (5, 2, np.array([1,-1])), (5, 1, np.array([1,0]))]
        # d = 2.73205081 ; d^6 = 415.846
        n6NN = [(0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 0, np.array([1,0])),
                (1, 1, np.array([0,1])), (1, 4, np.array([1,-2])), (1, 1, np.array([1,-1])), 
                (1, 1, np.array([1,0])), (2, 2, np.array([0,1])), (2, 2, np.array([1,-1])), 
                (2, 2, np.array([1,0])), (3, 3, np.array([0,1])), (3, 3, np.array([1,-1])), 
                (3, 3, np.array([1,0])), (3, 2, np.array([2,-1])), (4, 4, np.array([0,1])), 
                (4, 4, np.array([1,-1])), (4, 4, np.array([1,0])), (5, 5, np.array([0,1])), 
                (5, 5, np.array([1,-1])), (5, 5, np.array([1,0])), (5, 0, np.array([1,1]))]
        # d = 2.90931291 ; d^6 = 606.3768
        n7NN = [(0, 1, np.array([0,1])), (0, 2, np.array([1,0])), (1, 0, np.array([0,1])),
                (1, 3, np.array([1,-1])), (2, 4, np.array([1,-1])), (2, 0, np.array([1,0])), 
                (3, 1, np.array([1,-1])), (3, 5, np.array([1,0])), (4, 5, np.array([0,1])), 
                (4, 2, np.array([1,-1])), (5, 4, np.array([0,1])), (5, 3, np.array([1,0]))]
        # d = 3.34606 ; d^6 = 1403.48056
        n8NN = [(0, 4, np.array([1,-2])), (1, 2, np.array([1,-2])), (1, 5, np.array([1,-2])),
                (1, 2, np.array([2,-1])), (3, 4, np.array([1,-2])), (3, 0, np.array([1,1])), 
                (3, 0, np.array([2,-1])), (3, 4, np.array([2,-1])), (4, 0, np.array([1,1])), 
                (5, 1, np.array([1,1])), (5, 2, np.array([1,1])), (5, 2, np.array([2,-1]))]
        # d = 3.38586 ; d^6 = 1506.676739
        n9NN = [(0, 5, np.array([1,-1])), (1, 4, np.array([1,0])), (2, 3, np.array([0,1])),
                (3, 2, np.array([0,1])), (4, 1, np.array([1,0])), (5, 0, np.array([1,-1]))]

        
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        kwargs['pairs'].setdefault('n3_nearest_neighbors', n3NN)
        kwargs['pairs'].setdefault('n4_nearest_neighbors', n4NN)
        kwargs['pairs'].setdefault('n5_nearest_neighbors', n5NN)
        kwargs['pairs'].setdefault('n6_nearest_neighbors', n6NN)
        kwargs['pairs'].setdefault('n7_nearest_neighbors', n7NN)
        kwargs['pairs'].setdefault('n8_nearest_neighbors', n8NN)
        kwargs['pairs'].setdefault('n9_nearest_neighbors', n9NN)
       
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Kagome2(Kagome):
    """same as :class:`~tenpy.models.lattice.Kagome`, but add more pairs"""
    def __init__(self, Lx, Ly, sites, **kwargs):
        super().__init__(Lx, Ly, sites, **kwargs)
        self.pairs = dict(
            # d= 1. ; d^6 = 1
            NN=[(0, 1, np.array([0, 0])), (0, 2, np.array([0, 0])), (1, 2, np.array([0, 0])),
                (1, 0, np.array([1, 0])), (2, 0, np.array([0, 1])), (2, 1, np.array([-1, 1]))],
            # d= sqrt(3) ; d^6 = 27
            nNN=[(0, 1, np.array([0, -1])), (0, 2, np.array([1, -1])), (1, 0, np.array([1, -1])),
                 (1, 2, np.array([1, 0])), (2, 0, np.array([1, 0])), (2, 1, np.array([0, 1]))],
            # d= 2; d^6 = 64
            nnNN=[(0, 0, np.array([1, -1])), (0, 0, np.array([1, 0])), (0, 0, np.array([0, 1])),
                  (1, 1, np.array([1, -1])), (1, 1, np.array([1, 0])), (1, 1, np.array([0, 1])),
                  (2, 2, np.array([1, -1])), (2, 2, np.array([1, 0])), (2, 2, np.array([0, 1]))],
            # Longer range interactions
            # d = 2.645751 ; d^6 = 343.00
            n3NN=[(0, 1, np.array([0, 1])), (0, 2, np.array([1, -2])), (0, 1, np.array([1, -1])),
                  (0, 2, np.array([1, 0])), (1, 2, np.array([0, 1])), (1, 2, np.array([1, -2])),
                  (1, 0, np.array([1, 1])), (1, 0, np.array([2, -1])), (1, 2, np.array([2, -1])),
                  (2, 0, np.array([1, -1])), (2, 1, np.array([1, 0])), (2, 0, np.array([1, 1]))],
            # d = 3.000000 ; d^6 = 729.00
            n4NN=[(0, 2, np.array([0, 1])), (0, 1, np.array([1, 0])), (1, 2, np.array([2, -2])),
                  (1, 0, np.array([2, 0])), (2, 0, np.array([0, 2])), (2, 1, np.array([1, -1]))],
            # d = 3.464102 ; d^6 = 1728.00
            n5NN=[(0, 0, np.array([1, -2])), (0, 0, np.array([1, 1])), (0, 0, np.array([2, -1])),
                  (1, 1, np.array([1, -2])), (1, 1, np.array([1, 1])), (1, 1, np.array([2, -1])),
                  (2, 2, np.array([1, -2])), (2, 2, np.array([1, 1])), (2, 2, np.array([2, -1]))],
            # d = 3.605551 ; d^6 = 2197.00
            n6NN=[(0, 1, np.array([1, -2])), (0, 2, np.array([2, -2])), (0, 2, np.array([2, -1])),
                  (1, 0, np.array([0, 2])), (1, 0, np.array([1, -2])), (1, 2, np.array([1, 1])),
                  (1, 0, np.array([2, -2])), (1, 2, np.array([2, 0])), (2, 1, np.array([0, 2])),
                  (2, 1, np.array([1, 1])), (2, 0, np.array([2, -1])), (2, 0, np.array([2, 0]))]
        )



class Kagome3(Kagome):
    """same as :class:`~tenpy.models.lattice.Kagome`, but even more pairs (up to YC12)"""
    def __init__(self, Lx, Ly, sites, **kwargs):
        super().__init__(Lx, Ly, sites, **kwargs)
        self.pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 2, np.array([0,0])), (1, 2, np.array([1,-1])), (1, 0, np.array([1,0])), (2, 0, np.array([0,1]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 2, np.array([1,-1])), (1, 0, np.array([0,1])), (1, 0, np.array([1,-1])), (1, 2, np.array([1,0])), (2, 1, np.array([0,1])), (2, 0, np.array([1,0]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 0, np.array([1,0])), (1, 1, np.array([0,1])), (1, 1, np.array([1,-1])), (1, 1, np.array([1,0])), (2, 2, np.array([0,1])), (2, 2, np.array([1,-1])), (2, 2, np.array([1,0]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 1, np.array([0,1])), (0, 2, np.array([1,-2])), (0, 1, np.array([1,-1])), (0, 2, np.array([1,0])), (1, 2, np.array([0,1])), (1, 2, np.array([1,-2])), (1, 0, np.array([1,1])), (1, 0, np.array([2,-1])), (1, 2, np.array([2,-1])), (2, 0, np.array([1,-1])), (2, 1, np.array([1,0])), (2, 0, np.array([1,1]))],
            # d = 3.000000 ; d^3 = 27.00 
            n4_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 1, np.array([1,0])), (1, 2, np.array([2,-2])), (1, 0, np.array([2,0])), (2, 0, np.array([0,2])), (2, 1, np.array([1,-1]))],
            # d = 3.464102 ; d^3 = 41.57 
            n5_nearest_neighbors = [(0, 0, np.array([1,-2])), (0, 0, np.array([1,1])), (0, 0, np.array([2,-1])), (1, 1, np.array([1,-2])), (1, 1, np.array([1,1])), (1, 1, np.array([2,-1])), (2, 2, np.array([1,-2])), (2, 2, np.array([1,1])), (2, 2, np.array([2,-1]))],
            # d = 3.605551 ; d^3 = 46.87 
            n6_nearest_neighbors = [(0, 1, np.array([1,-2])), (0, 2, np.array([2,-2])), (0, 2, np.array([2,-1])), (1, 0, np.array([0,2])), (1, 0, np.array([1,-2])), (1, 2, np.array([1,1])), (1, 0, np.array([2,-2])), (1, 2, np.array([2,0])), (2, 1, np.array([0,2])), (2, 1, np.array([1,1])), (2, 0, np.array([2,-1])), (2, 0, np.array([2,0]))],
            # d = 4.000000 ; d^3 = 64.00 
            n7_nearest_neighbors = [(0, 0, np.array([0,2])), (0, 0, np.array([2,-2])), (0, 0, np.array([2,0])), (1, 1, np.array([0,2])), (1, 1, np.array([2,-2])), (1, 1, np.array([2,0])), (2, 2, np.array([0,2])), (2, 2, np.array([2,-2])), (2, 2, np.array([2,0]))],
            # d = 4.358899 ; d^3 = 82.82 
            n8_nearest_neighbors = [(0, 2, np.array([1,-3])), (0, 1, np.array([1,1])), (0, 2, np.array([1,1])), (0, 1, np.array([2,-1])), (1, 2, np.array([2,-3])), (1, 0, np.array([2,1])), (1, 2, np.array([3,-2])), (1, 0, np.array([3,-1])), (2, 0, np.array([1,-2])), (2, 1, np.array([1,-2])), (2, 0, np.array([1,2])), (2, 1, np.array([2,-1]))],
            # d = 4.582576 ; d^3 = 96.23 
            n9_nearest_neighbors = [(0, 1, np.array([0,2])), (0, 2, np.array([2,-3])), (0, 1, np.array([2,-2])), (0, 2, np.array([2,0])), (1, 2, np.array([0,2])), (1, 2, np.array([1,-3])), (1, 0, np.array([1,2])), (1, 0, np.array([3,-2])), (1, 2, np.array([3,-1])), (2, 0, np.array([2,-2])), (2, 1, np.array([2,0])), (2, 0, np.array([2,1]))],
            # d = 5.000000 ; d^3 = 125.00 
            n10_nearest_neighbors = [(0, 2, np.array([0,2])), (0, 1, np.array([2,0])), (1, 2, np.array([3,-3])), (1, 0, np.array([3,0])), (2, 0, np.array([0,3])), (2, 1, np.array([2,-2]))],
            # d = 5.196152 ; d^3 = 140.30 
            n11_nearest_neighbors = [(0, 1, np.array([1,-3])), (0, 2, np.array([3,-2])), (1, 0, np.array([2,-3])), (1, 2, np.array([2,1])), (2, 1, np.array([1,2])), (2, 0, np.array([3,-1]))],
            # d = 5.291503 ; d^3 = 148.16 
            n12_nearest_neighbors = [(0, 0, np.array([1,-3])), (0, 0, np.array([1,2])), (0, 0, np.array([2,-3])), (0, 0, np.array([2,1])), (0, 0, np.array([3,-2])), (0, 0, np.array([3,-1])), (1, 1, np.array([1,-3])), (1, 1, np.array([1,2])), (1, 1, np.array([2,-3])), (1, 1, np.array([2,1])), (1, 1, np.array([3,-2])), (1, 1, np.array([3,-1])), (2, 2, np.array([1,-3])), (2, 2, np.array([1,2])), (2, 2, np.array([2,-3])), (2, 2, np.array([2,1])), (2, 2, np.array([3,-2])), (2, 2, np.array([3,-1]))],
            # d = 5.567764 ; d^3 = 172.60 
            n13_nearest_neighbors = [(0, 1, np.array([2,-3])), (0, 2, np.array([3,-3])), (0, 2, np.array([3,-1])), (1, 0, np.array([0,3])), (1, 0, np.array([1,-3])), (1, 2, np.array([1,2])), (1, 0, np.array([3,-3])), (1, 2, np.array([3,0])), (2, 1, np.array([0,3])), (2, 1, np.array([2,1])), (2, 0, np.array([3,-2])), (2, 0, np.array([3,0]))],
            # d = 6.000000 ; d^3 = 216.00 
            n14_nearest_neighbors = [(0, 0, np.array([0,3])), (0, 0, np.array([3,-3])), (0, 0, np.array([3,0])), (1, 1, np.array([0,3])), (1, 1, np.array([3,-3])), (1, 1, np.array([3,0])), (2, 2, np.array([0,3])), (2, 2, np.array([3,-3])), (2, 2, np.array([3,0]))]
        )


class KagomeYC(Lattice):
    """A Kagome lattice with YC orientation and many long-range couplings.
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2  #: the dimension of the lattice
    Lu = 3  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 3)
        #   \   /
        #    \ /
        #     2
        #    / \
        #   /   \
        #  0-----1-----
        pos = np.array([[0, 0], [1, 0], [0.5, 0.5 * 3**0.5]])
        basis = [2 * pos[1], 2 * pos[2]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        # NN = [(0, 1, np.array([0, 0])), (0, 2, np.array([0, 0])), (1, 2, np.array([0, 0])),
        #       (1, 0, np.array([1, 0])), (2, 0, np.array([0, 1])), (2, 1, np.array([-1, 1]))]
        # nNN = [(0, 1, np.array([0, -1])), (0, 2, np.array([1, -1])), (1, 0, np.array([1, -1])),
        #        (1, 2, np.array([1, 0])), (2, 0, np.array([1, 0])), (2, 1, np.array([0, 1]))]
        # nnNN = [(0, 0, np.array([1, -1])), (0, 0, np.array([0, 1])), (0, 0, np.array([1, 0])),
        #         (1, 1, np.array([1, -1])), (1, 1, np.array([0, 1])), (1, 1, np.array([1, 0])),
        #         (2, 2, np.array([1, -1])), (2, 2, np.array([0, 1])), (2, 2, np.array([1, 0]))]
        # kwargs.setdefault('pairs', {})
        # kwargs['pairs'].setdefault('nearest_neighbors', NN)
        # kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        # kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        kagome_pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 2, np.array([0,0])), (1, 2, np.array([1,-1])), (1, 0, np.array([1,0])), (2, 0, np.array([0,1]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 2, np.array([1,-1])), (1, 0, np.array([0,1])), (1, 0, np.array([1,-1])), (1, 2, np.array([1,0])), (2, 1, np.array([0,1])), (2, 0, np.array([1,0]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 0, np.array([1,0])), (1, 1, np.array([0,1])), (1, 1, np.array([1,-1])), (1, 1, np.array([1,0])), (2, 2, np.array([0,1])), (2, 2, np.array([1,-1])), (2, 2, np.array([1,0]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 1, np.array([0,1])), (0, 2, np.array([1,-2])), (0, 1, np.array([1,-1])), (0, 2, np.array([1,0])), (1, 2, np.array([0,1])), (1, 2, np.array([1,-2])), (1, 0, np.array([1,1])), (1, 0, np.array([2,-1])), (1, 2, np.array([2,-1])), (2, 0, np.array([1,-1])), (2, 1, np.array([1,0])), (2, 0, np.array([1,1]))],
            # d = 3.000000 ; d^3 = 27.00 
            n4_nearest_neighbors = [(0, 2, np.array([0,1])), (0, 1, np.array([1,0])), (1, 2, np.array([2,-2])), (1, 0, np.array([2,0])), (2, 0, np.array([0,2])), (2, 1, np.array([1,-1]))],
            # d = 3.464102 ; d^3 = 41.57 
            n5_nearest_neighbors = [(0, 0, np.array([1,-2])), (0, 0, np.array([1,1])), (0, 0, np.array([2,-1])), (1, 1, np.array([1,-2])), (1, 1, np.array([1,1])), (1, 1, np.array([2,-1])), (2, 2, np.array([1,-2])), (2, 2, np.array([1,1])), (2, 2, np.array([2,-1]))],
            # d = 3.605551 ; d^3 = 46.87 
            n6_nearest_neighbors = [(0, 1, np.array([1,-2])), (0, 2, np.array([2,-2])), (0, 2, np.array([2,-1])), (1, 0, np.array([0,2])), (1, 0, np.array([1,-2])), (1, 2, np.array([1,1])), (1, 0, np.array([2,-2])), (1, 2, np.array([2,0])), (2, 1, np.array([0,2])), (2, 1, np.array([1,1])), (2, 0, np.array([2,-1])), (2, 0, np.array([2,0]))],
            # d = 4.000000 ; d^3 = 64.00 
            n7_nearest_neighbors = [(0, 0, np.array([0,2])), (0, 0, np.array([2,-2])), (0, 0, np.array([2,0])), (1, 1, np.array([0,2])), (1, 1, np.array([2,-2])), (1, 1, np.array([2,0])), (2, 2, np.array([0,2])), (2, 2, np.array([2,-2])), (2, 2, np.array([2,0]))],
            # d = 4.358899 ; d^3 = 82.82 
            n8_nearest_neighbors = [(0, 2, np.array([1,-3])), (0, 1, np.array([1,1])), (0, 2, np.array([1,1])), (0, 1, np.array([2,-1])), (1, 2, np.array([2,-3])), (1, 0, np.array([2,1])), (1, 2, np.array([3,-2])), (1, 0, np.array([3,-1])), (2, 0, np.array([1,-2])), (2, 1, np.array([1,-2])), (2, 0, np.array([1,2])), (2, 1, np.array([2,-1]))],
            # d = 4.582576 ; d^3 = 96.23 
            n9_nearest_neighbors = [(0, 1, np.array([0,2])), (0, 2, np.array([2,-3])), (0, 1, np.array([2,-2])), (0, 2, np.array([2,0])), (1, 2, np.array([0,2])), (1, 2, np.array([1,-3])), (1, 0, np.array([1,2])), (1, 0, np.array([3,-2])), (1, 2, np.array([3,-1])), (2, 0, np.array([2,-2])), (2, 1, np.array([2,0])), (2, 0, np.array([2,1]))],
            # d = 5.000000 ; d^3 = 125.00 
            n10_nearest_neighbors = [(0, 2, np.array([0,2])), (0, 1, np.array([2,0])), (1, 2, np.array([3,-3])), (1, 0, np.array([3,0])), (2, 0, np.array([0,3])), (2, 1, np.array([2,-2]))],
            # d = 5.196152 ; d^3 = 140.30 
            n11_nearest_neighbors = [(0, 1, np.array([1,-3])), (0, 2, np.array([3,-2])), (1, 0, np.array([2,-3])), (1, 2, np.array([2,1])), (2, 1, np.array([1,2])), (2, 0, np.array([3,-1]))],
            # d = 5.291503 ; d^3 = 148.16 
            n12_nearest_neighbors = [(0, 0, np.array([1,-3])), (0, 0, np.array([1,2])), (0, 0, np.array([2,-3])), (0, 0, np.array([2,1])), (0, 0, np.array([3,-2])), (0, 0, np.array([3,-1])), (1, 1, np.array([1,-3])), (1, 1, np.array([1,2])), (1, 1, np.array([2,-3])), (1, 1, np.array([2,1])), (1, 1, np.array([3,-2])), (1, 1, np.array([3,-1])), (2, 2, np.array([1,-3])), (2, 2, np.array([1,2])), (2, 2, np.array([2,-3])), (2, 2, np.array([2,1])), (2, 2, np.array([3,-2])), (2, 2, np.array([3,-1]))],
            # d = 5.567764 ; d^3 = 172.60 
            n13_nearest_neighbors = [(0, 1, np.array([2,-3])), (0, 2, np.array([3,-3])), (0, 2, np.array([3,-1])), (1, 0, np.array([0,3])), (1, 0, np.array([1,-3])), (1, 2, np.array([1,2])), (1, 0, np.array([3,-3])), (1, 2, np.array([3,0])), (2, 1, np.array([0,3])), (2, 1, np.array([2,1])), (2, 0, np.array([3,-2])), (2, 0, np.array([3,0]))],
            # d = 6.000000 ; d^3 = 216.00 
            n14_nearest_neighbors = [(0, 0, np.array([0,3])), (0, 0, np.array([3,-3])), (0, 0, np.array([3,0])), (1, 1, np.array([0,3])), (1, 1, np.array([3,-3])), (1, 1, np.array([3,0])), (2, 2, np.array([0,3])), (2, 2, np.array([3,-3])), (2, 2, np.array([3,0]))]
            )
        kwargs['pairs'] = kagome_pairs

        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class SkewKagomeVertices(Lattice):
    """NOTE: Auxilliary for "SkewKagome" only. Contains just the sites/vertices, for finding the pairs.
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2  #: the dimension of the lattice
    Lu = 3  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.

    def __init__(self, Lx, Ly, sites, alphapi=0.1, **kwargs):
        sites = _parse_sites(sites, 3)
        #   \   /
        #    \ /
        #     2
        #    / \
        #   /   \
        #  0-----1-----
        pos = np.array([[0, 0], [np.cos(alphapi*np.pi), np.sin(alphapi*np.pi)], [np.cos(alphapi*np.pi + np.pi/3), np.sin(alphapi*np.pi + np.pi/3)]])
        basis = np.cos(alphapi*np.pi)*np.array([[2,0],[1, np.sqrt(3)]])  # With this choice, all triangles are equilateral
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class SkewKagome(Lattice):
    """An kagome lattice with unit cells twisted by alphapi*pi. Becomes triangular at alphapi=1/6.
    The unit cell spacing also varies to keep it isostatic.
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2  #: the dimension of the lattice
    Lu = 3  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.

    def __init__(self, Lx, Ly, sites, alphapi=0.1, **kwargs):
        sites = _parse_sites(sites, 3)
        #   \   /
        #    \ /
        #     2
        #    / \
        #   /   \
        #  0-----1-----
        pos = np.array([[0, 0], [np.cos(alphapi*np.pi), np.sin(alphapi*np.pi)], [np.cos(alphapi*np.pi + np.pi/3), np.sin(alphapi*np.pi + np.pi/3)]])
        basis = np.cos(alphapi*np.pi)*np.array([[2,0],[1, np.sqrt(3)]])  # With this choice, all triangles are equilateral
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        
        
        ### Construct the pairs using the SkewKagomeVertices class as an auxilliary lattice
        # Create a large lattice patch - size unimportant, just need to call lat.position()
        cutoff = 6
        aux_lat = SkewKagomeVertices(Lx=cutoff, Ly=cutoff, sites=None, alphapi=alphapi, bc=['open', 'open'])
        # Use the tenpy method to get them
        aux_dict = aux_lat.find_coupling_pairs(cutoff)
        
        # The aux dictionary keys = coupling distance; we change to "nearest_neighbor" style
        new_dict = {}
        nNN=0
        for old_key in aux_dict.keys():
            new_key = 'nearest_neighbors' if nNN==0 else 'n%d_nearest_neighbors' %nNN
            new_dict[new_key]=aux_dict[old_key]
            nNN+=1
        
        kwargs['pairs'] = new_dict

        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Square2(Square):
    """same as :class:`~tenpy.models.lattice.Square`, but add more pairs"""
    def __init__(self, Lx, Ly, sites, **kwargs):
        super().__init__(Lx, Ly, sites, **kwargs)
        self.pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 0, np.array([0,1])), (0, 0, np.array([1,0]))],
            # d = 1.414214 ; d^3 = 2.83 
            n1_nearest_neighbors = [(0, 0, np.array([1,-1])), (0, 0, np.array([1,1]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 0, np.array([0,2])), (0, 0, np.array([2,0]))],
            # d = 2.236068 ; d^3 = 11.18 
            n3_nearest_neighbors = [(0, 0, np.array([1,-2])), (0, 0, np.array([1,2])), (0, 0, np.array([2,-1])), (0, 0, np.array([2,1]))],
            # d = 2.828427 ; d^3 = 22.63 
            n4_nearest_neighbors = [(0, 0, np.array([2,-2])), (0, 0, np.array([2,2]))],
            # d = 3.000000 ; d^3 = 27.00 
            n5_nearest_neighbors = [(0, 0, np.array([0,3])), (0, 0, np.array([3,0]))],
            # d = 3.162278 ; d^3 = 31.62 
            n6_nearest_neighbors = [(0, 0, np.array([1,-3])), (0, 0, np.array([1,3])), (0, 0, np.array([3,-1])), (0, 0, np.array([3,1]))],
            # d = 3.605551 ; d^3 = 46.87 
            n7_nearest_neighbors = [(0, 0, np.array([2,-3])), (0, 0, np.array([2,3])), (0, 0, np.array([3,-2])), (0, 0, np.array([3,2]))],
            # d = 4.000000 ; d^3 = 64.00 
            n8_nearest_neighbors = [(0, 0, np.array([0,4])), (0, 0, np.array([4,0]))],
            # d = 4.123106 ; d^3 = 70.09 
            n9_nearest_neighbors = [(0, 0, np.array([1,-4])), (0, 0, np.array([1,4])), (0, 0, np.array([4,-1])), (0, 0, np.array([4,1]))],
            # d = 4.242641 ; d^3 = 76.37 
            n10_nearest_neighbors = [(0, 0, np.array([3,-3])), (0, 0, np.array([3,3]))],
            # d = 4.472136 ; d^3 = 89.44 
            n11_nearest_neighbors = [(0, 0, np.array([2,-4])), (0, 0, np.array([2,4])), (0, 0, np.array([4,-2])), (0, 0, np.array([4,2]))],
            # d = 5.000000 ; d^3 = 125.00 
            n12_nearest_neighbors = [(0, 0, np.array([0,5])), (0, 0, np.array([3,-4])), (0, 0, np.array([3,4])), (0, 0, np.array([4,-3])), (0, 0, np.array([4,3])), (0, 0, np.array([5,0]))],
            # d = 5.099020 ; d^3 = 132.57 
            n13_nearest_neighbors = [(0, 0, np.array([1,-5])), (0, 0, np.array([1,5])), (0, 0, np.array([5,-1])), (0, 0, np.array([5,1]))],
            # d = 5.385165 ; d^3 = 156.17 
            n14_nearest_neighbors = [(0, 0, np.array([2,-5])), (0, 0, np.array([2,5])), (0, 0, np.array([5,-2])), (0, 0, np.array([5,2]))],
            # d = 5.656854 ; d^3 = 181.02 
            n15_nearest_neighbors = [(0, 0, np.array([4,-4])), (0, 0, np.array([4,4]))],
            # d = 5.830952 ; d^3 = 198.25 
            n16_nearest_neighbors = [(0, 0, np.array([3,-5])), (0, 0, np.array([3,5])), (0, 0, np.array([5,-3])), (0, 0, np.array([5,3]))],
            # d = 6.000000 ; d^3 = 216.00 
            n17_nearest_neighbors = [(0, 0, np.array([0,6])), (0, 0, np.array([6,0]))],
            # d = 6.082763 ; d^3 = 225.06 
            n18_nearest_neighbors = [(0, 0, np.array([1,-6])), (0, 0, np.array([1,6])), (0, 0, np.array([6,-1])), (0, 0, np.array([6,1]))]
        )


class Triangular2(Triangular):
    """same as :class:`~tenpy.models.lattice.Triangular`, but add more pairs"""
    def __init__(self, Lx, Ly, sites, **kwargs):
        super().__init__(Lx, Ly, sites, **kwargs)
        self.pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 0, np.array([0,1])), (0, 0, np.array([1,-1])), (0, 0, np.array([1,0]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 0, np.array([1,-2])), (0, 0, np.array([1,1])), (0, 0, np.array([2,-1]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 0, np.array([0,2])), (0, 0, np.array([2,-2])), (0, 0, np.array([2,0]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 0, np.array([1,-3])), (0, 0, np.array([1,2])), (0, 0, np.array([2,-3])), (0, 0, np.array([2,1])), (0, 0, np.array([3,-2])), (0, 0, np.array([3,-1]))],
            # d = 3.000000 ; d^3 = 27.00 
            n4_nearest_neighbors = [(0, 0, np.array([0,3])), (0, 0, np.array([3,-3])), (0, 0, np.array([3,0]))],
            # d = 3.464102 ; d^3 = 41.57 
            n5_nearest_neighbors = [(0, 0, np.array([2,-4])), (0, 0, np.array([2,2])), (0, 0, np.array([4,-2]))],
            # d = 3.605551 ; d^3 = 46.87 
            n6_nearest_neighbors = [(0, 0, np.array([1,-4])), (0, 0, np.array([1,3])), (0, 0, np.array([3,-4])), (0, 0, np.array([3,1])), (0, 0, np.array([4,-3])), (0, 0, np.array([4,-1]))],
            # d = 4.000000 ; d^3 = 64.00 
            n7_nearest_neighbors = [(0, 0, np.array([0,4])), (0, 0, np.array([4,-4])), (0, 0, np.array([4,0]))],
            # d = 4.358899 ; d^3 = 82.82 
            n8_nearest_neighbors = [(0, 0, np.array([2,-5])), (0, 0, np.array([2,3])), (0, 0, np.array([3,-5])), (0, 0, np.array([3,2])), (0, 0, np.array([5,-3])), (0, 0, np.array([5,-2]))],
            # d = 4.582576 ; d^3 = 96.23 
            n9_nearest_neighbors = [(0, 0, np.array([1,-5])), (0, 0, np.array([1,4])), (0, 0, np.array([4,-5])), (0, 0, np.array([4,1])), (0, 0, np.array([5,-4])), (0, 0, np.array([5,-1]))],
            # d = 5.000000 ; d^3 = 125.00 
            n10_nearest_neighbors = [(0, 0, np.array([0,5])), (0, 0, np.array([5,-5])), (0, 0, np.array([5,0]))],
            # d = 5.196152 ; d^3 = 140.30 
            n11_nearest_neighbors = [(0, 0, np.array([3,-6])), (0, 0, np.array([3,3])), (0, 0, np.array([6,-3]))],
            # d = 5.291503 ; d^3 = 148.16 
            n12_nearest_neighbors = [(0, 0, np.array([2,-6])), (0, 0, np.array([2,4])), (0, 0, np.array([4,-6])), (0, 0, np.array([4,2])), (0, 0, np.array([6,-4])), (0, 0, np.array([6,-2]))],
            # d = 5.567764 ; d^3 = 172.60 
            n13_nearest_neighbors = [(0, 0, np.array([1,-6])), (0, 0, np.array([1,5])), (0, 0, np.array([5,-6])), (0, 0, np.array([5,1])), (0, 0, np.array([6,-5])), (0, 0, np.array([6,-1]))],
            # d = 6.000000 ; d^3 = 216.00 
            n14_nearest_neighbors = [(0, 0, np.array([0,6])), (0, 0, np.array([6,-6])), (0, 0, np.array([6,0]))],
            # d = 6.082763 ; d^3 = 225.06 
            n15_nearest_neighbors = [(0, 0, np.array([3,-7])), (0, 0, np.array([3,4])), (0, 0, np.array([4,-7])), (0, 0, np.array([4,3])), (0, 0, np.array([7,-4])), (0, 0, np.array([7,-3]))],
            # d = 6.244998 ; d^3 = 243.55 
            n16_nearest_neighbors = [(0, 0, np.array([2,-7])), (0, 0, np.array([2,5])), (0, 0, np.array([5,-7])), (0, 0, np.array([5,2])), (0, 0, np.array([7,-5])), (0, 0, np.array([7,-2]))],
            # d = 6.557439 ; d^3 = 281.97 
            n17_nearest_neighbors = [(0, 0, np.array([1,-7])), (0, 0, np.array([1,6])), (0, 0, np.array([6,-7])), (0, 0, np.array([6,1])), (0, 0, np.array([7,-6])), (0, 0, np.array([7,-1]))],
            # d = 6.928203 ; d^3 = 332.55 
            n18_nearest_neighbors = [(0, 0, np.array([4,4]))]
        )


class RubyXC_12(Lattice):
    """A ruby lattice with aspect ratio rho=sqrt(3)  (kagome links), 12 sites/cell
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 12

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 12)
        #            10---11
        #      /----/ \   /
        #     5        9
        #    / \  /---/
        #   3---4
        #   |   |
        #   |   |
        #   1---2
        #   \   / \----\
        #     0         8
        #      \-----\ / \
        #             6---7
        
        pos = np.array([[0, 0],  [-0.5, 0.5 * 3**0.5],  [0.5, 0.5 * 3**0.5],
                       [-0.5, 1.5 * 3**0.5], [0.5, 1.5 * 3**0.5], [0, 2* 3**0.5],
                       [1.5, -0.5 * 3**0.5], [2.5, -0.5 * 3**0.5], [2, 0],
                       [2, 2 * 3**0.5], [1.5, 2.5 * 3**0.5], [2.5, 2.5 * 3**0.5]])
        
        basis = [[4, 0], [0, 4 * 3**0.5]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        
        ruby_pairs = dict(
            # d = 1.000000 ; d^6 = 1.00 
            n0NN = [(0, 1, np.array([0,0])), (0, 2, np.array([0,0])), (1, 2, np.array([0,0])),
                    (3, 4, np.array([0,0])), (3, 5, np.array([0,0])), (4, 5, np.array([0,0])),
                    (6, 7, np.array([0,0])), (6, 8, np.array([0,0])), (7, 8, np.array([0,0])),
                    (9, 10, np.array([0,0])), (9, 11, np.array([0,0])), (10, 11, np.array([0,0]))],
            # d = 1.732051 ; d^6 = 27.00 
            n1NN = [(0, 6, np.array([0,0])), (1, 3, np.array([0,0])), (2, 4, np.array([0,0])),
                    (2, 8, np.array([0,0])), (4, 9, np.array([0,0])), (5, 10, np.array([0,0])),
                    (7, 0, np.array([1,0])), (8, 1, np.array([1,0])), (9, 3, np.array([1,0])),
                    (10, 6, np.array([0,1])), (11, 7, np.array([0,1])), (11, 5, np.array([1,0]))],
            # d = 2.000000 ; d^6 = 64.00 
            n2NN = [(0, 8, np.array([0,0])), (1, 4, np.array([0,0])), (2, 3, np.array([0,0])),
                    (2, 6, np.array([0,0])), (4, 10, np.array([0,0])), (5, 9, np.array([0,0])),
                    (7, 1, np.array([1,0])), (8, 0, np.array([1,0])), (9, 5, np.array([1,0])),
                    (10, 7, np.array([0,1])), (11, 6, np.array([0,1])), (11, 3, np.array([1,0]))],
            # d = 2.645751 ; d^6 = 343.00 
            n3NN = [(0, 3, np.array([0,0])), (0, 4, np.array([0,0])), (0, 7, np.array([0,0])),
                    (1, 5, np.array([0,0])), (1, 6, np.array([0,0])), (1, 8, np.array([0,0])),
                    (2, 5, np.array([0,0])), (2, 7, np.array([0,0])), (3, 9, np.array([0,0])),
                    (3, 10, np.array([0,0])), (4, 11, np.array([0,0])), (5, 11, np.array([0,0])),
                    (6, 0, np.array([1,0])), (6, 1, np.array([1,0])), (7, 2, np.array([1,0])),
                    (8, 2, np.array([1,0])), (9, 6, np.array([0,1])), (9, 7, np.array([0,1])),
                    (9, 4, np.array([1,0])), (10, 8, np.array([0,1])), (10, 3, np.array([1,0])),
                    (10, 5, np.array([1,0])), (11, 8, np.array([0,1])), (11, 4, np.array([1,0]))],
            # d = 3.000000 ; d^6 = 729.00 
            n4NN = [(2, 9, np.array([0,0])), (2, 1, np.array([1,0])), (4, 8, np.array([0,0])),
                    (4, 3, np.array([1,0])), (5, 6, np.array([0,1])), (7, 5, np.array([1,-1])),
                    (7, 6, np.array([1,0])), (8, 3, np.array([1,0])), (9, 1, np.array([1,0])),
                    (10, 0, np.array([0,1])), (11, 10, np.array([1,0])), (11, 0, np.array([1,1]))],
            # d = 3.464102 ; d^6 = 1728.00 
            n5NN = [(0, 5, np.array([0,0])), (1, 7, np.array([0,0])), (2, 3, np.array([1,0])),
                    (3, 11, np.array([0,0])), (4, 1, np.array([1,0])), (5, 0, np.array([0,1])),
                    (6, 2, np.array([1,0])), (7, 10, np.array([1,-1])), (8, 9, np.array([0,0])),
                    (9, 8, np.array([0,1])), (10, 4, np.array([1,0])), (11, 6, np.array([1,1]))],
            # d = 3.605551 ; d^6 = 2197.00 
            n6NN = [(0, 1, np.array([1,0])), (1, 9, np.array([0,0])), (2, 10, np.array([0,0])),
                    (2, 0, np.array([1,0])), (3, 8, np.array([0,0])), (4, 6, np.array([0,0])),
                    (4, 6, np.array([0,1])), (4, 5, np.array([1,0])), (5, 7, np.array([0,1])),
                    (5, 3, np.array([1,0])), (6, 5, np.array([1,-1])), (7, 3, np.array([1,-1])),
                    (7, 3, np.array([1,0])), (7, 8, np.array([1,0])), (8, 4, np.array([1,0])),
                    (8, 6, np.array([1,0])), (9, 2, np.array([1,0])), (9, 10, np.array([1,0])),
                    (10, 2, np.array([0,1])), (10, 0, np.array([1,1])), (11, 0, np.array([0,1])),
                    (11, 1, np.array([1,0])), (11, 9, np.array([1,0])), (11, 1, np.array([1,1]))],
            # d = 4.000000 ; d^6 = 4096.00 
            n7NN = [(0, 9, np.array([0,0])), (0, 0, np.array([1,0])), (1, 10, np.array([0,0])),
                    (1, 1, np.array([1,0])), (2, 11, np.array([0,0])), (2, 2, np.array([1,0])),
                    (3, 6, np.array([0,0])), (3, 6, np.array([0,1])), (3, 3, np.array([1,0])),
                    (4, 7, np.array([0,0])), (4, 7, np.array([0,1])), (4, 4, np.array([1,0])),
                    (5, 8, np.array([0,0])), (5, 8, np.array([0,1])), (5, 5, np.array([1,0])),
                    (6, 3, np.array([1,-1])), (6, 3, np.array([1,0])), (6, 6, np.array([1,0])),
                    (7, 4, np.array([1,-1])), (7, 4, np.array([1,0])), (7, 7, np.array([1,0])),
                    (8, 5, np.array([1,-1])), (8, 5, np.array([1,0])), (8, 8, np.array([1,0])),
                    (9, 0, np.array([0,1])), (9, 0, np.array([1,0])), (9, 9, np.array([1,0])),
                    (9, 0, np.array([1,1])), (10, 1, np.array([0,1])), (10, 1, np.array([1,0])),
                    (10, 10, np.array([1,0])), (10, 1, np.array([1,1])), (11, 2, np.array([0,1])),
                    (11, 2, np.array([1,0])), (11, 11, np.array([1,0])), (11, 2, np.array([1,1]))],
            # d = 4.358899 ; d^6 = 6859.00 
            n8NN = [(0, 3, np.array([1,0])), (1, 3, np.array([1,0])), (2, 4, np.array([1,0])),
                    (2, 5, np.array([1,0])), (3, 0, np.array([0,1])), (3, 1, np.array([1,0])),
                    (4, 0, np.array([0,1])), (4, 0, np.array([1,0])), (4, 2, np.array([1,0])),
                    (5, 1, np.array([0,1])), (5, 2, np.array([0,1])), (5, 1, np.array([1,0])),
                    (6, 9, np.array([0,0])), (6, 10, np.array([1,-1])), (7, 9, np.array([0,0])),
                    (7, 9, np.array([1,-1])), (7, 11, np.array([1,-1])), (8, 10, np.array([0,0])),
                    (8, 11, np.array([0,0])), (8, 10, np.array([1,-1])), (9, 6, np.array([1,1])),
                    (10, 6, np.array([1,1])), (11, 7, np.array([1,1])), (11, 8, np.array([1,1]))],
            # d = 4.582576 ; d^6 = 9261.00 
            n9NN = [(0, 10, np.array([0,0])), (0, 2, np.array([1,0])), (1, 11, np.array([0,0])),
                    (1, 0, np.array([1,0])), (3, 7, np.array([0,0])), (3, 7, np.array([0,1])),
                    (3, 5, np.array([1,0])), (4, 8, np.array([0,1])), (5, 6, np.array([0,0])),
                    (5, 4, np.array([1,0])), (6, 4, np.array([1,-1])), (6, 4, np.array([1,0])),
                    (6, 8, np.array([1,0])), (7, 5, np.array([1,0])), (8, 3, np.array([1,-1])),
                    (8, 7, np.array([1,0])), (9, 2, np.array([0,1])), (9, 11, np.array([1,0])),
                    (9, 1, np.array([1,1])), (10, 2, np.array([1,0])), (10, 9, np.array([1,0])),
                    (10, 2, np.array([1,1])), (11, 1, np.array([0,1])), (11, 0, np.array([1,0]))],
            # d = 5.000000 ; d^6 = 15625.00 
            n10NN = [(0, 11, np.array([0,0])), (1, 2, np.array([1,0])), (3, 8, np.array([0,1])),
                    (3, 4, np.array([1,0])), (5, 7, np.array([0,0])), (6, 5, np.array([1,0])),
                    (6, 7, np.array([1,0])), (8, 4, np.array([1,-1])), (9, 1, np.array([0,1])),
                    (9, 2, np.array([1,1])), (10, 0, np.array([1,0])), (10, 11, np.array([1,0]))],
            # d = 5.196152 ; d^6 = 19683.00 
            n11NN = [(0, 4, np.array([1,0])), (1, 5, np.array([1,0])), (3, 1, np.array([0,1])),
                    (3, 0, np.array([1,0])), (4, 2, np.array([0,1])), (5, 2, np.array([1,0])),
                    (6, 10, np.array([0,0])), (6, 9, np.array([1,-1])), (7, 11, np.array([0,0])),
                    (8, 11, np.array([1,-1])), (9, 7, np.array([1,1])), (10, 8, np.array([1,1]))],
            # d = 5.291503 ; d^6 = 21952.00 
            n12NN = [(0, 5, np.array([1,-1])), (0, 5, np.array([1,0])), (1, 4, np.array([1,0])),
                    (2, 6, np.array([0,1])), (2, 6, np.array([1,0])), (3, 2, np.array([0,1])),
                    (3, 2, np.array([1,0])), (4, 1, np.array([0,1])), (4, 10, np.array([1,0])),
                    (5, 0, np.array([1,0])), (5, 0, np.array([1,1])), (6, 11, np.array([0,0])),
                    (6, 11, np.array([1,-1])), (7, 10, np.array([0,0])), (7, 1, np.array([1,-1])),
                    (7, 1, np.array([2,0])), (8, 9, np.array([1,-1])), (8, 9, np.array([1,0])),
                    (9, 8, np.array([1,0])), (9, 8, np.array([1,1])), (10, 4, np.array([0,1])),
                    (10, 7, np.array([1,1])), (11, 3, np.array([1,1])), (11, 3, np.array([2,0]))],
            # d = 5.567764 ; d^6 = 29791.00 
            n13NN = [(0, 3, np.array([1,-1])), (0, 6, np.array([1,0])), (1, 6, np.array([0,1])),
                    (2, 7, np.array([0,1])), (2, 5, np.array([1,-1])), (2, 8, np.array([1,0])),
                    (4, 9, np.array([1,0])), (4, 0, np.array([1,1])), (5, 10, np.array([1,0])),
                    (5, 1, np.array([1,1])), (6, 1, np.array([1,-1])), (7, 2, np.array([1,-1])),
                    (7, 9, np.array([1,0])), (7, 0, np.array([2,0])), (8, 10, np.array([1,0])),
                    (8, 1, np.array([2,0])), (9, 6, np.array([1,0])), (9, 3, np.array([2,0])),
                    (10, 3, np.array([0,1])), (10, 3, np.array([1,1])), (11, 4, np.array([0,1])),
                    (11, 8, np.array([1,0])), (11, 4, np.array([1,1])), (11, 5, np.array([2,0]))]
        )
        
        kwargs['pairs'] = ruby_pairs
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)

class RubyXC_blockade(Lattice):
    """A ruby lattice for Hilbert space blockade models and grouped sites (hexagon)
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 2

    def __init__(self, Lx, Ly, sites, rho= 1, **kwargs):
        sites = _parse_sites(sites, 2)
        #     1        
        #     |
        #     0
      
        pos = np.array([[0, 0],  [0, 1]])
        pos = pos*(rho+1/np.sqrt(3))
        basis = [[0.5*(1+(3**0.5)*rho), -0.5*(3*rho+(3**0.5))], [0, (3**0.5)+3*rho]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class RubyXC_group_rhoVertices(Lattice):
    """NOTE: Auxilliary for "RubyXC_group_rho" only. Contains just the sites/vertices, for finding the pairs.
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    rho : float
        Anisotropy ratio of the rectangles. 
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    def __init__(self, Lx, Ly, sites, rho= 1, **kwargs):
        sites = _parse_sites(sites, 12)
        #               11
        #              / \
        #             9---10
        #             |   |
        #             |   |
        #             7---8
        #              \ /
        #     5         6
        #    / \
        #   3---4       
        #   |   |
        #   |   |
        #   1---2
        #   \   / 
        #     0

        shift = np.array([0.5*(1+(3**0.5)*rho), 0.5*(3*rho+(3**0.5))])
        pos1 = np.array([[0, 0],  [-0.5, 0.5*(3**0.5)],  [0.5, 0.5*(3**0.5)],
                       [-0.5, 0.5*(3**0.5)+rho], [0.5, 0.5*(3**0.5)+rho], [0, (3**0.5)+rho]])
        pos2 = pos1 + shift
        pos = np.vstack((pos1, pos2))
        basis = [[(3**0.5)*rho+1, 0], [0, 3*rho+3**0.5]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)
    

class RubyXC_group_rho(Lattice):
    """A ruby lattice with a general aspect ratio rho (kwargs)
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    rho : float
        Anisotropy ratio of the rectangles. 
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2
    Lu = 12

    def __init__(self, Lx, Ly, sites, rho= 1, **kwargs):
        sites = _parse_sites(sites, 12) 
        #               11
        #              / \
        #             9---10
        #             |   |
        #             |   |
        #             7---8
        #              \ /
        #     5         6
        #    / \
        #   3---4       
        #   |   |
        #   |   |
        #   1---2
        #   \   / 
        #     0  
      
        shift = np.array([0.5*(1+(3**0.5)*rho), 0.5*(3*rho+(3**0.5))])
        pos1 = np.array([[0, 0],  [-0.5, 0.5*(3**0.5)],  [0.5, 0.5*(3**0.5)],
                       [-0.5, 0.5*(3**0.5)+rho], [0.5, 0.5*(3**0.5)+rho], [0, (3**0.5)+rho]])
        pos2 = pos1 + shift
        pos = np.vstack((pos1, pos2))
        basis = [[(3**0.5)*rho+1, 0], [0, 3*rho+3**0.5]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        ### Construct the pairs using the RubyXC_rhoVertices class as an auxilliary lattice
        # Create a large lattice patch - size unimportant, just need to call lat.position()
        cutoff = 8
        aux_lat = RubyXC_group_rhoVertices(Lx=cutoff, Ly=cutoff, sites=None, rho=rho, bc=['open', 'open'])
        # Use the tenpy method to get them
        aux_dict = aux_lat.find_coupling_pairs(cutoff)

        # The aux dictionary keys = coupling distance; we change to "nearest_neighbor" style
        new_dict = {}
        nNN = 0
        for old_key in aux_dict.keys():
            new_key = 'nearest_neighbors' if nNN==0 else 'n%d_nearest_neighbors' %nNN
            new_dict[new_key]=aux_dict[old_key]
            nNN+=1
        
        kwargs['pairs'] = new_dict
        
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class DoubledKagomeYC(Lattice):
    """A Kagome lattice with YC orientation, and a doubled unit cell (for pi-flux models)
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.DoubledKagomeYC(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2  #: the dimension of the lattice
    Lu = 6  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.

    def __init__(self, Lx, Ly, sites, **kwargs):
        sites = _parse_sites(sites, 6)
        #           5
        #          / \
        #         /   \
        #   -----3-----4-----
        #   \   /
        #    \ /
        #     2
        #    / \
        #   /   \
        #  0-----1-----
        pos = np.array([[0, 0], [1, 0], [0.5, 0.5 * 3**0.5],
                        [1, 3**0.5], [2, 3**0.5], [1.5, 1.5 * 3**0.5]])
        basis = [2 * pos[1], 4 * pos[2]]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        # NN = [(0, 1, np.array([0, 0])), (0, 2, np.array([0, 0])), (1, 2, np.array([0, 0])),
        #       (1, 0, np.array([1, 0])), (2, 0, np.array([0, 1])), (2, 1, np.array([-1, 1]))]
        # nNN = [(0, 1, np.array([0, -1])), (0, 2, np.array([1, -1])), (1, 0, np.array([1, -1])),
        #        (1, 2, np.array([1, 0])), (2, 0, np.array([1, 0])), (2, 1, np.array([0, 1]))]
        # nnNN = [(0, 0, np.array([1, -1])), (0, 0, np.array([0, 1])), (0, 0, np.array([1, 0])),
        #         (1, 1, np.array([1, -1])), (1, 1, np.array([0, 1])), (1, 1, np.array([1, 0])),
        #         (2, 2, np.array([1, -1])), (2, 2, np.array([0, 1])), (2, 2, np.array([1, 0]))]
        # kwargs.setdefault('pairs', {})
        # kwargs['pairs'].setdefault('nearest_neighbors', NN)
        # kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        # kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        kagome_pairs = dict(
            # d = 1.000000 ; d^3 = 1.00 
            nearest_neighbors = [(0, 1, np.array([0, 0])), (0, 1, np.array([-1,  0])), (0, 2, np.array([0, 0])),
                                 (0, 5, np.array([ 0, -1])), (1, 2, np.array([0, 0])), (1, 5, np.array([ 1, -1])),
                                 (2, 3, np.array([0, 0])), (2, 4, np.array([-1,  0])), (3, 4, np.array([0, 0])),
                                 (3, 4, np.array([-1,  0])), (3, 5, np.array([0, 0])), (4, 5, np.array([0, 0]))],
            # d = 1.732051 ; d^3 = 5.20 
            n1_nearest_neighbors = [(0, 2, np.array([-1,  0])), (0, 4, np.array([ 0, -1])), (0, 4, np.array([-1,  0])),
                                    (0, 5, np.array([ 1, -1])), (1, 2, np.array([1, 0])), (1, 3, np.array([ 1, -1])),
                                    (1, 3, np.array([0, 0])), (1, 5, np.array([ 0, -1])), (2, 3, np.array([-1,  0])),
                                    (2, 4, np.array([0, 0])), (3, 5, np.array([-1,  0])), (4, 5, np.array([1, 0]))],
            # d = 2.000000 ; d^3 = 8.00 
            n2_nearest_neighbors = [(0, 0, np.array([1, 0])), (0, 3, np.array([ 1, -1])), (0, 3, np.array([0, 0])),
                                    (0, 3, np.array([ 0, -1])), (0, 3, np.array([-1,  0])), (1, 1, np.array([1, 0])),
                                    (1, 4, np.array([ 1, -1])), (1, 4, np.array([0, 0])), (1, 4, np.array([ 0, -1])),
                                    (1, 4, np.array([-1,  0])), (2, 2, np.array([1, 0])), (2, 5, np.array([ 1, -1])),
                                    (2, 5, np.array([0, 0])), (2, 5, np.array([ 0, -1])), (2, 5, np.array([-1,  0])),
                                    (3, 3, np.array([1, 0])), (4, 4, np.array([1, 0])), (5, 5, np.array([1, 0]))],
            # d = 2.645751 ; d^3 = 18.52 
            n3_nearest_neighbors = [(0, 2, np.array([1, 0])), (0, 2, np.array([ 1, -1])), (0, 4, np.array([ 1, -1])),
                                    (0, 4, np.array([0, 0])), (0, 4, np.array([-1, -1])), (0, 4, np.array([-2,  0])),
                                    (0, 5, np.array([-1,  0])), (0, 5, np.array([-1, -1])), (1, 2, np.array([ 1, -1])),
                                    (1, 2, np.array([-1,  0])), (1, 3, np.array([ 2, -1])), (1, 3, np.array([1, 0])),
                                    (1, 3, np.array([ 0, -1])), (1, 3, np.array([-1,  0])), (1, 5, np.array([ 2, -1])),
                                    (1, 5, np.array([0, 0])), (2, 3, np.array([1, 0])), (2, 3, np.array([ 1, -1])),
                                    (2, 4, np.array([ 0, -1])), (2, 4, np.array([-2,  0])), (3, 5, np.array([1, 0])),
                                    (3, 5, np.array([ 1, -1])), (4, 5, np.array([ 1, -1])), (4, 5, np.array([-1,  0]))]
        )
        kwargs['pairs'] = kagome_pairs

        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


# up to 20x20
class Square3(Square):
    """same as :class:`~tenpy.models.lattice.Square`, but add more pairs"""
    def __init__(self, Lx, Ly, sites, **kwargs):
        super().__init__(Lx, Ly, sites, **kwargs)
        self.pairs = dict(
            # d = 1.000000 ; d^3 = 1.00
            nearest_neighbors = [(0, 0, np.array([1, 0])), (0, 0, np.array([0, 1]))],
            # d = 1.414214 ; d^3 = 2.83
            n1_nearest_neighbors = [(0, 0, np.array([1, 1])), (0, 0, np.array([1, -1]))],
            # d = 2.000000 ; d^3 = 8.00
            n2_nearest_neighbors = [(0, 0, np.array([2, 0])), (0, 0, np.array([0, 2]))],
            # d = 2.236068 ; d^3 = 11.18
            n3_nearest_neighbors = [(0, 0, np.array([2, 1])), (0, 0, np.array([2, -1])), (0, 0, np.array([1, 2])), (0, 0, np.array([1, -2]))],
            # d = 2.828427 ; d^3 = 22.63
            n4_nearest_neighbors = [(0, 0, np.array([2, 2])), (0, 0, np.array([2, -2]))],
            # d = 3.000000 ; d^3 = 27.00
            n5_nearest_neighbors = [(0, 0, np.array([3, 0])), (0, 0, np.array([0, 3]))],
            # d = 3.162278 ; d^3 = 31.62
            n6_nearest_neighbors = [(0, 0, np.array([3, 1])), (0, 0, np.array([3, -1])), (0, 0, np.array([1, 3])), (0, 0, np.array([1, -3]))],
            # d = 3.605551 ; d^3 = 46.87
            n7_nearest_neighbors = [(0, 0, np.array([3, 2])), (0, 0, np.array([3, -2])), (0, 0, np.array([2, 3])), (0, 0, np.array([2, -3]))],
            # d = 4.000000 ; d^3 = 64.00
            n8_nearest_neighbors = [(0, 0, np.array([4, 0])), (0, 0, np.array([0, 4]))],
            # d = 4.123106 ; d^3 = 70.09
            n9_nearest_neighbors = [(0, 0, np.array([4, 1])), (0, 0, np.array([4, -1])), (0, 0, np.array([1, 4])), (0, 0, np.array([1, -4]))],
            # d = 4.242641 ; d^3 = 76.37
            n10_nearest_neighbors = [(0, 0, np.array([3, 3])), (0, 0, np.array([3, -3]))],
            # d = 4.472136 ; d^3 = 89.44
            n11_nearest_neighbors = [(0, 0, np.array([4, 2])), (0, 0, np.array([4, -2])), (0, 0, np.array([2, 4])), (0, 0, np.array([2, -4]))],
            # d = 5.000000 ; d^3 = 125.00
            n12_nearest_neighbors = [(0, 0, np.array([5, 0])), (0, 0, np.array([4, 3])), (0, 0, np.array([4, -3])), (0, 0, np.array([3, 4])), (0, 0, np.array([3, -4])), (0, 0, np.array([0, 5]))],
            # d = 5.099020 ; d^3 = 132.57
            n13_nearest_neighbors = [(0, 0, np.array([5, 1])), (0, 0, np.array([5, -1])), (0, 0, np.array([1, 5])), (0, 0, np.array([1, -5]))],
            # d = 5.385165 ; d^3 = 156.17
            n14_nearest_neighbors = [(0, 0, np.array([5, 2])), (0, 0, np.array([5, -2])), (0, 0, np.array([2, 5])), (0, 0, np.array([2, -5]))],
            # d = 5.656854 ; d^3 = 181.02
            n15_nearest_neighbors = [(0, 0, np.array([4, 4])), (0, 0, np.array([4, -4]))],
            # d = 5.830952 ; d^3 = 198.25
            n16_nearest_neighbors = [(0, 0, np.array([5, 3])), (0, 0, np.array([5, -3])), (0, 0, np.array([3, 5])), (0, 0, np.array([3, -5]))],
            # d = 6.000000 ; d^3 = 216.00
            n17_nearest_neighbors = [(0, 0, np.array([6, 0])), (0, 0, np.array([0, 6]))],
            # d = 6.082763 ; d^3 = 225.06
            n18_nearest_neighbors = [(0, 0, np.array([6, 1])), (0, 0, np.array([6, -1])), (0, 0, np.array([1, 6])), (0, 0, np.array([1, -6]))],
            # d = 6.324555 ; d^3 = 252.98
            n19_nearest_neighbors = [(0, 0, np.array([6, 2])), (0, 0, np.array([6, -2])), (0, 0, np.array([2, 6])), (0, 0, np.array([2, -6]))],
            # d = 6.403124 ; d^3 = 262.53
            n20_nearest_neighbors = [(0, 0, np.array([5, 4])), (0, 0, np.array([5, -4])), (0, 0, np.array([4, 5])), (0, 0, np.array([4, -5]))],
            # d = 6.708204 ; d^3 = 301.87
            n21_nearest_neighbors = [(0, 0, np.array([6, 3])), (0, 0, np.array([6, -3])), (0, 0, np.array([3, 6])), (0, 0, np.array([3, -6]))],
            # d = 7.000000 ; d^3 = 343.00
            n22_nearest_neighbors = [(0, 0, np.array([7, 0])), (0, 0, np.array([0, 7]))],
            # d = 7.071068 ; d^3 = 353.55
            n23_nearest_neighbors = [(0, 0, np.array([7, 1])), (0, 0, np.array([7, -1])), (0, 0, np.array([5, 5])), (0, 0, np.array([5, -5])), (0, 0, np.array([1, 7])), (0, 0, np.array([1, -7]))],
            # d = 7.211103 ; d^3 = 374.98
            n24_nearest_neighbors = [(0, 0, np.array([6, 4])), (0, 0, np.array([6, -4])), (0, 0, np.array([4, 6])), (0, 0, np.array([4, -6]))],
            # d = 7.280110 ; d^3 = 385.85
            n25_nearest_neighbors = [(0, 0, np.array([7, 2])), (0, 0, np.array([7, -2])), (0, 0, np.array([2, 7])), (0, 0, np.array([2, -7]))],
            # d = 7.615773 ; d^3 = 441.71
            n26_nearest_neighbors = [(0, 0, np.array([7, 3])), (0, 0, np.array([7, -3])), (0, 0, np.array([3, 7])), (0, 0, np.array([3, -7]))],
            # d = 7.810250 ; d^3 = 476.43
            n27_nearest_neighbors = [(0, 0, np.array([6, 5])), (0, 0, np.array([6, -5])), (0, 0, np.array([5, 6])), (0, 0, np.array([5, -6]))],
            # d = 8.000000 ; d^3 = 512.00
            n28_nearest_neighbors = [(0, 0, np.array([8, 0])), (0, 0, np.array([0, 8]))],
            # d = 8.062258 ; d^3 = 524.05
            n29_nearest_neighbors = [(0, 0, np.array([8, 1])), (0, 0, np.array([8, -1])), (0, 0, np.array([7, 4])), (0, 0, np.array([7, -4])), (0, 0, np.array([4, 7])), (0, 0, np.array([4, -7])), (0, 0, np.array([1, 8])), (0, 0, np.array([1, -8]))],
            # d = 8.246211 ; d^3 = 560.74
            n30_nearest_neighbors = [(0, 0, np.array([8, 2])), (0, 0, np.array([8, -2])), (0, 0, np.array([2, 8])), (0, 0, np.array([2, -8]))],
            # d = 8.485281 ; d^3 = 610.94
            n31_nearest_neighbors = [(0, 0, np.array([6, 6])), (0, 0, np.array([6, -6]))],
            # d = 8.544004 ; d^3 = 623.71
            n32_nearest_neighbors = [(0, 0, np.array([8, 3])), (0, 0, np.array([8, -3])), (0, 0, np.array([3, 8])), (0, 0, np.array([3, -8]))],
            # d = 8.602325 ; d^3 = 636.57
            n33_nearest_neighbors = [(0, 0, np.array([7, 5])), (0, 0, np.array([7, -5])), (0, 0, np.array([5, 7])), (0, 0, np.array([5, -7]))],
            # d = 8.944272 ; d^3 = 715.54
            n34_nearest_neighbors = [(0, 0, np.array([8, 4])), (0, 0, np.array([8, -4])), (0, 0, np.array([4, 8])), (0, 0, np.array([4, -8]))],
            # d = 9.000000 ; d^3 = 729.00
            n35_nearest_neighbors = [(0, 0, np.array([9, 0])), (0, 0, np.array([0, 9]))],
            # d = 9.055385 ; d^3 = 742.54
            n36_nearest_neighbors = [(0, 0, np.array([9, 1])), (0, 0, np.array([9, -1])), (0, 0, np.array([1, 9])), (0, 0, np.array([1, -9]))],
            # d = 9.219544 ; d^3 = 783.66
            n37_nearest_neighbors = [(0, 0, np.array([9, 2])), (0, 0, np.array([9, -2])), (0, 0, np.array([7, 6])), (0, 0, np.array([7, -6])), (0, 0, np.array([6, 7])), (0, 0, np.array([6, -7])), (0, 0, np.array([2, 9])), (0, 0, np.array([2, -9]))],
            # d = 9.433981 ; d^3 = 839.62
            n38_nearest_neighbors = [(0, 0, np.array([8, 5])), (0, 0, np.array([8, -5])), (0, 0, np.array([5, 8])), (0, 0, np.array([5, -8]))],
            # d = 9.486833 ; d^3 = 853.81
            n39_nearest_neighbors = [(0, 0, np.array([9, 3])), (0, 0, np.array([9, -3])), (0, 0, np.array([3, 9])), (0, 0, np.array([3, -9]))],
            # d = 9.848858 ; d^3 = 955.34
            n40_nearest_neighbors = [(0, 0, np.array([9, 4])), (0, 0, np.array([9, -4])), (0, 0, np.array([4, 9])), (0, 0, np.array([4, -9]))],
            # d = 9.899495 ; d^3 = 970.15
            n41_nearest_neighbors = [(0, 0, np.array([7, 7])), (0, 0, np.array([7, -7]))],
            # d = 10.000000 ; d^3 = 1000.00
            n42_nearest_neighbors = [(0, 0, np.array([10,  0])), (0, 0, np.array([8, 6])), (0, 0, np.array([8, -6])), (0, 0, np.array([6, 8])), (0, 0, np.array([6, -8])), (0, 0, np.array([0, 10]))],
            # d = 10.049876 ; d^3 = 1015.04
            n43_nearest_neighbors = [(0, 0, np.array([10,  1])), (0, 0, np.array([10, -1])), (0, 0, np.array([1, 10])), (0, 0, np.array([ 1, -10]))],
            # d = 10.198039 ; d^3 = 1060.60
            n44_nearest_neighbors = [(0, 0, np.array([10,  2])), (0, 0, np.array([10, -2])), (0, 0, np.array([2, 10])), (0, 0, np.array([ 2, -10]))],
            # d = 10.295630 ; d^3 = 1091.34
            n45_nearest_neighbors = [(0, 0, np.array([9, 5])), (0, 0, np.array([9, -5])), (0, 0, np.array([5, 9])), (0, 0, np.array([5, -9]))],
            # d = 10.440307 ; d^3 = 1137.99
            n46_nearest_neighbors = [(0, 0, np.array([10,  3])), (0, 0, np.array([10, -3])), (0, 0, np.array([3, 10])), (0, 0, np.array([ 3, -10]))],
            # d = 10.630146 ; d^3 = 1201.21
            n47_nearest_neighbors = [(0, 0, np.array([8, 7])), (0, 0, np.array([8, -7])), (0, 0, np.array([7, 8])), (0, 0, np.array([7, -8]))],
            # d = 10.770330 ; d^3 = 1249.36
            n48_nearest_neighbors = [(0, 0, np.array([10,  4])), (0, 0, np.array([10, -4])), (0, 0, np.array([4, 10])), (0, 0, np.array([ 4, -10]))],
            # d = 10.816654 ; d^3 = 1265.55
            n49_nearest_neighbors = [(0, 0, np.array([9, 6])), (0, 0, np.array([9, -6])), (0, 0, np.array([6, 9])), (0, 0, np.array([6, -9]))],
            # d = 11.000000 ; d^3 = 1331.00
            n50_nearest_neighbors = [(0, 0, np.array([11,  0])), (0, 0, np.array([0, 11]))],
            # d = 11.045361 ; d^3 = 1347.53
            n51_nearest_neighbors = [(0, 0, np.array([11,  1])), (0, 0, np.array([11, -1])), (0, 0, np.array([1, 11])), (0, 0, np.array([ 1, -11]))],
            # d = 11.180340 ; d^3 = 1397.54
            n52_nearest_neighbors = [(0, 0, np.array([11,  2])), (0, 0, np.array([11, -2])), (0, 0, np.array([10,  5])), (0, 0, np.array([10, -5])), (0, 0, np.array([5, 10])), (0, 0, np.array([ 5, -10])), (0, 0, np.array([2, 11])), (0, 0, np.array([ 2, -11]))],
            # d = 11.313708 ; d^3 = 1448.15
            n53_nearest_neighbors = [(0, 0, np.array([8, 8])), (0, 0, np.array([8, -8]))],
            # d = 11.401754 ; d^3 = 1482.23
            n54_nearest_neighbors = [(0, 0, np.array([11,  3])), (0, 0, np.array([11, -3])), (0, 0, np.array([9, 7])), (0, 0, np.array([9, -7])), (0, 0, np.array([7, 9])), (0, 0, np.array([7, -9])), (0, 0, np.array([3, 11])), (0, 0, np.array([ 3, -11]))],
            # d = 11.661904 ; d^3 = 1586.02
            n55_nearest_neighbors = [(0, 0, np.array([10,  6])), (0, 0, np.array([10, -6])), (0, 0, np.array([6, 10])), (0, 0, np.array([ 6, -10]))],
            # d = 11.704700 ; d^3 = 1603.54
            n56_nearest_neighbors = [(0, 0, np.array([11,  4])), (0, 0, np.array([11, -4])), (0, 0, np.array([4, 11])), (0, 0, np.array([ 4, -11]))],
            # d = 12.000000 ; d^3 = 1728.00
            n57_nearest_neighbors = [(0, 0, np.array([12,  0])), (0, 0, np.array([0, 12]))],
            # d = 12.041595 ; d^3 = 1746.03
            n58_nearest_neighbors = [(0, 0, np.array([12,  1])), (0, 0, np.array([12, -1])), (0, 0, np.array([9, 8])), (0, 0, np.array([9, -8])), (0, 0, np.array([8, 9])), (0, 0, np.array([8, -9])), (0, 0, np.array([1, 12])), (0, 0, np.array([ 1, -12]))],
            # d = 12.083046 ; d^3 = 1764.12
            n59_nearest_neighbors = [(0, 0, np.array([11,  5])), (0, 0, np.array([11, -5])), (0, 0, np.array([5, 11])), (0, 0, np.array([ 5, -11]))],
            # d = 12.165525 ; d^3 = 1800.50
            n60_nearest_neighbors = [(0, 0, np.array([12,  2])), (0, 0, np.array([12, -2])), (0, 0, np.array([2, 12])), (0, 0, np.array([ 2, -12]))],
            # d = 12.206556 ; d^3 = 1818.78
            n61_nearest_neighbors = [(0, 0, np.array([10,  7])), (0, 0, np.array([10, -7])), (0, 0, np.array([7, 10])), (0, 0, np.array([ 7, -10]))],
            # d = 12.369317 ; d^3 = 1892.51
            n62_nearest_neighbors = [(0, 0, np.array([12,  3])), (0, 0, np.array([12, -3])), (0, 0, np.array([3, 12])), (0, 0, np.array([ 3, -12]))],
            # d = 12.529964 ; d^3 = 1967.20
            n63_nearest_neighbors = [(0, 0, np.array([11,  6])), (0, 0, np.array([11, -6])), (0, 0, np.array([6, 11])), (0, 0, np.array([ 6, -11]))],
            # d = 12.649111 ; d^3 = 2023.86
            n64_nearest_neighbors = [(0, 0, np.array([12,  4])), (0, 0, np.array([12, -4])), (0, 0, np.array([4, 12])), (0, 0, np.array([ 4, -12]))],
            # d = 12.727922 ; d^3 = 2061.92
            n65_nearest_neighbors = [(0, 0, np.array([9, 9])), (0, 0, np.array([9, -9]))],
            # d = 12.806248 ; d^3 = 2100.22
            n66_nearest_neighbors = [(0, 0, np.array([10,  8])), (0, 0, np.array([10, -8])), (0, 0, np.array([8, 10])), (0, 0, np.array([ 8, -10]))],
            # d = 13.000000 ; d^3 = 2197.00
            n67_nearest_neighbors = [(0, 0, np.array([13,  0])), (0, 0, np.array([12,  5])), (0, 0, np.array([12, -5])), (0, 0, np.array([5, 12])), (0, 0, np.array([ 5, -12])), (0, 0, np.array([0, 13]))],
            # d = 13.038405 ; d^3 = 2216.53
            n68_nearest_neighbors = [(0, 0, np.array([13,  1])), (0, 0, np.array([13, -1])), (0, 0, np.array([11,  7])), (0, 0, np.array([11, -7])), (0, 0, np.array([7, 11])), (0, 0, np.array([ 7, -11])), (0, 0, np.array([1, 13])), (0, 0, np.array([ 1, -13]))],
            # d = 13.152946 ; d^3 = 2275.46
            n69_nearest_neighbors = [(0, 0, np.array([13,  2])), (0, 0, np.array([13, -2])), (0, 0, np.array([2, 13])), (0, 0, np.array([ 2, -13]))],
            # d = 13.341664 ; d^3 = 2374.82
            n70_nearest_neighbors = [(0, 0, np.array([13,  3])), (0, 0, np.array([13, -3])), (0, 0, np.array([3, 13])), (0, 0, np.array([ 3, -13]))],
            # d = 13.416408 ; d^3 = 2414.95
            n71_nearest_neighbors = [(0, 0, np.array([12,  6])), (0, 0, np.array([12, -6])), (0, 0, np.array([6, 12])), (0, 0, np.array([ 6, -12]))],
            # d = 13.453624 ; d^3 = 2435.11
            n72_nearest_neighbors = [(0, 0, np.array([10,  9])), (0, 0, np.array([10, -9])), (0, 0, np.array([9, 10])), (0, 0, np.array([ 9, -10]))],
            # d = 13.601471 ; d^3 = 2516.27
            n73_nearest_neighbors = [(0, 0, np.array([13,  4])), (0, 0, np.array([13, -4])), (0, 0, np.array([11,  8])), (0, 0, np.array([11, -8])), (0, 0, np.array([8, 11])), (0, 0, np.array([ 8, -11])), (0, 0, np.array([4, 13])), (0, 0, np.array([ 4, -13]))],
            # d = 13.892444 ; d^3 = 2681.24
            n74_nearest_neighbors = [(0, 0, np.array([12,  7])), (0, 0, np.array([12, -7])), (0, 0, np.array([7, 12])), (0, 0, np.array([ 7, -12]))],
            # d = 13.928388 ; d^3 = 2702.11
            n75_nearest_neighbors = [(0, 0, np.array([13,  5])), (0, 0, np.array([13, -5])), (0, 0, np.array([5, 13])), (0, 0, np.array([ 5, -13]))],
            # d = 14.000000 ; d^3 = 2744.00
            n76_nearest_neighbors = [(0, 0, np.array([14,  0])), (0, 0, np.array([0, 14]))],
            # d = 14.035669 ; d^3 = 2765.03
            n77_nearest_neighbors = [(0, 0, np.array([14,  1])), (0, 0, np.array([14, -1])), (0, 0, np.array([1, 14])), (0, 0, np.array([ 1, -14]))],
            # d = 14.142136 ; d^3 = 2828.43
            n78_nearest_neighbors = [(0, 0, np.array([14,  2])), (0, 0, np.array([14, -2])), (0, 0, np.array([10, 10])), (0, 0, np.array([10, -10])), (0, 0, np.array([2, 14])), (0, 0, np.array([ 2, -14]))],
            # d = 14.212670 ; d^3 = 2870.96
            n79_nearest_neighbors = [(0, 0, np.array([11,  9])), (0, 0, np.array([11, -9])), (0, 0, np.array([9, 11])), (0, 0, np.array([ 9, -11]))],
            # d = 14.317821 ; d^3 = 2935.15
            n80_nearest_neighbors = [(0, 0, np.array([14,  3])), (0, 0, np.array([14, -3])), (0, 0, np.array([13,  6])), (0, 0, np.array([13, -6])), (0, 0, np.array([6, 13])), (0, 0, np.array([ 6, -13])), (0, 0, np.array([3, 14])), (0, 0, np.array([ 3, -14]))],
            # d = 14.422205 ; d^3 = 2999.82
            n81_nearest_neighbors = [(0, 0, np.array([12,  8])), (0, 0, np.array([12, -8])), (0, 0, np.array([8, 12])), (0, 0, np.array([ 8, -12]))],
            # d = 14.560220 ; d^3 = 3086.77
            n82_nearest_neighbors = [(0, 0, np.array([14,  4])), (0, 0, np.array([14, -4])), (0, 0, np.array([4, 14])), (0, 0, np.array([ 4, -14]))],
            # d = 14.764823 ; d^3 = 3218.73
            n83_nearest_neighbors = [(0, 0, np.array([13,  7])), (0, 0, np.array([13, -7])), (0, 0, np.array([7, 13])), (0, 0, np.array([ 7, -13]))],
            # d = 14.866069 ; d^3 = 3285.40
            n84_nearest_neighbors = [(0, 0, np.array([14,  5])), (0, 0, np.array([14, -5])), (0, 0, np.array([11, 10])), (0, 0, np.array([11, -10])), (0, 0, np.array([10, 11])), (0, 0, np.array([10, -11])), (0, 0, np.array([5, 14])), (0, 0, np.array([ 5, -14]))],
            # d = 15.000000 ; d^3 = 3375.00
            n85_nearest_neighbors = [(0, 0, np.array([15,  0])), (0, 0, np.array([12,  9])), (0, 0, np.array([12, -9])), (0, 0, np.array([9, 12])), (0, 0, np.array([ 9, -12])), (0, 0, np.array([0, 15]))],
            # d = 15.033296 ; d^3 = 3397.52
            n86_nearest_neighbors = [(0, 0, np.array([15,  1])), (0, 0, np.array([15, -1])), (0, 0, np.array([1, 15])), (0, 0, np.array([ 1, -15]))],
            # d = 15.132746 ; d^3 = 3465.40
            n87_nearest_neighbors = [(0, 0, np.array([15,  2])), (0, 0, np.array([15, -2])), (0, 0, np.array([2, 15])), (0, 0, np.array([ 2, -15]))],
            # d = 15.231546 ; d^3 = 3533.72
            n88_nearest_neighbors = [(0, 0, np.array([14,  6])), (0, 0, np.array([14, -6])), (0, 0, np.array([6, 14])), (0, 0, np.array([ 6, -14]))],
            # d = 15.264338 ; d^3 = 3556.59
            n89_nearest_neighbors = [(0, 0, np.array([13,  8])), (0, 0, np.array([13, -8])), (0, 0, np.array([8, 13])), (0, 0, np.array([ 8, -13]))],
            # d = 15.297059 ; d^3 = 3579.51
            n90_nearest_neighbors = [(0, 0, np.array([15,  3])), (0, 0, np.array([15, -3])), (0, 0, np.array([3, 15])), (0, 0, np.array([ 3, -15]))],
            # d = 15.524175 ; d^3 = 3741.33
            n91_nearest_neighbors = [(0, 0, np.array([15,  4])), (0, 0, np.array([15, -4])), (0, 0, np.array([4, 15])), (0, 0, np.array([ 4, -15]))],
            # d = 15.556349 ; d^3 = 3764.64
            n92_nearest_neighbors = [(0, 0, np.array([11, 11])), (0, 0, np.array([11, -11]))],
            # d = 15.620499 ; d^3 = 3811.40
            n93_nearest_neighbors = [(0, 0, np.array([12, 10])), (0, 0, np.array([12, -10])), (0, 0, np.array([10, 12])), (0, 0, np.array([10, -12]))],
            # d = 15.652476 ; d^3 = 3834.86
            n94_nearest_neighbors = [(0, 0, np.array([14,  7])), (0, 0, np.array([14, -7])), (0, 0, np.array([7, 14])), (0, 0, np.array([ 7, -14]))],
            # d = 15.811388 ; d^3 = 3952.85
            n95_nearest_neighbors = [(0, 0, np.array([15,  5])), (0, 0, np.array([15, -5])), (0, 0, np.array([13,  9])), (0, 0, np.array([13, -9])), (0, 0, np.array([9, 13])), (0, 0, np.array([ 9, -13])), (0, 0, np.array([5, 15])), (0, 0, np.array([ 5, -15]))],
            # d = 16.000000 ; d^3 = 4096.00
            n96_nearest_neighbors = [(0, 0, np.array([16,  0])), (0, 0, np.array([0, 16]))],
            # d = 16.031220 ; d^3 = 4120.02
            n97_nearest_neighbors = [(0, 0, np.array([16,  1])), (0, 0, np.array([16, -1])), (0, 0, np.array([1, 16])), (0, 0, np.array([ 1, -16]))],
            # d = 16.124515 ; d^3 = 4192.37
            n98_nearest_neighbors = [(0, 0, np.array([16,  2])), (0, 0, np.array([16, -2])), (0, 0, np.array([14,  8])), (0, 0, np.array([14, -8])), (0, 0, np.array([8, 14])), (0, 0, np.array([ 8, -14])), (0, 0, np.array([2, 16])), (0, 0, np.array([ 2, -16]))],
            # d = 16.155494 ; d^3 = 4216.58
            n99_nearest_neighbors = [(0, 0, np.array([15,  6])), (0, 0, np.array([15, -6])), (0, 0, np.array([6, 15])), (0, 0, np.array([ 6, -15]))],
            # d = 16.278821 ; d^3 = 4313.89
            n100_nearest_neighbors = [(0, 0, np.array([16,  3])), (0, 0, np.array([16, -3])), (0, 0, np.array([12, 11])), (0, 0, np.array([12, -11])), (0, 0, np.array([11, 12])), (0, 0, np.array([11, -12])), (0, 0, np.array([3, 16])), (0, 0, np.array([ 3, -16]))],
            # d = 16.401219 ; d^3 = 4411.93
            n101_nearest_neighbors = [(0, 0, np.array([13, 10])), (0, 0, np.array([13, -10])), (0, 0, np.array([10, 13])), (0, 0, np.array([10, -13]))],
            # d = 16.492423 ; d^3 = 4485.94
            n102_nearest_neighbors = [(0, 0, np.array([16,  4])), (0, 0, np.array([16, -4])), (0, 0, np.array([4, 16])), (0, 0, np.array([ 4, -16]))],
            # d = 16.552945 ; d^3 = 4535.51
            n103_nearest_neighbors = [(0, 0, np.array([15,  7])), (0, 0, np.array([15, -7])), (0, 0, np.array([7, 15])), (0, 0, np.array([ 7, -15]))],
            # d = 16.643317 ; d^3 = 4610.20
            n104_nearest_neighbors = [(0, 0, np.array([14,  9])), (0, 0, np.array([14, -9])), (0, 0, np.array([9, 14])), (0, 0, np.array([ 9, -14]))],
            # d = 16.763055 ; d^3 = 4710.42
            n105_nearest_neighbors = [(0, 0, np.array([16,  5])), (0, 0, np.array([16, -5])), (0, 0, np.array([5, 16])), (0, 0, np.array([ 5, -16]))],
            # d = 16.970563 ; d^3 = 4887.52
            n106_nearest_neighbors = [(0, 0, np.array([12, 12])), (0, 0, np.array([12, -12]))],
            # d = 17.000000 ; d^3 = 4913.00
            n107_nearest_neighbors = [(0, 0, np.array([17,  0])), (0, 0, np.array([15,  8])), (0, 0, np.array([15, -8])), (0, 0, np.array([8, 15])), (0, 0, np.array([ 8, -15])), (0, 0, np.array([0, 17]))],
            # d = 17.029386 ; d^3 = 4938.52
            n108_nearest_neighbors = [(0, 0, np.array([17,  1])), (0, 0, np.array([17, -1])), (0, 0, np.array([13, 11])), (0, 0, np.array([13, -11])), (0, 0, np.array([11, 13])), (0, 0, np.array([11, -13])), (0, 0, np.array([1, 17])), (0, 0, np.array([ 1, -17]))],
            # d = 17.088007 ; d^3 = 4989.70
            n109_nearest_neighbors = [(0, 0, np.array([16,  6])), (0, 0, np.array([16, -6])), (0, 0, np.array([6, 16])), (0, 0, np.array([ 6, -16]))],
            # d = 17.117243 ; d^3 = 5015.35
            n110_nearest_neighbors = [(0, 0, np.array([17,  2])), (0, 0, np.array([17, -2])), (0, 0, np.array([2, 17])), (0, 0, np.array([ 2, -17]))],
            # d = 17.204651 ; d^3 = 5092.58
            n111_nearest_neighbors = [(0, 0, np.array([14, 10])), (0, 0, np.array([14, -10])), (0, 0, np.array([10, 14])), (0, 0, np.array([10, -14]))],
            # d = 17.262677 ; d^3 = 5144.28
            n112_nearest_neighbors = [(0, 0, np.array([17,  3])), (0, 0, np.array([17, -3])), (0, 0, np.array([3, 17])), (0, 0, np.array([ 3, -17]))],
            # d = 17.464249 ; d^3 = 5326.60
            n113_nearest_neighbors = [(0, 0, np.array([17,  4])), (0, 0, np.array([17, -4])), (0, 0, np.array([16,  7])), (0, 0, np.array([16, -7])), (0, 0, np.array([7, 16])), (0, 0, np.array([ 7, -16])), (0, 0, np.array([4, 17])), (0, 0, np.array([ 4, -17]))],
            # d = 17.492856 ; d^3 = 5352.81
            n114_nearest_neighbors = [(0, 0, np.array([15,  9])), (0, 0, np.array([15, -9])), (0, 0, np.array([9, 15])), (0, 0, np.array([ 9, -15]))],
            # d = 17.691806 ; d^3 = 5537.54
            n115_nearest_neighbors = [(0, 0, np.array([13, 12])), (0, 0, np.array([13, -12])), (0, 0, np.array([12, 13])), (0, 0, np.array([12, -13]))],
            # d = 17.720045 ; d^3 = 5564.09
            n116_nearest_neighbors = [(0, 0, np.array([17,  5])), (0, 0, np.array([17, -5])), (0, 0, np.array([5, 17])), (0, 0, np.array([ 5, -17]))],
            # d = 17.804494 ; d^3 = 5644.02
            n117_nearest_neighbors = [(0, 0, np.array([14, 11])), (0, 0, np.array([14, -11])), (0, 0, np.array([11, 14])), (0, 0, np.array([11, -14]))],
            # d = 17.888544 ; d^3 = 5724.33
            n118_nearest_neighbors = [(0, 0, np.array([16,  8])), (0, 0, np.array([16, -8])), (0, 0, np.array([8, 16])), (0, 0, np.array([ 8, -16]))],
            # d = 18.000000 ; d^3 = 5832.00
            n119_nearest_neighbors = [(0, 0, np.array([18,  0])), (0, 0, np.array([0, 18]))],
            # d = 18.027756 ; d^3 = 5859.02
            n120_nearest_neighbors = [(0, 0, np.array([18,  1])), (0, 0, np.array([18, -1])), (0, 0, np.array([17,  6])), (0, 0, np.array([17, -6])), (0, 0, np.array([15, 10])), (0, 0, np.array([15, -10])), (0, 0, np.array([10, 15])), (0, 0, np.array([10, -15])), (0, 0, np.array([6, 17])), (0, 0, np.array([ 6, -17])), (0, 0, np.array([1, 18])), (0, 0, np.array([ 1, -18]))],
            # d = 18.110770 ; d^3 = 5940.33
            n121_nearest_neighbors = [(0, 0, np.array([18,  2])), (0, 0, np.array([18, -2])), (0, 0, np.array([2, 18])), (0, 0, np.array([ 2, -18]))],
            # d = 18.248288 ; d^3 = 6076.68
            n122_nearest_neighbors = [(0, 0, np.array([18,  3])), (0, 0, np.array([18, -3])), (0, 0, np.array([3, 18])), (0, 0, np.array([ 3, -18]))],
            # d = 18.357560 ; d^3 = 6186.50
            n123_nearest_neighbors = [(0, 0, np.array([16,  9])), (0, 0, np.array([16, -9])), (0, 0, np.array([9, 16])), (0, 0, np.array([ 9, -16]))],
            # d = 18.384776 ; d^3 = 6214.05
            n124_nearest_neighbors = [(0, 0, np.array([17,  7])), (0, 0, np.array([17, -7])), (0, 0, np.array([13, 13])), (0, 0, np.array([13, -13])), (0, 0, np.array([7, 17])), (0, 0, np.array([ 7, -17]))],
            # d = 18.439089 ; d^3 = 6269.29
            n125_nearest_neighbors = [(0, 0, np.array([18,  4])), (0, 0, np.array([18, -4])), (0, 0, np.array([14, 12])), (0, 0, np.array([14, -12])), (0, 0, np.array([12, 14])), (0, 0, np.array([12, -14])), (0, 0, np.array([4, 18])), (0, 0, np.array([ 4, -18]))],
            # d = 18.601075 ; d^3 = 6435.97
            n126_nearest_neighbors = [(0, 0, np.array([15, 11])), (0, 0, np.array([15, -11])), (0, 0, np.array([11, 15])), (0, 0, np.array([11, -15]))],
            # d = 18.681542 ; d^3 = 6519.86
            n127_nearest_neighbors = [(0, 0, np.array([18,  5])), (0, 0, np.array([18, -5])), (0, 0, np.array([5, 18])), (0, 0, np.array([ 5, -18]))],
            # d = 18.788294 ; d^3 = 6632.27
            n128_nearest_neighbors = [(0, 0, np.array([17,  8])), (0, 0, np.array([17, -8])), (0, 0, np.array([8, 17])), (0, 0, np.array([ 8, -17]))],
            # d = 18.867962 ; d^3 = 6716.99
            n129_nearest_neighbors = [(0, 0, np.array([16, 10])), (0, 0, np.array([16, -10])), (0, 0, np.array([10, 16])), (0, 0, np.array([10, -16]))],
            # d = 18.973666 ; d^3 = 6830.52
            n130_nearest_neighbors = [(0, 0, np.array([18,  6])), (0, 0, np.array([18, -6])), (0, 0, np.array([6, 18])), (0, 0, np.array([ 6, -18]))],
            # d = 19.000000 ; d^3 = 6859.00
            n131_nearest_neighbors = [(0, 0, np.array([19,  0])), (0, 0, np.array([0, 19]))],
            # d = 19.026298 ; d^3 = 6887.52
            n132_nearest_neighbors = [(0, 0, np.array([19,  1])), (0, 0, np.array([19, -1])), (0, 0, np.array([1, 19])), (0, 0, np.array([ 1, -19]))],
            # d = 19.104973 ; d^3 = 6973.32
            n133_nearest_neighbors = [(0, 0, np.array([19,  2])), (0, 0, np.array([19, -2])), (0, 0, np.array([14, 13])), (0, 0, np.array([14, -13])), (0, 0, np.array([13, 14])), (0, 0, np.array([13, -14])), (0, 0, np.array([2, 19])), (0, 0, np.array([ 2, -19]))],
            # d = 19.209373 ; d^3 = 7088.26
            n134_nearest_neighbors = [(0, 0, np.array([15, 12])), (0, 0, np.array([15, -12])), (0, 0, np.array([12, 15])), (0, 0, np.array([12, -15]))],
            # d = 19.235384 ; d^3 = 7117.09
            n135_nearest_neighbors = [(0, 0, np.array([19,  3])), (0, 0, np.array([19, -3])), (0, 0, np.array([17,  9])), (0, 0, np.array([17, -9])), (0, 0, np.array([9, 17])), (0, 0, np.array([ 9, -17])), (0, 0, np.array([3, 19])), (0, 0, np.array([ 3, -19]))],
            # d = 19.313208 ; d^3 = 7203.83
            n136_nearest_neighbors = [(0, 0, np.array([18,  7])), (0, 0, np.array([18, -7])), (0, 0, np.array([7, 18])), (0, 0, np.array([ 7, -18]))],
            # d = 19.416488 ; d^3 = 7320.02
            n137_nearest_neighbors = [(0, 0, np.array([19,  4])), (0, 0, np.array([19, -4])), (0, 0, np.array([16, 11])), (0, 0, np.array([16, -11])), (0, 0, np.array([11, 16])), (0, 0, np.array([11, -16])), (0, 0, np.array([4, 19])), (0, 0, np.array([ 4, -19]))],
            # d = 19.646883 ; d^3 = 7583.70
            n138_nearest_neighbors = [(0, 0, np.array([19,  5])), (0, 0, np.array([19, -5])), (0, 0, np.array([5, 19])), (0, 0, np.array([ 5, -19]))],
            # d = 19.697716 ; d^3 = 7642.71
            n139_nearest_neighbors = [(0, 0, np.array([18,  8])), (0, 0, np.array([18, -8])), (0, 0, np.array([8, 18])), (0, 0, np.array([ 8, -18]))],
            # d = 19.723083 ; d^3 = 7672.28
            n140_nearest_neighbors = [(0, 0, np.array([17, 10])), (0, 0, np.array([17, -10])), (0, 0, np.array([10, 17])), (0, 0, np.array([10, -17]))],
            # d = 19.798990 ; d^3 = 7761.20
            n141_nearest_neighbors = [(0, 0, np.array([14, 14])), (0, 0, np.array([14, -14]))],
            # d = 19.849433 ; d^3 = 7820.68
            n142_nearest_neighbors = [(0, 0, np.array([15, 13])), (0, 0, np.array([15, -13])), (0, 0, np.array([13, 15])), (0, 0, np.array([13, -15]))],
            # d = 19.924859 ; d^3 = 7910.17
            n143_nearest_neighbors = [(0, 0, np.array([19,  6])), (0, 0, np.array([19, -6])), (0, 0, np.array([6, 19])), (0, 0, np.array([ 6, -19]))],
            # d = 20.000000 ; d^3 = 8000.00
            n144_nearest_neighbors = [(0, 0, np.array([20,  0])), (0, 0, np.array([16, 12])), (0, 0, np.array([16, -12])), (0, 0, np.array([12, 16])), (0, 0, np.array([12, -16])), (0, 0, np.array([0, 20]))],
            # d = 20.024984 ; d^3 = 8030.02
            n145_nearest_neighbors = [(0, 0, np.array([20,  1])), (0, 0, np.array([20, -1])), (0, 0, np.array([1, 20])), (0, 0, np.array([ 1, -20]))],
            # d = 20.099751 ; d^3 = 8120.30
            n146_nearest_neighbors = [(0, 0, np.array([20,  2])), (0, 0, np.array([20, -2])), (0, 0, np.array([2, 20])), (0, 0, np.array([ 2, -20]))],
            # d = 20.124612 ; d^3 = 8150.47
            n147_nearest_neighbors = [(0, 0, np.array([18,  9])), (0, 0, np.array([18, -9])), (0, 0, np.array([9, 18])), (0, 0, np.array([ 9, -18]))],
            # d = 20.223748 ; d^3 = 8271.51
            n148_nearest_neighbors = [(0, 0, np.array([20,  3])), (0, 0, np.array([20, -3])), (0, 0, np.array([3, 20])), (0, 0, np.array([ 3, -20]))],
            # d = 20.248457 ; d^3 = 8301.87
            n149_nearest_neighbors = [(0, 0, np.array([19,  7])), (0, 0, np.array([19, -7])), (0, 0, np.array([17, 11])), (0, 0, np.array([17, -11])), (0, 0, np.array([11, 17])), (0, 0, np.array([11, -17])), (0, 0, np.array([7, 19])), (0, 0, np.array([ 7, -19]))],
            # d = 20.396078 ; d^3 = 8484.77
            n150_nearest_neighbors = [(0, 0, np.array([20,  4])), (0, 0, np.array([20, -4])), (0, 0, np.array([4, 20])), (0, 0, np.array([ 4, -20]))],
            # d = 20.518285 ; d^3 = 8638.20
            n151_nearest_neighbors = [(0, 0, np.array([15, 14])), (0, 0, np.array([15, -14])), (0, 0, np.array([14, 15])), (0, 0, np.array([14, -15]))],
            # d = 20.591260 ; d^3 = 8730.69
            n152_nearest_neighbors = [(0, 0, np.array([18, 10])), (0, 0, np.array([18, -10])), (0, 0, np.array([10, 18])), (0, 0, np.array([10, -18]))],
            # d = 20.615528 ; d^3 = 8761.60
            n153_nearest_neighbors = [(0, 0, np.array([20,  5])), (0, 0, np.array([20, -5])), (0, 0, np.array([19,  8])), (0, 0, np.array([19, -8])), (0, 0, np.array([16, 13])), (0, 0, np.array([16, -13])), (0, 0, np.array([13, 16])), (0, 0, np.array([13, -16])), (0, 0, np.array([8, 19])), (0, 0, np.array([ 8, -19])), (0, 0, np.array([5, 20])), (0, 0, np.array([ 5, -20]))],
            # d = 20.808652 ; d^3 = 9010.15
            n154_nearest_neighbors = [(0, 0, np.array([17, 12])), (0, 0, np.array([17, -12])), (0, 0, np.array([12, 17])), (0, 0, np.array([12, -17]))],
            # d = 20.880613 ; d^3 = 9103.95
            n155_nearest_neighbors = [(0, 0, np.array([20,  6])), (0, 0, np.array([20, -6])), (0, 0, np.array([6, 20])), (0, 0, np.array([ 6, -20]))],
            # d = 21.000000 ; d^3 = 9261.00
            n156_nearest_neighbors = [(0, 0, np.array([21,  0])), (0, 0, np.array([0, 21]))],
            # d = 21.023796 ; d^3 = 9292.52
            n157_nearest_neighbors = [(0, 0, np.array([21,  1])), (0, 0, np.array([21, -1])), (0, 0, np.array([19,  9])), (0, 0, np.array([19, -9])), (0, 0, np.array([9, 19])), (0, 0, np.array([ 9, -19])), (0, 0, np.array([1, 21])), (0, 0, np.array([ 1, -21]))],
            # d = 21.095023 ; d^3 = 9387.29
            n158_nearest_neighbors = [(0, 0, np.array([21,  2])), (0, 0, np.array([21, -2])), (0, 0, np.array([18, 11])), (0, 0, np.array([18, -11])), (0, 0, np.array([11, 18])), (0, 0, np.array([11, -18])), (0, 0, np.array([2, 21])), (0, 0, np.array([ 2, -21]))],
            # d = 21.189620 ; d^3 = 9514.14
            n159_nearest_neighbors = [(0, 0, np.array([20,  7])), (0, 0, np.array([20, -7])), (0, 0, np.array([7, 20])), (0, 0, np.array([ 7, -20]))],
            # d = 21.213203 ; d^3 = 9545.94
            n160_nearest_neighbors = [(0, 0, np.array([21,  3])), (0, 0, np.array([21, -3])), (0, 0, np.array([15, 15])), (0, 0, np.array([15, -15])), (0, 0, np.array([3, 21])), (0, 0, np.array([ 3, -21]))],
            # d = 21.260292 ; d^3 = 9609.65
            n161_nearest_neighbors = [(0, 0, np.array([16, 14])), (0, 0, np.array([16, -14])), (0, 0, np.array([14, 16])), (0, 0, np.array([14, -16]))],
            # d = 21.377558 ; d^3 = 9769.54
            n162_nearest_neighbors = [(0, 0, np.array([21,  4])), (0, 0, np.array([21, -4])), (0, 0, np.array([4, 21])), (0, 0, np.array([ 4, -21]))],
            # d = 21.400935 ; d^3 = 9801.63
            n163_nearest_neighbors = [(0, 0, np.array([17, 13])), (0, 0, np.array([17, -13])), (0, 0, np.array([13, 17])), (0, 0, np.array([13, -17]))],
            # d = 21.470911 ; d^3 = 9898.09
            n164_nearest_neighbors = [(0, 0, np.array([19, 10])), (0, 0, np.array([19, -10])), (0, 0, np.array([10, 19])), (0, 0, np.array([10, -19]))],
            # d = 21.540659 ; d^3 = 9994.87
            n165_nearest_neighbors = [(0, 0, np.array([20,  8])), (0, 0, np.array([20, -8])), (0, 0, np.array([8, 20])), (0, 0, np.array([ 8, -20]))],
            # d = 21.587033 ; d^3 = 10059.56
            n166_nearest_neighbors = [(0, 0, np.array([21,  5])), (0, 0, np.array([21, -5])), (0, 0, np.array([5, 21])), (0, 0, np.array([ 5, -21]))],
            # d = 21.633308 ; d^3 = 10124.39
            n167_nearest_neighbors = [(0, 0, np.array([18, 12])), (0, 0, np.array([18, -12])), (0, 0, np.array([12, 18])), (0, 0, np.array([12, -18]))],
            # d = 21.840330 ; d^3 = 10417.84
            n168_nearest_neighbors = [(0, 0, np.array([21,  6])), (0, 0, np.array([21, -6])), (0, 0, np.array([6, 21])), (0, 0, np.array([ 6, -21]))],
            # d = 21.931712 ; d^3 = 10549.15
            n169_nearest_neighbors = [(0, 0, np.array([20,  9])), (0, 0, np.array([20, -9])), (0, 0, np.array([16, 15])), (0, 0, np.array([16, -15])), (0, 0, np.array([15, 16])), (0, 0, np.array([15, -16])), (0, 0, np.array([9, 20])), (0, 0, np.array([ 9, -20]))],
            # d = 21.954498 ; d^3 = 10582.07
            n170_nearest_neighbors = [(0, 0, np.array([19, 11])), (0, 0, np.array([19, -11])), (0, 0, np.array([11, 19])), (0, 0, np.array([11, -19]))],
            # d = 22.000000 ; d^3 = 10648.00
            n171_nearest_neighbors = [(0, 0, np.array([22,  0])), (0, 0, np.array([0, 22]))],
            # d = 22.022716 ; d^3 = 10681.02
            n172_nearest_neighbors = [(0, 0, np.array([22,  1])), (0, 0, np.array([22, -1])), (0, 0, np.array([17, 14])), (0, 0, np.array([17, -14])), (0, 0, np.array([14, 17])), (0, 0, np.array([14, -17])), (0, 0, np.array([1, 22])), (0, 0, np.array([ 1, -22]))],
            # d = 22.090722 ; d^3 = 10780.27
            n173_nearest_neighbors = [(0, 0, np.array([22,  2])), (0, 0, np.array([22, -2])), (0, 0, np.array([2, 22])), (0, 0, np.array([ 2, -22]))],
            # d = 22.135944 ; d^3 = 10846.61
            n174_nearest_neighbors = [(0, 0, np.array([21,  7])), (0, 0, np.array([21, -7])), (0, 0, np.array([7, 21])), (0, 0, np.array([ 7, -21]))],
            # d = 22.203603 ; d^3 = 10946.38
            n175_nearest_neighbors = [(0, 0, np.array([22,  3])), (0, 0, np.array([22, -3])), (0, 0, np.array([18, 13])), (0, 0, np.array([18, -13])), (0, 0, np.array([13, 18])), (0, 0, np.array([13, -18])), (0, 0, np.array([3, 22])), (0, 0, np.array([ 3, -22]))],
            # d = 22.360680 ; d^3 = 11180.34
            n176_nearest_neighbors = [(0, 0, np.array([22,  4])), (0, 0, np.array([22, -4])), (0, 0, np.array([20, 10])), (0, 0, np.array([20, -10])), (0, 0, np.array([10, 20])), (0, 0, np.array([10, -20])), (0, 0, np.array([4, 22])), (0, 0, np.array([ 4, -22]))],
            # d = 22.472205 ; d^3 = 11348.46
            n177_nearest_neighbors = [(0, 0, np.array([21,  8])), (0, 0, np.array([21, -8])), (0, 0, np.array([19, 12])), (0, 0, np.array([19, -12])), (0, 0, np.array([12, 19])), (0, 0, np.array([12, -19])), (0, 0, np.array([8, 21])), (0, 0, np.array([ 8, -21]))],
            # d = 22.561028 ; d^3 = 11483.56
            n178_nearest_neighbors = [(0, 0, np.array([22,  5])), (0, 0, np.array([22, -5])), (0, 0, np.array([5, 22])), (0, 0, np.array([ 5, -22]))],
            # d = 22.627417 ; d^3 = 11585.24
            n179_nearest_neighbors = [(0, 0, np.array([16, 16])), (0, 0, np.array([16, -16]))],
            # d = 22.671568 ; d^3 = 11653.19
            n180_nearest_neighbors = [(0, 0, np.array([17, 15])), (0, 0, np.array([17, -15])), (0, 0, np.array([15, 17])), (0, 0, np.array([15, -17]))],
            # d = 22.803509 ; d^3 = 11857.82
            n181_nearest_neighbors = [(0, 0, np.array([22,  6])), (0, 0, np.array([22, -6])), (0, 0, np.array([18, 14])), (0, 0, np.array([18, -14])), (0, 0, np.array([14, 18])), (0, 0, np.array([14, -18])), (0, 0, np.array([6, 22])), (0, 0, np.array([ 6, -22]))],
            # d = 22.825424 ; d^3 = 11892.05
            n182_nearest_neighbors = [(0, 0, np.array([20, 11])), (0, 0, np.array([20, -11])), (0, 0, np.array([11, 20])), (0, 0, np.array([11, -20]))],
            # d = 22.847319 ; d^3 = 11926.30
            n183_nearest_neighbors = [(0, 0, np.array([21,  9])), (0, 0, np.array([21, -9])), (0, 0, np.array([9, 21])), (0, 0, np.array([ 9, -21]))],
            # d = 23.000000 ; d^3 = 12167.00
            n184_nearest_neighbors = [(0, 0, np.array([23,  0])), (0, 0, np.array([0, 23]))],
            # d = 23.021729 ; d^3 = 12201.52
            n185_nearest_neighbors = [(0, 0, np.array([23,  1])), (0, 0, np.array([23, -1])), (0, 0, np.array([19, 13])), (0, 0, np.array([19, -13])), (0, 0, np.array([13, 19])), (0, 0, np.array([13, -19])), (0, 0, np.array([1, 23])), (0, 0, np.array([ 1, -23]))],
            # d = 23.086793 ; d^3 = 12305.26
            n186_nearest_neighbors = [(0, 0, np.array([23,  2])), (0, 0, np.array([23, -2])), (0, 0, np.array([22,  7])), (0, 0, np.array([22, -7])), (0, 0, np.array([7, 22])), (0, 0, np.array([ 7, -22])), (0, 0, np.array([2, 23])), (0, 0, np.array([ 2, -23]))],
            # d = 23.194827 ; d^3 = 12478.82
            n187_nearest_neighbors = [(0, 0, np.array([23,  3])), (0, 0, np.array([23, -3])), (0, 0, np.array([3, 23])), (0, 0, np.array([ 3, -23]))],
            # d = 23.259407 ; d^3 = 12583.34
            n188_nearest_neighbors = [(0, 0, np.array([21, 10])), (0, 0, np.array([21, -10])), (0, 0, np.array([10, 21])), (0, 0, np.array([10, -21]))],
            # d = 23.323808 ; d^3 = 12688.15
            n189_nearest_neighbors = [(0, 0, np.array([20, 12])), (0, 0, np.array([20, -12])), (0, 0, np.array([12, 20])), (0, 0, np.array([12, -20]))],
            # d = 23.345235 ; d^3 = 12723.15
            n190_nearest_neighbors = [(0, 0, np.array([23,  4])), (0, 0, np.array([23, -4])), (0, 0, np.array([17, 16])), (0, 0, np.array([17, -16])), (0, 0, np.array([16, 17])), (0, 0, np.array([16, -17])), (0, 0, np.array([4, 23])), (0, 0, np.array([ 4, -23]))],
            # d = 23.409400 ; d^3 = 12828.35
            n191_nearest_neighbors = [(0, 0, np.array([22,  8])), (0, 0, np.array([22, -8])), (0, 0, np.array([8, 22])), (0, 0, np.array([ 8, -22]))],
            # d = 23.430749 ; d^3 = 12863.48
            n192_nearest_neighbors = [(0, 0, np.array([18, 15])), (0, 0, np.array([18, -15])), (0, 0, np.array([15, 18])), (0, 0, np.array([15, -18]))],
            # d = 23.537205 ; d^3 = 13039.61
            n193_nearest_neighbors = [(0, 0, np.array([23,  5])), (0, 0, np.array([23, -5])), (0, 0, np.array([5, 23])), (0, 0, np.array([ 5, -23]))],
            # d = 23.600847 ; d^3 = 13145.67
            n194_nearest_neighbors = [(0, 0, np.array([19, 14])), (0, 0, np.array([19, -14])), (0, 0, np.array([14, 19])), (0, 0, np.array([14, -19]))],
            # d = 23.706539 ; d^3 = 13323.08
            n195_nearest_neighbors = [(0, 0, np.array([21, 11])), (0, 0, np.array([21, -11])), (0, 0, np.array([11, 21])), (0, 0, np.array([11, -21]))],
            # d = 23.769729 ; d^3 = 13429.90
            n196_nearest_neighbors = [(0, 0, np.array([23,  6])), (0, 0, np.array([23, -6])), (0, 0, np.array([22,  9])), (0, 0, np.array([22, -9])), (0, 0, np.array([9, 22])), (0, 0, np.array([ 9, -22])), (0, 0, np.array([6, 23])), (0, 0, np.array([ 6, -23]))],
            # d = 23.853721 ; d^3 = 13572.77
            n197_nearest_neighbors = [(0, 0, np.array([20, 13])), (0, 0, np.array([20, -13])), (0, 0, np.array([13, 20])), (0, 0, np.array([13, -20]))],
            # d = 24.000000 ; d^3 = 13824.00
            n198_nearest_neighbors = [(0, 0, np.array([24,  0])), (0, 0, np.array([0, 24]))],
            # d = 24.020824 ; d^3 = 13860.02
            n199_nearest_neighbors = [(0, 0, np.array([24,  1])), (0, 0, np.array([24, -1])), (0, 0, np.array([1, 24])), (0, 0, np.array([ 1, -24]))],
            # d = 24.041631 ; d^3 = 13896.06
            n200_nearest_neighbors = [(0, 0, np.array([23,  7])), (0, 0, np.array([23, -7])), (0, 0, np.array([17, 17])), (0, 0, np.array([17, -17])), (0, 0, np.array([7, 23])), (0, 0, np.array([ 7, -23]))],
            # d = 24.083189 ; d^3 = 13968.25
            n201_nearest_neighbors = [(0, 0, np.array([24,  2])), (0, 0, np.array([24, -2])), (0, 0, np.array([18, 16])), (0, 0, np.array([18, -16])), (0, 0, np.array([16, 18])), (0, 0, np.array([16, -18])), (0, 0, np.array([2, 24])), (0, 0, np.array([ 2, -24]))],
            # d = 24.166092 ; d^3 = 14113.00
            n202_nearest_neighbors = [(0, 0, np.array([22, 10])), (0, 0, np.array([22, -10])), (0, 0, np.array([10, 22])), (0, 0, np.array([10, -22]))],
            # d = 24.186773 ; d^3 = 14149.26
            n203_nearest_neighbors = [(0, 0, np.array([24,  3])), (0, 0, np.array([24, -3])), (0, 0, np.array([21, 12])), (0, 0, np.array([21, -12])), (0, 0, np.array([12, 21])), (0, 0, np.array([12, -21])), (0, 0, np.array([3, 24])), (0, 0, np.array([ 3, -24]))],
            # d = 24.207437 ; d^3 = 14185.56
            n204_nearest_neighbors = [(0, 0, np.array([19, 15])), (0, 0, np.array([19, -15])), (0, 0, np.array([15, 19])), (0, 0, np.array([15, -19]))],
            # d = 24.331050 ; d^3 = 14403.98
            n205_nearest_neighbors = [(0, 0, np.array([24,  4])), (0, 0, np.array([24, -4])), (0, 0, np.array([4, 24])), (0, 0, np.array([ 4, -24]))],
            # d = 24.351591 ; d^3 = 14440.49
            n206_nearest_neighbors = [(0, 0, np.array([23,  8])), (0, 0, np.array([23, -8])), (0, 0, np.array([8, 23])), (0, 0, np.array([ 8, -23]))],
            # d = 24.413111 ; d^3 = 14550.21
            n207_nearest_neighbors = [(0, 0, np.array([20, 14])), (0, 0, np.array([20, -14])), (0, 0, np.array([14, 20])), (0, 0, np.array([14, -20]))],
            # d = 24.515301 ; d^3 = 14733.70
            n208_nearest_neighbors = [(0, 0, np.array([24,  5])), (0, 0, np.array([24, -5])), (0, 0, np.array([5, 24])), (0, 0, np.array([ 5, -24]))],
            # d = 24.596748 ; d^3 = 14881.03
            n209_nearest_neighbors = [(0, 0, np.array([22, 11])), (0, 0, np.array([22, -11])), (0, 0, np.array([11, 22])), (0, 0, np.array([11, -22]))],
            # d = 24.698178 ; d^3 = 15065.89
            n210_nearest_neighbors = [(0, 0, np.array([23,  9])), (0, 0, np.array([23, -9])), (0, 0, np.array([21, 13])), (0, 0, np.array([21, -13])), (0, 0, np.array([13, 21])), (0, 0, np.array([13, -21])), (0, 0, np.array([9, 23])), (0, 0, np.array([ 9, -23]))],
            # d = 24.738634 ; d^3 = 15140.04
            n211_nearest_neighbors = [(0, 0, np.array([24,  6])), (0, 0, np.array([24, -6])), (0, 0, np.array([6, 24])), (0, 0, np.array([ 6, -24]))],
            # d = 24.758837 ; d^3 = 15177.17
            n212_nearest_neighbors = [(0, 0, np.array([18, 17])), (0, 0, np.array([18, -17])), (0, 0, np.array([17, 18])), (0, 0, np.array([17, -18]))],
            # d = 24.839485 ; d^3 = 15325.96
            n213_nearest_neighbors = [(0, 0, np.array([19, 16])), (0, 0, np.array([19, -16])), (0, 0, np.array([16, 19])), (0, 0, np.array([16, -19]))],
            # d = 25.000000 ; d^3 = 15625.00
            n214_nearest_neighbors = [(0, 0, np.array([25,  0])), (0, 0, np.array([24,  7])), (0, 0, np.array([24, -7])), (0, 0, np.array([20, 15])), (0, 0, np.array([20, -15])), (0, 0, np.array([15, 20])), (0, 0, np.array([15, -20])), (0, 0, np.array([7, 24])), (0, 0, np.array([ 7, -24])), (0, 0, np.array([0, 25]))],
            # d = 25.019992 ; d^3 = 15662.51
            n215_nearest_neighbors = [(0, 0, np.array([25,  1])), (0, 0, np.array([25, -1])), (0, 0, np.array([1, 25])), (0, 0, np.array([ 1, -25]))],
            # d = 25.059928 ; d^3 = 15737.63
            n216_nearest_neighbors = [(0, 0, np.array([22, 12])), (0, 0, np.array([22, -12])), (0, 0, np.array([12, 22])), (0, 0, np.array([12, -22]))],
            # d = 25.079872 ; d^3 = 15775.24
            n217_nearest_neighbors = [(0, 0, np.array([25,  2])), (0, 0, np.array([25, -2])), (0, 0, np.array([23, 10])), (0, 0, np.array([23, -10])), (0, 0, np.array([10, 23])), (0, 0, np.array([10, -23])), (0, 0, np.array([2, 25])), (0, 0, np.array([ 2, -25]))],
            # d = 25.179357 ; d^3 = 15963.71
            n218_nearest_neighbors = [(0, 0, np.array([25,  3])), (0, 0, np.array([25, -3])), (0, 0, np.array([3, 25])), (0, 0, np.array([ 3, -25]))],
            # d = 25.238859 ; d^3 = 16077.15
            n219_nearest_neighbors = [(0, 0, np.array([21, 14])), (0, 0, np.array([21, -14])), (0, 0, np.array([14, 21])), (0, 0, np.array([14, -21]))],
            # d = 25.298221 ; d^3 = 16190.86
            n220_nearest_neighbors = [(0, 0, np.array([24,  8])), (0, 0, np.array([24, -8])), (0, 0, np.array([8, 24])), (0, 0, np.array([ 8, -24]))],
            # d = 25.317978 ; d^3 = 16228.82
            n221_nearest_neighbors = [(0, 0, np.array([25,  4])), (0, 0, np.array([25, -4])), (0, 0, np.array([4, 25])), (0, 0, np.array([ 4, -25]))],
            # d = 25.455844 ; d^3 = 16495.39
            n222_nearest_neighbors = [(0, 0, np.array([18, 18])), (0, 0, np.array([18, -18]))],
            # d = 25.495098 ; d^3 = 16571.81
            n223_nearest_neighbors = [(0, 0, np.array([25,  5])), (0, 0, np.array([25, -5])), (0, 0, np.array([23, 11])), (0, 0, np.array([23, -11])), (0, 0, np.array([19, 17])), (0, 0, np.array([19, -17])), (0, 0, np.array([17, 19])), (0, 0, np.array([17, -19])), (0, 0, np.array([11, 23])), (0, 0, np.array([11, -23])), (0, 0, np.array([5, 25])), (0, 0, np.array([ 5, -25]))],
            # d = 25.553865 ; d^3 = 16686.67
            n224_nearest_neighbors = [(0, 0, np.array([22, 13])), (0, 0, np.array([22, -13])), (0, 0, np.array([13, 22])), (0, 0, np.array([13, -22]))],
            # d = 25.612497 ; d^3 = 16801.80
            n225_nearest_neighbors = [(0, 0, np.array([20, 16])), (0, 0, np.array([20, -16])), (0, 0, np.array([16, 20])), (0, 0, np.array([16, -20]))],
            # d = 25.632011 ; d^3 = 16840.23
            n226_nearest_neighbors = [(0, 0, np.array([24,  9])), (0, 0, np.array([24, -9])), (0, 0, np.array([9, 24])), (0, 0, np.array([ 9, -24]))],
            # d = 25.709920 ; d^3 = 16994.26
            n227_nearest_neighbors = [(0, 0, np.array([25,  6])), (0, 0, np.array([25, -6])), (0, 0, np.array([6, 25])), (0, 0, np.array([ 6, -25]))],
            # d = 25.806976 ; d^3 = 17187.45
            n228_nearest_neighbors = [(0, 0, np.array([21, 15])), (0, 0, np.array([21, -15])), (0, 0, np.array([15, 21])), (0, 0, np.array([15, -21]))],
            # d = 25.942244 ; d^3 = 17459.13
            n229_nearest_neighbors = [(0, 0, np.array([23, 12])), (0, 0, np.array([23, -12])), (0, 0, np.array([12, 23])), (0, 0, np.array([12, -23]))],
            # d = 25.961510 ; d^3 = 17498.06
            n230_nearest_neighbors = [(0, 0, np.array([25,  7])), (0, 0, np.array([25, -7])), (0, 0, np.array([7, 25])), (0, 0, np.array([ 7, -25]))],
            # d = 26.000000 ; d^3 = 17576.00
            n231_nearest_neighbors = [(0, 0, np.array([26,  0])), (0, 0, np.array([24, 10])), (0, 0, np.array([24, -10])), (0, 0, np.array([10, 24])), (0, 0, np.array([10, -24])), (0, 0, np.array([0, 26]))],
            # d = 26.019224 ; d^3 = 17615.01
            n232_nearest_neighbors = [(0, 0, np.array([26,  1])), (0, 0, np.array([26, -1])), (0, 0, np.array([1, 26])), (0, 0, np.array([ 1, -26]))],
            # d = 26.076810 ; d^3 = 17732.23
            n233_nearest_neighbors = [(0, 0, np.array([26,  2])), (0, 0, np.array([26, -2])), (0, 0, np.array([22, 14])), (0, 0, np.array([22, -14])), (0, 0, np.array([14, 22])), (0, 0, np.array([14, -22])), (0, 0, np.array([2, 26])), (0, 0, np.array([ 2, -26]))],
            # d = 26.172505 ; d^3 = 17928.17
            n234_nearest_neighbors = [(0, 0, np.array([26,  3])), (0, 0, np.array([26, -3])), (0, 0, np.array([19, 18])), (0, 0, np.array([19, -18])), (0, 0, np.array([18, 19])), (0, 0, np.array([18, -19])), (0, 0, np.array([3, 26])), (0, 0, np.array([ 3, -26]))],
            # d = 26.248809 ; d^3 = 18085.43
            n235_nearest_neighbors = [(0, 0, np.array([25,  8])), (0, 0, np.array([25, -8])), (0, 0, np.array([20, 17])), (0, 0, np.array([20, -17])), (0, 0, np.array([17, 20])), (0, 0, np.array([17, -20])), (0, 0, np.array([8, 25])), (0, 0, np.array([ 8, -25]))],
            # d = 26.305893 ; d^3 = 18203.68
            n236_nearest_neighbors = [(0, 0, np.array([26,  4])), (0, 0, np.array([26, -4])), (0, 0, np.array([4, 26])), (0, 0, np.array([ 4, -26]))],
            # d = 26.400758 ; d^3 = 18401.33
            n237_nearest_neighbors = [(0, 0, np.array([24, 11])), (0, 0, np.array([24, -11])), (0, 0, np.array([21, 16])), (0, 0, np.array([21, -16])), (0, 0, np.array([16, 21])), (0, 0, np.array([16, -21])), (0, 0, np.array([11, 24])), (0, 0, np.array([11, -24]))],
            # d = 26.419690 ; d^3 = 18440.94
            n238_nearest_neighbors = [(0, 0, np.array([23, 13])), (0, 0, np.array([23, -13])), (0, 0, np.array([13, 23])), (0, 0, np.array([13, -23]))],
            # d = 26.476405 ; d^3 = 18559.96
            n239_nearest_neighbors = [(0, 0, np.array([26,  5])), (0, 0, np.array([26, -5])), (0, 0, np.array([5, 26])), (0, 0, np.array([ 5, -26]))],
            # d = 26.570661 ; d^3 = 18758.89
            n240_nearest_neighbors = [(0, 0, np.array([25,  9])), (0, 0, np.array([25, -9])), (0, 0, np.array([9, 25])), (0, 0, np.array([ 9, -25]))],
            # d = 26.627054 ; d^3 = 18878.58
            n241_nearest_neighbors = [(0, 0, np.array([22, 15])), (0, 0, np.array([22, -15])), (0, 0, np.array([15, 22])), (0, 0, np.array([15, -22]))],
            # d = 26.683328 ; d^3 = 18998.53
            n242_nearest_neighbors = [(0, 0, np.array([26,  6])), (0, 0, np.array([26, -6])), (0, 0, np.array([6, 26])), (0, 0, np.array([ 6, -26]))],
            # d = 26.832816 ; d^3 = 19319.63
            n243_nearest_neighbors = [(0, 0, np.array([24, 12])), (0, 0, np.array([24, -12])), (0, 0, np.array([12, 24])), (0, 0, np.array([12, -24]))],
            # d = 26.870058 ; d^3 = 19400.18
            n244_nearest_neighbors = [(0, 0, np.array([19, 19])), (0, 0, np.array([19, -19]))],
            # d = 26.907248 ; d^3 = 19480.85
            n245_nearest_neighbors = [(0, 0, np.array([20, 18])), (0, 0, np.array([20, -18])), (0, 0, np.array([18, 20])), (0, 0, np.array([18, -20]))],
            # d = 26.925824 ; d^3 = 19521.22
            n246_nearest_neighbors = [(0, 0, np.array([26,  7])), (0, 0, np.array([26, -7])), (0, 0, np.array([25, 10])), (0, 0, np.array([25, -10])), (0, 0, np.array([23, 14])), (0, 0, np.array([23, -14])), (0, 0, np.array([14, 23])), (0, 0, np.array([14, -23])), (0, 0, np.array([10, 25])), (0, 0, np.array([10, -25])), (0, 0, np.array([7, 26])), (0, 0, np.array([ 7, -26]))],
            # d = 27.000000 ; d^3 = 19683.00
            n247_nearest_neighbors = [(0, 0, np.array([27,  0])), (0, 0, np.array([0, 27]))],
            # d = 27.018512 ; d^3 = 19723.51
            n248_nearest_neighbors = [(0, 0, np.array([27,  1])), (0, 0, np.array([27, -1])), (0, 0, np.array([21, 17])), (0, 0, np.array([21, -17])), (0, 0, np.array([17, 21])), (0, 0, np.array([17, -21])), (0, 0, np.array([1, 27])), (0, 0, np.array([ 1, -27]))],
            # d = 27.073973 ; d^3 = 19845.22
            n249_nearest_neighbors = [(0, 0, np.array([27,  2])), (0, 0, np.array([27, -2])), (0, 0, np.array([2, 27])), (0, 0, np.array([ 2, -27]))],
            # d = 27.166155 ; d^3 = 20048.62
            n250_nearest_neighbors = [(0, 0, np.array([27,  3])), (0, 0, np.array([27, -3])), (0, 0, np.array([3, 27])), (0, 0, np.array([ 3, -27]))],
            # d = 27.202941 ; d^3 = 20130.18
            n251_nearest_neighbors = [(0, 0, np.array([26,  8])), (0, 0, np.array([26, -8])), (0, 0, np.array([22, 16])), (0, 0, np.array([22, -16])), (0, 0, np.array([16, 22])), (0, 0, np.array([16, -22])), (0, 0, np.array([8, 26])), (0, 0, np.array([ 8, -26]))],
            # d = 27.294688 ; d^3 = 20334.54
            n252_nearest_neighbors = [(0, 0, np.array([27,  4])), (0, 0, np.array([27, -4])), (0, 0, np.array([24, 13])), (0, 0, np.array([24, -13])), (0, 0, np.array([13, 24])), (0, 0, np.array([13, -24])), (0, 0, np.array([4, 27])), (0, 0, np.array([ 4, -27]))],
            # d = 27.313001 ; d^3 = 20375.50
            n253_nearest_neighbors = [(0, 0, np.array([25, 11])), (0, 0, np.array([25, -11])), (0, 0, np.array([11, 25])), (0, 0, np.array([11, -25]))],
            # d = 27.459060 ; d^3 = 20704.13
            n254_nearest_neighbors = [(0, 0, np.array([27,  5])), (0, 0, np.array([27, -5])), (0, 0, np.array([23, 15])), (0, 0, np.array([23, -15])), (0, 0, np.array([15, 23])), (0, 0, np.array([15, -23])), (0, 0, np.array([5, 27])), (0, 0, np.array([ 5, -27]))],
            # d = 27.513633 ; d^3 = 20827.82
            n255_nearest_neighbors = [(0, 0, np.array([26,  9])), (0, 0, np.array([26, -9])), (0, 0, np.array([9, 26])), (0, 0, np.array([ 9, -26]))],
            # d = 27.586228 ; d^3 = 20993.12
            n256_nearest_neighbors = [(0, 0, np.array([20, 19])), (0, 0, np.array([20, -19])), (0, 0, np.array([19, 20])), (0, 0, np.array([19, -20]))],
            # d = 27.658633 ; d^3 = 21158.85
            n257_nearest_neighbors = [(0, 0, np.array([27,  6])), (0, 0, np.array([27, -6])), (0, 0, np.array([21, 18])), (0, 0, np.array([21, -18])), (0, 0, np.array([18, 21])), (0, 0, np.array([18, -21])), (0, 0, np.array([6, 27])), (0, 0, np.array([ 6, -27]))],
            # d = 27.730849 ; d^3 = 21325.02
            n258_nearest_neighbors = [(0, 0, np.array([25, 12])), (0, 0, np.array([25, -12])), (0, 0, np.array([12, 25])), (0, 0, np.array([12, -25]))],
            # d = 27.784888 ; d^3 = 21449.93
            n259_nearest_neighbors = [(0, 0, np.array([24, 14])), (0, 0, np.array([24, -14])), (0, 0, np.array([14, 24])), (0, 0, np.array([14, -24]))],
            # d = 27.802878 ; d^3 = 21491.62
            n260_nearest_neighbors = [(0, 0, np.array([22, 17])), (0, 0, np.array([22, -17])), (0, 0, np.array([17, 22])), (0, 0, np.array([17, -22]))],
            # d = 27.856777 ; d^3 = 21616.86
            n261_nearest_neighbors = [(0, 0, np.array([26, 10])), (0, 0, np.array([26, -10])), (0, 0, np.array([10, 26])), (0, 0, np.array([10, -26]))],
            # d = 27.892651 ; d^3 = 21700.48
            n262_nearest_neighbors = [(0, 0, np.array([27,  7])), (0, 0, np.array([27, -7])), (0, 0, np.array([7, 27])), (0, 0, np.array([ 7, -27]))],
            # d = 28.000000 ; d^3 = 21952.00
            n263_nearest_neighbors = [(0, 0, np.array([28,  0])), (0, 0, np.array([0, 28]))],
            # d = 28.017851 ; d^3 = 21994.01
            n264_nearest_neighbors = [(0, 0, np.array([28,  1])), (0, 0, np.array([28, -1])), (0, 0, np.array([23, 16])), (0, 0, np.array([23, -16])), (0, 0, np.array([16, 23])), (0, 0, np.array([16, -23])), (0, 0, np.array([1, 28])), (0, 0, np.array([ 1, -28]))],
            # d = 28.071338 ; d^3 = 22120.21
            n265_nearest_neighbors = [(0, 0, np.array([28,  2])), (0, 0, np.array([28, -2])), (0, 0, np.array([2, 28])), (0, 0, np.array([ 2, -28]))],
            # d = 28.160256 ; d^3 = 22331.08
            n266_nearest_neighbors = [(0, 0, np.array([28,  3])), (0, 0, np.array([28, -3])), (0, 0, np.array([27,  8])), (0, 0, np.array([27, -8])), (0, 0, np.array([8, 27])), (0, 0, np.array([ 8, -27])), (0, 0, np.array([3, 28])), (0, 0, np.array([ 3, -28]))],
            # d = 28.178006 ; d^3 = 22373.34
            n267_nearest_neighbors = [(0, 0, np.array([25, 13])), (0, 0, np.array([25, -13])), (0, 0, np.array([13, 25])), (0, 0, np.array([13, -25]))],
            # d = 28.231188 ; d^3 = 22500.26
            n268_nearest_neighbors = [(0, 0, np.array([26, 11])), (0, 0, np.array([26, -11])), (0, 0, np.array([11, 26])), (0, 0, np.array([11, -26]))],
            # d = 28.284271 ; d^3 = 22627.42
            n269_nearest_neighbors = [(0, 0, np.array([28,  4])), (0, 0, np.array([28, -4])), (0, 0, np.array([20, 20])), (0, 0, np.array([20, -20])), (0, 0, np.array([4, 28])), (0, 0, np.array([ 4, -28]))],
            # d = 28.301943 ; d^3 = 22669.86
            n270_nearest_neighbors = [(0, 0, np.array([24, 15])), (0, 0, np.array([24, -15])), (0, 0, np.array([15, 24])), (0, 0, np.array([15, -24]))],
            # d = 28.319605 ; d^3 = 22712.32
            n271_nearest_neighbors = [(0, 0, np.array([21, 19])), (0, 0, np.array([21, -19])), (0, 0, np.array([19, 21])), (0, 0, np.array([19, -21]))],
            # d = 28.425341 ; d^3 = 22967.68
            n272_nearest_neighbors = [(0, 0, np.array([22, 18])), (0, 0, np.array([22, -18])), (0, 0, np.array([18, 22])), (0, 0, np.array([18, -22]))],
            # d = 28.442925 ; d^3 = 23010.33
            n273_nearest_neighbors = [(0, 0, np.array([28,  5])), (0, 0, np.array([28, -5])), (0, 0, np.array([5, 28])), (0, 0, np.array([ 5, -28]))],
            # d = 28.460499 ; d^3 = 23053.00
            n274_nearest_neighbors = [(0, 0, np.array([27,  9])), (0, 0, np.array([27, -9])), (0, 0, np.array([9, 27])), (0, 0, np.array([ 9, -27]))],
            # d = 28.600699 ; d^3 = 23395.37
            n275_nearest_neighbors = [(0, 0, np.array([23, 17])), (0, 0, np.array([23, -17])), (0, 0, np.array([17, 23])), (0, 0, np.array([17, -23]))],
            # d = 28.635642 ; d^3 = 23481.23
            n276_nearest_neighbors = [(0, 0, np.array([28,  6])), (0, 0, np.array([28, -6])), (0, 0, np.array([26, 12])), (0, 0, np.array([26, -12])), (0, 0, np.array([12, 26])), (0, 0, np.array([12, -26])), (0, 0, np.array([6, 28])), (0, 0, np.array([ 6, -28]))],
            # d = 28.653098 ; d^3 = 23524.19
            n277_nearest_neighbors = [(0, 0, np.array([25, 14])), (0, 0, np.array([25, -14])), (0, 0, np.array([14, 25])), (0, 0, np.array([14, -25]))],
            # d = 28.792360 ; d^3 = 23868.87
            n278_nearest_neighbors = [(0, 0, np.array([27, 10])), (0, 0, np.array([27, -10])), (0, 0, np.array([10, 27])), (0, 0, np.array([10, -27]))],
            # d = 28.844410 ; d^3 = 23998.55
            n279_nearest_neighbors = [(0, 0, np.array([24, 16])), (0, 0, np.array([24, -16])), (0, 0, np.array([16, 24])), (0, 0, np.array([16, -24]))],
            # d = 28.861739 ; d^3 = 24041.83
            n280_nearest_neighbors = [(0, 0, np.array([28,  7])), (0, 0, np.array([28, -7])), (0, 0, np.array([7, 28])), (0, 0, np.array([ 7, -28]))],
            # d = 29.000000 ; d^3 = 24389.00
            n281_nearest_neighbors = [(0, 0, np.array([29,  0])), (0, 0, np.array([21, 20])), (0, 0, np.array([21, -20])), (0, 0, np.array([20, 21])), (0, 0, np.array([20, -21])), (0, 0, np.array([0, 29]))]
        )

class CircleVertices(Lattice):
    """NOTE: Auxilliary for "Circle" only. Contains just the sites/vertices, for finding the pairs.
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    N : int
        Number of points. 
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2

    def __init__(self, Lx, Ly, sites, N=2, **kwargs):
        self.Lu = N
        sites = _parse_sites(sites, N) 

        x = 1/2/np.sin(np.pi/N)*np.cos(2*np.pi*np.arange(N)/N)
        y = 1/2/np.sin(np.pi/N)*np.sin(2*np.pi*np.arange(N)/N)
        pos = np.stack((x, y), axis=-1)

        basis = [[100/np.sin(np.pi/N), 0], [0, 100/np.sin(np.pi/N)]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


class Circle(Lattice):
    """A circular lattice with a custom number of points N (kwargs)
    .. plot ::
        import matplotlib.pyplot as plt
        from tenpy.models import lattice
        plt.figure(figsize=(5, 4))
        ax = plt.gca()
        lat = lattice.Kagome(4, 4, None, bc='periodic')
        lat.plot_coupling(ax, linewidth=3.)
        lat.plot_order(ax, linestyle=':')
        lat.plot_sites(ax)
        lat.plot_basis(ax, origin=-0.25*(lat.basis[0] + lat.basis[1]))
        ax.set_aspect('equal')
        ax.set_xlim(-1)
        ax.set_ylim(-1)
        plt.show()
    Parameters
    ----------
    Lx, Ly : int
        The length in each direction.
    sites : (list of) :class:`~tenpy.networks.site.Site`
        The two local lattice sites making the `unit_cell` of the :class:`Lattice`.
        If only a single :class:`~tenpy.networks.site.Site` is given, it is used for both sites.
    N : int
        Number of points. 
    **kwargs :
        Additional keyword arguments given to the :class:`Lattice`.
        `basis`, `pos` and `pairs` are set accordingly.
    """
    dim = 2

    def __init__(self, Lx, Ly, sites, N=2, **kwargs):
        self.Lu = N
        sites = _parse_sites(sites, N) 

        x = 1/2/np.sin(np.pi/N)*np.cos(2*np.pi*np.arange(N)/N)
        y = 1/2/np.sin(np.pi/N)*np.sin(2*np.pi*np.arange(N)/N)
        pos = np.stack((x, y), axis=-1)

        basis = [[100/np.sin(np.pi/N), 0], [0, 100/np.sin(np.pi/N)]]
        
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)

        ### Construct the pairs using the CircleVertices class as an auxilliary lattice
        aux_lat = CircleVertices(Lx=1, Ly=1, sites=None, N=N, bc=['open', 'open'])
        # Use the tenpy method to get them
        aux_dict = aux_lat.find_coupling_pairs(cutoff=1/np.sin(np.pi/N)+1)

        # The aux dictionary keys = coupling distance; we change to "nearest_neighbor" style
        new_dict = {}
        nNN = 0
        for old_key in aux_dict.keys():
            new_key = 'nearest_neighbors' if nNN==0 else 'n%d_nearest_neighbors' %nNN
            new_dict[new_key]=aux_dict[old_key]
            nNN+=1
        
        kwargs['pairs'] = new_dict
        
        Lattice.__init__(self, [Lx, Ly], sites, **kwargs)
