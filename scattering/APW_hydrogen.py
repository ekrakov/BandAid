import numpy as np
from scipy.special import sph_harm, genlaguerre, factorial,spherical_jn,eval_legendre,lpmv
import matplotlib.pyplot as plt
import argparse
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Constants:
    def __init__(self, a0=5.29e-11, a=9e-10, f=2, rmt=2.5e-10,a1=[0,1],a2=[1,0]):
        """
        Define constants 
        :param a0: Bohr radius (default: 5.29e-11)
        :param a: periodic spacing between lattices (default: 9e-10)
        :param f: Recipocal lattice frequency. G= 2pi*f/a
        :param rmt: muffin tin radius (default: rmt=2.5e-10)
        """
        self.a0 = a0       # Bohr radius
        self.a = a         # Lattice constants
        self.f = f         # Frequency factor
        self.rmt = rmt     # Matching radius
        self.a1 = a1 
        self.a2= a2 
    def recip_lat(self):
        """Calculate the reciprocal lattice vector G"""
        return (self.f * 2 * np.pi) / self.a

    def full_pw(self,r):
        """
        Function to compute full plane wave --still need to update for different directions 
        :param r: vector of plane wave propagation 
        :return: full plane wave e^(i G dot r) for a particular recip. lattice vector G 
        """
        G = self.recip_lat()
        return np.real(np.exp(1j*G*r))

    def expand_pw(self, l, m, r, X, Y):
        """
        Function to compute expanded plane wave 
        :param l: Angular quantum number 
        :param m: Magnetic quantum number
        :param r: Radius
        :return: full plane wave e^(i G dot r) for a particular recip. lattice vector G 
        """
        G = self.recip_lat()
    
        # sherical harmonic value 
        sph_harm_value = sph_harm(m, l, np.arctan2(Y,X),np.pi/2)
        exp_pw = 4*np.pi*(1j**l) * sph_harm_value * sph_harm(m, l, np.arctan2(G*Y,G*X*0),np.pi/2) * spherical_jn(l, G * r)
           
        # exp_pw = (1j**l) * sph_harm_value * sph_harm(m, l, np.arctan2(theta[0],2*np.pi*4/theta[1]),phi) * spherical_jn(l, G * r)
        #  4*np.pi*(1j**l) * sph_harm_value * sph_harm(m, l, np.arctan2(2*np.pi*2/theta[1],2**np.pi*2/theta[0]),0) * spherical_jn(l, G * r)
        return exp_pw

    
constants = Constants()


def hydrogen_wavefunction(n, l, m, r, constants): 
    """
    Function to compute hydrogen wavefunctions.
    :param n: Principal quantum number
    :param l: Angular quantum number
    :param m: Magnetic quantum number
    :param r: Radius
    :a0: Bohr radius (default: 5.29e-11)
    """
    # Normalization factor
    norm = np.sqrt((2.0/n/constants.a0)**3 * factorial(n-l-1) / (2.0 * n * factorial(n+l)))

    # Radial part
    rho = 2.0 * r / n / constants.a0
    radial = np.exp(-rho / 2) * rho**l * genlaguerre(n-l-1, 2*l+1)(rho)

    # Angular part
    return norm * radial


def matched(n,l, m, r, X, Y,constants):
    """
    Function to match wavefunctions.
    :param n: Principal quantum number
    :param l: Angular quantum number
    :param m: Magnetic quantum number
    :param r: Radius
    :param theta: Azimuthal angle
    :param phi: Polar angle
    :return: Matched wavefunction value
    """
    # plane wave expansion at the muffin tin radius (rmt)
    match_pw=constants.expand_pw(l,m,constants.rmt,X, Y)

    #hydrogen wavefunction at RMT 
    match_hwf=hydrogen_wavefunction(n, l, m, constants.rmt, constants)

    # full function for the radial component matched at RMT
    Rmatch = (match_pw * hydrogen_wavefunction(n, l, m, r,constants))/ match_hwf

    return Rmatch


def totalsum(lmax,r,X,Y,constants):
    n=lmax+1
    """
    Function to sum over all l and m quantum numbers. 
    Nothing makes me cry like using a for loop in python, but genlaguerre does not take array values. 
    I can make my own later. For now, it is a tad slower than it should be. 
    :param lmax: Maximum angular quantum number 
    :param r: radius
    """
    plane_w=4*np.pi*sum(constants.expand_pw(l, m,r,X,Y) for l in range(lmax + 1) for m in range(-l, l + 1))
    # full_solution=4*np.pi*sum(np.real(matched(n,l, m,r,theta, phi,constants)) for l in range(lmax + 1) for m in range(-l, l + 1))
    full_solution=sum(np.real(matched(n,l, m,r,X,Y,constants)) for l in range(lmax + 1) for m in range(-l, l + 1))

    
    return plane_w,full_solution


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(
                    description='Plotting to verify APW  basis matches interstitial for a given angular cutoff',
                    epilog='See Martin ch 16 for more information on the APW plotting. There is also a mathematica notebook in this folder ')

    # Add an argument for LMAX
    parser.add_argument("lmax_values", type=int, nargs='+', help="List of LMAX values to compare fitting")

    args = parser.parse_args()
    # setting the range for plotting 
    up_range=constants.a0*10
    low_range=-constants.a0*10
    itter=.1*constants.a0

 
    x = np.arange(low_range, up_range, itter)
    y = np.arange(low_range, up_range,itter)
    X, Y = np.meshgrid(x, y, indexing='ij')

    def generate_plot(lmax,X,Y):
        # create meshgrid for 3D plotting

        # evaluate the full solution. Convert to cartesian cordinates. It is messy right now (crying)
        pw,full_solution = totalsum(lmax,np.sqrt(X**2+Y**2),X,Y,constants)
        # plane_wave = constants.full_pw(np.cos(np.arctan2(X,Y))*np.sqrt(X**2+Y**2))
        plane_wave = constants.full_pw(Y)

        radial_mask = np.sqrt(X ** 2 + Y ** 2)
        # Create a mask for so the plane wave is true outside of the muffin tin and the atomic solutions is true inside the muffin tin.
        mask_pw = radial_mask >= constants.rmt
        mask_atom = radial_mask <= constants.rmt
        # mask_atom = radial_mask <= constants.rmt
        # Applying the mask. There is definately a better way to do this :,(
        RMTplane_wave = np.where(mask_pw, plane_wave, np.nan)
        RMT_atomic= np.where(mask_atom, full_solution, np.nan)

        return RMTplane_wave, RMT_atomic,pw


num_lmax = len(args.lmax_values)

fig = make_subplots(rows=1, cols=num_lmax, subplot_titles=[f"LMAX={lmax}" for lmax in args.lmax_values],
                            specs=[[{'type': 'surface'}] *num_lmax])

for i, lmax in enumerate(args.lmax_values, start=1):
    RMTplane_wave, RMT_atomic,pw= generate_plot(lmax,X,Y)
    pw=np.real(pw)

   
    fig.add_trace(go.Surface(x=X, y=Y, z=RMTplane_wave,showscale=False), row=1, col=i),
    fig.add_trace(go.Surface(x=X, y=Y, z=RMT_atomic,showscale=False,colorscale="sunset"), row=1, col=i)
    # fig.add_trace(go.Surface(x=X, y=Y, z=pw,showscale=False,colorscale="sunset"), row=1, col=i)

 
fig.update_layout(height=600, width=600*num_lmax, title_text="Subplots for Different LMAX Values")

fig.show()

