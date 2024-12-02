import numpy as np
import matplotlib.pyplot as plt

# plt.style.use("~/.matplotlib/styles/style.mplstyle")


def get_bound(E,a,b,V,m=1):
    kappa=np.sqrt(2*m*E)
    q= np.sqrt(2*m*(E-V))
    t1 = np.cos((a-b)*kappa)*np.cosh(q*b)
    t2 = -((q**2-kappa**2)/(2*kappa*q))*np.sin(kappa*(a-b))*np.sinh(q*b)
    val=t1+t2
    return -np.arccos(val), np.arccos(val) 

def get_scatter(E,a,b,V,m=1):
    q=np.sqrt(2*m*(V+E))
    kappa=np.sqrt(2*m*E)
    t1=np.cos(q*b)*(np.cos(kappa*(a-b)))
    t2=-((kappa**2+q**2)/(2*kappa*q))*np.sin(q*b)*np.sin(kappa*(a-b))

    val=t1+t2
    return -np.arccos(val), np.arccos(val) 

# get bound band
E_bound_all = np.linspace(0, 50, 100000) 
k_bound_all = np.array([get_bound(E,1,.01,.1) for E in E_bound_all])

# get scattering band
E_scatter_all = np.linspace(0, 50, 100000) 
k_scatter_all = np.array([get_scatter(E,1,.1,3) for E in E_scatter_all])

# get zero from both 
k_scatter_zero = np.array([get_scatter(E,1,0,0) for E in E_scatter_all])
k_bound_zero = np.array([get_bound(E,1,0,0) for E in E_bound_all])


#plt.plot(k_scatter,E_scatter,color='purple')
plt.plot(k_bound_all,E_bound_all,color='pink')
# plt.plot(k_bound_zero,E_bound_all,color='orange')

plt.plot(k_scatter_all,E_scatter_all,color='purple')
# plt.plot(k_scatter_zero,E_scatter_all,color='black')


plt.show()


