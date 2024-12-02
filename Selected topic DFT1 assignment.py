# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:09:57 2024

@author: abedn
"""

import numpy as np 
import matplotlib.pyplot as plt 

plt.close('all')
L=20
x=np.linspace(0, L, 1000)

def density(n):
    den=0
    for i in range(n+1):
        den+=(2/L)*(np.sin(i*np.pi*x/L))**2
    return den

def norm(A):
    return A/max(A)

n1=norm(density(1))
n2=norm(density(2))
n3=norm(density(10))
n4=norm(density(100))

plt.figure()
plt.subplot(221)
plt.plot(x, n1, "b", label="n=1")
plt.ylabel("Total Density (a.u)")
plt.xlabel("Distance L (a.u)")
plt.legend()
plt.grid()

plt.subplot(222)
plt.plot(x, n2, "r", label="n=2")
plt.ylabel("Total Density (a.u)")
plt.xlabel("Distance L (a.u)")
plt.legend()
plt.grid()

plt.subplot(223)
plt.plot(x, n3, "b", label="n=10")
plt.ylabel("Total Density (a.u)")
plt.xlabel("Distance L (a.u)")
plt.legend()
plt.grid()

plt.subplot(224)
plt.plot(x, n4, "r", label="n=100")
plt.ylabel("Total Density (a.u)")
plt.xlabel("Distance L (a.u)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()