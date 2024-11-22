# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:22:47 2024

@author: abedn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Given values | dimension
I0 = 4 * 1e14  # W/sm^2
Ip = 21.5  # eV
lmbda = 800  # in nm
c = 299.792458  # in nm/fs

# Ponderomotive energy
def P_motive(I_0):
    # I_0 in W/sm^2 and lambda in mkm and returns eV
    return 9.33 * 1e-14 * I_0 * (lmbda * 1e-3) ** 2

Up = P_motive(I0)

# Omega in fs^-1
omega = 2 * np.pi * c / lmbda

# hbar in eV*fs dimension
hbar = 0.6582
# Omega in terms of eV
omega_eV = omega * hbar

# Data to save:
loc_min_real_1 = []
loc_min_im_1 = []
loc_min_real_2 = []
loc_min_im_2 = []

# Saddle point function
def saddle_point(N, om_eV, Ip, Up, PR_re, PR_im):
    pR_re, pR_im = np.meshgrid(PR_re, PR_im)
    gam = np.sqrt(Ip / (2 * Up))
    gam_N = np.sqrt((N * om_eV - Ip) / (2 * Up))

    p1 = np.cosh(pR_im) * np.sin(pR_re) + gam_N
    p2 = np.sinh(pR_im) * np.cos(pR_re)

    gam_t = gam + p2
    P = np.square(p1) + np.square(gam_t) + 1
    D = np.sqrt(P ** 2 - 4 * p1 ** 2)

    pI_re = np.arcsin(np.sqrt((P - D) / 2))
    pI_im = np.arccosh(np.sqrt((P + D) / 2))

    F1 = p1 * (pR_re - pI_re) - p2 * (pR_im - pI_im) - np.cos(pI_re) * np.cosh(pI_im) + np.cosh(pR_im) * np.cos(pR_re)
    F2 = p1 * (pR_im - pI_im) + p2 * (pR_re - pI_re) + np.sin(pI_re) * np.sinh(pI_im) - np.sinh(pR_im) * np.sin(pR_re)

    return np.square(F1) + np.square(F2)

dp = 0.0002

# In radians
pR_re = np.arange(0.1, 1.0 + dp, dp) * 2 * np.pi
pR_im = np.arange(-0.1, 0.1 + dp, dp) * 2 * np.pi

for N in range(20, 80, 2):
    F = saddle_point(N, omega_eV, Ip, Up, pR_re, pR_im)

    # Grid of real and imaginary return time
    tr_re, tr_im = np.meshgrid(pR_re, pR_im)

    # In terms of fs
    tr_re, tr_im = tr_re / omega, tr_im / omega

    # Find local minima in the F array
    local_minima = ndimage.label(F == ndimage.minimum_filter(F, size=(300, 300)))[0]

    # Find the two smallest local minima
    unique_labels, counts = np.unique(local_minima, return_counts=True)
    sorted_labels = unique_labels[np.argsort(counts)]
    smallest_minima = sorted_labels[0:2]

    # Get the coordinates of the two smallest local minima
    minima_coords = []
    for label in smallest_minima:
        coords = np.argwhere(local_minima == label)
        minima_coords.append(coords[0])

    # Extract the corresponding tr_re and tr_im values
    minima_tr_re = [tr_re[coord[0], coord[1]] for coord in minima_coords]
    minima_tr_im = [tr_im[coord[0], coord[1]] for coord in minima_coords]

    # Print the corresponding tr_re and tr_im values of the local minima
    for i, (re, im) in enumerate(zip(minima_tr_re, minima_tr_im)):
        print(f'Local Minimum {i + 1}: tr_re = {re}, tr_im = {im}')
        if i == 0:
            loc_min_real_1.append(re)
            loc_min_im_1.append(im)
        if i == 1:
            loc_min_real_2.append(re)
            loc_min_im_2.append(im)

# Save the array to a text file
np.savetxt("final_real_1.txt", np.array(loc_min_real_1), delimiter='\n')
np.savetxt("final_real_2.txt", np.array(loc_min_real_2), delimiter='\n')

N = np.arange(20, 80, 2)
plt.figure(6)
plt.plot(loc_min_real_1, N, "b--")
plt.plot(loc_min_real_2, N, "g--")
plt.xlabel("return time, $t_r$, fs")
plt.ylabel("Harmonic order N")
plt.show()


# Data to save:
itime_min_real_1 = []
itime_min_real_2 = []

# Saddle point function in terms of ionization time
def ion_saddle_point(phi, Ip, Up, PI_re, PI_im):
    pI_re, pI_im = np.meshgrid(PI_re, PI_im)
    gam = np.sqrt(Ip / (2 * Up))

    # p1 and p2 in terms of ionization time
    p1 = np.cosh(pI_im) * np.sin(pI_re)
    p2 = np.sinh(pI_im) * np.cos(pI_re) - gam

    # Return times in terms of ionization time:
    F1 = p1 * (phi - pI_re) + p2 * pI_im - np.cos(pI_re) * np.cosh(pI_im) + np.cos(phi)
    F2 = -p1 * pI_im + p2 * (phi - pI_im) + np.sin(pI_re) * np.sinh(pI_im)

    return np.square(F1) + np.square(F2)

# Function to find local minima
def find_local_minima(array):
    local_minima = ndimage.label(array == ndimage.minimum_filter(array, size=(300, 300)))[0]
    return local_minima

dp = 0.0002

# In radians
pI_re = np.arange(0.1, 1.0 + dp, dp) * np.pi / 2
pI_im = np.arange(0.1, 0.3 + dp, dp) * np.pi / 2

t2 = []
file_path = 'final_real_2.txt'
try:
    with open(file_path, 'r') as file:
        for line in file:
            try:
                number = float(line)
                t2.append(number)
            except ValueError:
                print(f"Skipping non-numeric line: {line}")
except FileNotFoundError:
    print(f"File '{file_path}' not found.")

t1 = []
file_path = 'final_real_1.txt'
try:
    with open(file_path, 'r') as file:
        for line in file:
            try:
                number = float(line)
                t1.append(number)
            except ValueError:
                print(f"Skipping non-numeric line: {line}")
except FileNotFoundError:
    print(f"File '{file_path}' not found.")

Phi1 = [x * omega for x in t1]
Phi2 = [x * omega for x in t2]

for phi in Phi1:
    F = ion_saddle_point(phi, Ip, Up, pI_re, pI_im)

    # Grid of real and imaginary ionization time
    tI_re, tI_im = np.meshgrid(pI_re, pI_im)

    # In terms of fs
    tI_re, tI_im = tI_re / omega, tI_im / omega

    # Find local minima in the F array
    local_minima = find_local_minima(F)

    # Find the two smallest local minima
    unique_labels, counts = np.unique(local_minima, return_counts=True)
    sorted_labels = unique_labels[np.argsort(counts)]
    smallest_minima = sorted_labels[0:2]  # Excluding label 0 (background)

    # Get the coordinates of the two smallest local minima
    minima_coords = []
    for label in smallest_minima:
        coords = np.argwhere(local_minima == label)
        minima_coords.append(coords[0])

    # Extract the corresponding tI_re and tI_im values
    minima_phiI_re = [tI_re[coord[0], coord[1]] for coord in minima_coords]
    minima_phiI_im = [tI_im[coord[0], coord[1]] for coord in minima_coords]

    # Print the corresponding tI_re and tI_im values of the local minima
    for i, (re, im) in enumerate(zip(minima_phiI_re, minima_phiI_im)):
        print(f'Local Minimum {i + 1}: tI_re = {re}, tI_im = {im}')
        if i == 0:
            itime_min_real_1.append(re)

for phi in Phi2:
    F = ion_saddle_point(phi, Ip, Up, pI_re, pI_im)

    # Grid of real and imaginary ionization time
    tI_re, tI_im = np.meshgrid(pI_re, pI_im)

    # In terms of fs
    tI_re, tI_im = tI_re / omega, tI_im / omega

    # Find local minima in the F array
    local_minima = find_local_minima(F)

    # Find the two smallest local minima
    unique_labels, counts = np.unique(local_minima, return_counts=True)
    sorted_labels = unique_labels[np.argsort(counts)]
    smallest_minima = sorted_labels[0:2]  # Excluding label 0 (background)

    # Get the coordinates of the two smallest local minima
    minima_coords = []
    for label in smallest_minima:
        coords = np.argwhere(local_minima == label)
        minima_coords.append(coords[0])

    # Extract the corresponding tI_re and tI_im values
    minima_phiI_re = [tI_re[coord[0], coord[1]] for coord in minima_coords]
    minima_phiI_im = [tI_im[coord[0], coord[1]] for coord in minima_coords]

    # Print the corresponding tI_re and tI_im values of the local minima
    for i, (re, im) in enumerate(zip(minima_phiI_re, minima_phiI_im)):
       # print(f'Local Minimum {i + 1}: tI_re = {re}, tI_im = {im}')
        if i == 0:
            itime_min_real_2.append(re)

# Save the array to a text file
np.savetxt("ion_time_real_1.txt", np.array(itime_min_real_1), delimiter='\n')
np.savetxt("ion_time_real_2.txt", np.array(itime_min_real_2), delimiter='\n')


# Importing data from saddle point model
t2 = []
file_path = 'final_real_2.txt'
try:
    with open(file_path, 'r') as file:
        for line in file:
            try:
                number = float(line)
                t2.append(number)
            except ValueError:
                print(f"Skipping non-numeric line: {line}")
except FileNotFoundError:
    print(f"File '{file_path}' not found.")

tr_re_2 = [x for x in t2]

t1 = []
file_path = 'final_real_1.txt'
try:
    with open(file_path, 'r') as file:
        for line in file:
            try:
                number = float(line)
                t1.append(number)
            except ValueError:
                print(f"Skipping non-numeric line: {line}")
except FileNotFoundError:
    print(f"File '{file_path}' not found.")

tr_re_1 = [x for x in t1]

i2 = []
file_path = "ion_time_real_2.txt"
try:
    with open(file_path, 'r') as file:
        for line in file:
            try:
                number = float(line)
                i2.append(number)
            except ValueError:
                print(f"Skipping non-numeric line: {line}")
except FileNotFoundError:
    print(f"File '{file_path}' not found.")

ti_re_2 = [x for x in i2]

i1 = []
file_path = "ion_time_real_1.txt"
try:
    with open(file_path, 'r') as file:
        for line in file:
            try:
                number = float(line)
                i1.append(number)
            except ValueError:
                print(f"Skipping non-numeric line: {line}")
except FileNotFoundError:
    print(f"File '{file_path}' not found.")

ti_re_1 = [x for x in i1]

N = np.arange(20, 80, 2)

# Classical model
Ip = 21.5  # in eV
I0 = 4 * 1e14  # in W/sm^2
lmbda0 = 800  # in nm
c = 299.792458  # in nm/fs

w0 = 2 * np.pi * c / lmbda0
T = 2 * np.pi / w0

# Ponderomotive energy (I0 [W/sm^2] and lambda[mkm] -> eV)
Up = (9.33 * 1e-14) * I0 * (lmbda0 * 1e-3) ** 2  # in eV
hbar = 0.6582
# Omega in terms of eV
omega_eV = w0 * hbar

# Kinetic energy in terms of ionization time
def K(ti):
    return (Ip + 2 * Up * (np.cos(3 * np.arcsin((2 / np.pi) * w0 * ti - 1)) - np.sin(w0 * ti)) ** 2) / omega_eV

# Kinetic energy in terms of recombination time
def Kr(tr):
    return (Ip + 2 * Up * (np.sin(w0 * tr) - np.cos((np.pi / 2) * np.sin((1 / 3) * w0 * tr - np.pi / 6))) ** 2) / omega_eV

ti = np.linspace(0, T / 4, 1000)
tr = np.linspace(T / 4, T, 1000)

# Plotting all together
plt.figure()
plt.plot(ti, K(ti), color="black", label="K(ti)")
plt.plot(tr, Kr(tr), 'r-', label="Kr(tr)")
plt.plot(tr_re_1, N, "b--", label="Real Return Time 1")
plt.plot(tr_re_2, N, "b--", label="Real Return Time 2")
plt.plot(ti_re_1, N, "r--", label="Ionization Time 1")
plt.plot(ti_re_2, N, "r--", label="Ionization Time 2")
plt.ylabel("Harmonic order, N")
plt.xlabel("time, fs")
plt.legend()
plt.show()


