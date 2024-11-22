# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 07:20:26 2024

@author: abedn
"""

#%% Getting the input pulse

import numpy as np
import matplotlib.pyplot as plt

T_p = 50 # fs  describes the pulse duration at the sample input (Gausssian)
#T_p is pulse duration FWHM

# T_p = np.sqrt(2 * np.log(2)) * T_G0# In fs unit
#For a typical guassian pulse
T_G0 = T_p / (np.sqrt(2 * np.log(2))) #The pulse duration

wavelength = 800 # nm
c = 300 # nm/fs
w_l =  2 * np.pi * c / wavelength # fs


z = 0 # mm
a = 0 # chirp parameter is zero meaning fourier limited
x = T_G0**2 / 4*(1 + a**2)
y_z = (a * T_G0**2 / 4*(1 + a**2) ) - 36.1*z /2 # 36.1 is the second derivative of propagation constant
t = np.linspace(-100, 100, 200)

#Lets get the electric  field
E = np.exp(-1 * (1 + 1j*y_z / x) * (t / np.sqrt(4 / x * (x**2 + y_z**2)))**2 )

#Get the intensity
I = abs(E**2)


#%% The plot
plt.figure()
plt.plot(t, I, 'r-', label = ' z = 0mm' )
plt.ylabel('Intensity [au]', fontsize = 14)
plt.xlabel('Time [fs]', fontsize = 14)
plt.legend(loc = 'upper right', fontsize = 14)
plt.grid()
plt.show()

#%% The temporal after passing throgh material at specific length
l_d = (T_G0**2) / (2 * 36.1) # mm (the second derivative of propagation constant in unit of fs**2/mm)

z_1 = np.arange(20, 50,10) # mm the length
print(z_1)

def y_z1(y):
    s = (a * T_G0**2 / 4*(1 + a**2) ) - 36.1*y /2 # 36.1 is the second derivative of propagation constant
    return s


def E (f):
    return  np.exp(-1 * (1 + (1j*y_z1(f)/ x)) * (t / np.sqrt(4 / x * (x**2 + y_z1(f)**2)))**2)
for num in z_1:
    intensity = abs(E(num))**2
    plt.plot(t,intensity, label = f" Z = {num} mm")
    plt.ylabel("Intensity [a.u]", fontsize = 14)
    plt.xlabel('Time [fs]', fontsize = 14)
    plt.legend(loc = 'upper right', fontsize = 11)
    plt.grid()
    #%% To get the distance the pulse duration doubles
l_d = (T_G0**2) / (2 * 36.1) # mm (the second derivative of propagation constant in unit of fs**2/mm)
z_2 = 1.73 * l_d
print(f"{l_d} mm")
print(f"{z_2} mm")

#%% Now we get the plot for the double duration
#The double duration  and the original pulse
z_3 = np.array([0, 43.211]) #mm
def y_z2(p):
    q = (a * T_G0**2 / 4*(1 + a**2) ) - 36.1*p /2 # 36.1 is the second derivative of propagation constant
    return q

def E (g):
    return  np.exp(-1 * (1 + (1j*y_z2(g)/ x)) * (t / np.sqrt(4 / x * (x**2 + y_z2(g)**2)))**2)
for num in z_3:
    intensity = abs(E(num))**2
    plt.plot(t,intensity, label = f" Z = {num} mm")
    plt.ylabel("Intensity [a.u]", fontsize = 14)
    plt.xlabel('Time [fs]', fontsize = 14)
    plt.legend(loc = 'upper right', fontsize = 11)
    plt.grid(True)
    
#%% Problem 2
#we find the carrier frequency of the laser w_0
w_0 = 2 * np.pi * (c / wavelength) #/fs
#The intensity
I_0 = 1e12 # W/cm^2
#the non-linear refractive index
n_2 = 0.85e-19 #cm^2/W
#the length
L = 1e9 #nm
#The intensity I(t)
I_1 = I_0 *np.exp(-2 *(t / T_G0)**2)
#Instantaneous frequency
w_t = w_0 + (8 * np.pi * L * n_2 * t * I_1) / (wavelength * T_G0**2)
#%% A function to get the FWHM
def FWHM(x, y):
    max_y = max(y)  # Find the maximum y value
    half_max_y = max_y / 2  # Calculate half of the maximum y value

    # Find the indices where y crosses half of the max value
    condition = y >= half_max_y  # Boolean array for values greater than or equal to half max
    # Extract x values corresponding to y values greater than half max
    xpoints = x[condition]
    # Return the Full Width at Half Maximum (FWHM)
    return max(xpoints) - min(xpoints)

#%%The intensity with I_0 against normalised time
plt.figure()
plt.plot(t, I_1/max(I_1), label = ' Input pulse', color = 'red')
plt.ylabel('Normalized intensity [au]', fontsize = 14)
plt.xlabel('Time [fs]', fontsize = 14)
plt.legend(loc = 'upper right', fontsize = 14)
plt.grid()
plt.show()
#Getting the FWHM
fwhm_value = FWHM(t, I_1 / max(I_1))  
print(f"{fwhm_value} is the FWHM")
#%%
#The plot of instantaneous frequency
#instantaneous frequency induced by SPM
del_w = 8 * np.pi * L * n_2 * t * I_1 / (wavelength * T_G0**2)
plt.figure()
plt.plot(t, w_t, label = ' instantaneous frequency', color = 'blue')
plt.ylabel('I [/fs]')
plt.xlabel('Time [fs]')
plt.legend(loc = 'upper left')
plt.grid()
plt.show()
#%%The instantaneous frequency tangent
w_tangent = w_0 + (8 * np.pi * L * n_2 * t * I_0) / (wavelength * T_G0**2)
#plotting with the instantaneous frequency and the tangent
plt.figure()
plt.plot(t, w_t, 'b-', label = ' instantaneous frequency' )
plt.plot(t, w_tangent,  'r--')
plt.ylabel('Instantaneous frequency [/fs]')
plt.xlabel('Time [fs]')
plt.legend(loc = 'upper left')
plt.grid()
plt.show()
#%% Getting the electric field
#the permitivity 
e_eshislon = 8.854e-3 #F/nm
#the amplitude of the field 
E_0 = np.sqrt( (2 * (I_0 / 1e14))/(c * e_eshislon)) #convert the intensity to W/nm^2
print(E_0)
#the phase modulation before the input  at zero length
phi_t0 = w_0 * t
#the field at zero length as function of time
E_t0 = E_0  * np.exp(-1 * (t/T_G0)**2) * np.exp(1j * phi_t0)
plt.figure()
plt.plot(t, E_t0, label = ' Input field', color = 'red')
plt.ylabel('Electric field [au]', fontsize = 14)
plt.xlabel('Time [fs]', fontsize = 14)
plt.legend(loc = 'upper right', fontsize = 14)
plt.grid()
plt.show()

#%% Getting the plot superimposed of the Field  and the instantaneous frequency

# Create figure and axis objects
fig, ax1 = plt.subplots()

# Plot the intensity on the first axis
ax1.plot(t, np.absolute(E_t0)/max(E_t0), color = 'red', label='Input field') #plot the absolute value of the real
ax1.set_xlabel('Time [fs]', fontsize = 14)
ax1.set_ylabel('Normalised Electric field [a.u.]', color='r', fontsize = 11)
ax1.tick_params(axis='y', labelcolor='r')

# Create a secondary axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(t, w_t, color = 'b', label='Instantaneous Frequency')
ax2.set_ylabel('Instantaneous frequency [/fs]', color='black')
ax2.tick_params(axis='y', labelcolor='b')

ax1.grid(True)
fig.tight_layout() 
plt.show()
#%% The field through 1m of the medium
#The phase modulation at this length
n_0 = 1 # at pressure of 1bar
n_I = n_0 + n_2 * I_1
I_t0 = np.abs(E_t0)**2
phi_t1 = w_0 * t - n_2 * I_1 * ((2 * np.pi * L) / wavelength) #the phase modulation (non-linear)
#the electric field at this length and visualization
E_t1 = E_0 * np.exp(-1 * (t/T_G0)**2) * np.exp(1j * phi_t1)
plt.figure()
plt.plot(t, abs(E_t1)/max(E_t1), label = ' Output field', color = 'red') #normalize
plt.plot(t, np.absolute(E_t0)/max(E_t0), color = 'blue', label='Input field')
plt.ylabel(' Normalised electric field [au]', fontsize = 14)
plt.xlabel('Time [fs]', fontsize = 14)
plt.legend(loc = 'upper right', fontsize = 11)
plt.grid()
plt.show()

#Getting the FWHM
# Calculate FWHM for input and output fields
fwhm_input = FWHM(t, E_t0/max(E_t0))  # FWHM of the input intensity
fwhm_output = FWHM(t, E_t1/max(E_t1))  # FWHM of the output intensity

print(f"Input FWHM: {fwhm_input:.2f} fs")
print(f"Output FWHM: {fwhm_output:.2f}  fs")

#%% Getting some parameter of SPM

w = np.linspace(2.0, 2.6, len(t))
phi_max =  2 * np.pi * I_0 * L * n_2 / wavelength #maximum phase shift
SPM_coeff = 2 * np.pi * n_2 * 1e14 * L / wavelength


#%% Fourier transform to spectra domain
from scipy.integrate import simps
w = np.linspace(2.0, 2.6, len(t))
def fourier_transform(field, t, w):
    Ew = []
    for omega in w:
        exp_term = np.exp(-1j * omega * t)
        integral = field * exp_term
        Ew.append(simps(integral, t))
    return np.array(Ew)


# Compute Fourier Transform of the input electric field
E_in_omega = fourier_transform(E_t0, t, w)
# Compute Fourier Transform of the output electric field
E_out_omega = fourier_transform(E_t1, t, w)

E_in_norm = np.abs(E_in_omega)/max(E_in_omega)
E_out_norm = np.abs(E_out_omega)/max(E_out_omega)

plt.figure(figsize=(12, 6))
plt.plot(w, E_in_norm , label='Input Pulse Spectrum ',  color = "red")
plt.plot(w, E_out_norm, label='Output Pulse Spectrum ' , color = "blue" )
plt.xlabel('Angular Frequency (rad/fs)', fontsize= 13)
plt.ylabel('Normalized Spectral Amplitude', fontsize= 13)
plt.legend(fontsize= 11)
plt.grid(True)
plt.show()

#Getting the FWHM
# Calculate FWHM for input and output fields
fwhm_input = FWHM(w, E_in_norm)  # FWHM of the input intensity
fwhm_output = FWHM(w, E_out_norm)  # FWHM of the output intensity

print(f"Input FWHM: {fwhm_input:.2f} PHz")
print(f"Output FWHM: {fwhm_output:.2f}  PHz")
#%%

