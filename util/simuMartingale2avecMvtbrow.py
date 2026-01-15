# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:10:31 2024

@author: CYTech Student
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# Simuler une loi normale N(0, 1)
valeur_normale = random.gauss(0, 1)
print(valeur_normale)


#T=1
#N=100
#Nmc = 1
    
def simuMt2_t(N = 100, Nmc = 1000, T = 1):
    
    all_M = []
    deltaT = T/N
    last_value = []
    for j in range(Nmc):
        M, t =[0], [0]
        for i in range(N):
            t.append(t[i] + deltaT)
            M.append(( (M[i] + t[i])**0.5 + (deltaT**0.5) * random.gauss(0, 1))**2 - t[i+1])
            
        all_M.append(M)
        last_value.append(M[-1])
    esperance = np.mean(last_value)
    variance = np.var(last_value)
    
    print("Esperance ", esperance)
    print("Variance ", variance)
    
    return all_M, t

simuB, t = simuMt2_t()

for e in simuB:
    plt.plot(t,e)
    
plt.title("Mvt Brownien")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique

def martingale(k = 50, N = 100, Nmc = 10, T = 1):
    
    all_M = []
    deltaT = T/N
    last_value = []
    
    Mk, t = [0], [i * deltaT for i in range(N + 1)]
    
    for i in range(k):
        #t.append(t[i] + deltaT)
        Mk.append(( (Mk[i] + t[i])**0.5 + (deltaT**0.5) * random.gauss(0, 1))**2 - t[i+1])
    
    for j in range(Nmc):
        M = Mk.copy()
        
        for i in range(k, N):
            
            """ if j == 0:
                t.append(t[i] + deltaT) # pour opti la complexité"""
            M.append(( (M[i] + t[i])**0.5 + (deltaT**0.5) * random.gauss(0, 1))**2 - t[i+1])
            
        all_M.append(M)
        last_value.append(M[-1])
    print("Last value Mk = ", Mk[-1])
    esperance = np.mean(last_value)
    print("Esperance last value = ", esperance)
    
    return all_M, t

mart, t = martingale(750, 1000, 100, 1)
plt.figure(figsize=(8, 4))
for e in mart:
    plt.plot(t,e)
    
plt.title("Martingale de Mt = Wt² - t")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique