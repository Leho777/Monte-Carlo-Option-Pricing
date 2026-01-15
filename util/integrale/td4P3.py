# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:07:21 2024

@author: CYTech Student
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# Simuler une loi normale N(0, 1)
valeur_normale = random.gauss(0, 1)
print(valeur_normale)


T=5
tau = T/2
N=100
Nmc = 100

def simuBrownien():
    
    lastvalue_WW2 = []
    deltaT = T/N
    deltatau = tau/N
  
    for j in range(Nmc):
        W, t =[0], [0]
        W2, t2 = [0], [0]
        for i in range(N):
            W.append(W[i] + ((deltaT**0.5) * random.gauss(0, 1)))
            t.append(t[i] + deltaT)
            
        lastvalue_WW2.append((W[-1]**2)*(W[int(N/2)]**2))
            
        
    esperance = np.mean(lastvalue_WW2)
    variance = np.var(lastvalue_WW2)
    
    print("Esperance ", esperance)
    print("Variance ", variance)
    
    return lastvalue_WW2, t

simuB, t = simuBrownien()
#plt.plot(simuB)
"""
for e in simuB:
    plt.plot(t,e)
    
plt.title("Mvt Brownien")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique"""