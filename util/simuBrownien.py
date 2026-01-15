# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:10:58 2024

@author: CYTech Student
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# Simuler une loi normale N(0, 1)
valeur_normale = random.gauss(0, 1)
print(valeur_normale)


T=100
N=1000
Nmc = 1000

def simuBrownien():
    
    all_W = []
    deltaT = T/N
    last_value = []
    for j in range(Nmc):
        W, t =[0], [0]
        for i in range(N):
            W.append(W[i] + ((deltaT**0.5) * random.gauss(0, 1)))
            t.append(t[i] + deltaT)
        all_W.append(W)
        last_value.append(W[-1])
    esperance = np.mean(last_value)
    variance = np.var(last_value)
    
    print("Esperance ", esperance)
    print("Variance ", variance)
    
    return all_W, t

simuB, t = simuBrownien()
#plt.plot(simuB)
for e in simuB:
    plt.plot(t,e)
    
plt.title("Mvt Brownien")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique