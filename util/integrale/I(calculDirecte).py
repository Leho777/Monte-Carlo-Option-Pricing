# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:51:36 2024

@author: CYTech Student
"""

import random
import matplotlib.pyplot as plt
import numpy as np


def theta(t, Wt):
    return t + Wt**2
    
def simuI(Nmc = 100, N = 100, T = 1):
    all_I = []
    deltaT = T/N
    for j in range(Nmc):
        I, t = 0, [0]
        W =[0]
        for i in range(N):
            
            W.append(W[i] + (deltaT**0.5) * random.gauss(0, 1))
            
            I += theta(t[i], W[i]) * (W[i+1] - W[i])
            t.append(t[i] + deltaT)
            
        all_I.append(I)

    esperance = np.mean(all_I)
    variance = np.var(all_I)
    
    print("Esperance ", esperance)
    print("Variance ", variance)
    
    return all_I, t
    
"""all_I, t = simuI()
print(all_I)
for i in range(len(all_I)):
    plt.plot(i, all_I[i])
plt.title("I")
plt.xlabel('Itération')
plt.ylabel('Valeur de I')
plt.show()  # Afficher le graphique"""

all_I, t = simuI()

# Tracé du graphique de l'évolution des valeurs de I
plt.plot(range(len(all_I)), all_I, marker='o', linestyle='-')
plt.title("Évolution de I sur différentes simulations")
plt.xlabel('Itération')
plt.ylabel('Valeur de I')
plt.show()