# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:26:06 2024

@author: CYTech Student
"""
import random
import matplotlib.pyplot as plt
import numpy as np

def VQ(Nmc = 1, N = 100, T =1):
    all_VQ = []
    deltaT = T/N
    last_value = []
    for j in range(Nmc):
        VQ, t =[0], [0]
        W =[0]
        for i in range(N):
            
            W.append(W[i] + (deltaT**0.5) * random.gauss(0, 1))
            
            VQ.append(VQ[i] + (W[i+1] - W[i])**2)
            t.append(t[i] + deltaT)
            
        all_VQ.append(VQ)
        last_value.append(VQ[-1])
    esperance = np.mean(last_value)
    variance = np.var(last_value)
    
    print("Esperance ", esperance)
    print("Variance ", variance)
    
    all_VQ.append([t[i] for i in range(len(t))])
    
    return all_VQ, t    

simuVQ, t = VQ()
for e in simuVQ:
    plt.plot(t,e)
plt.title("Variation Quadratique")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique