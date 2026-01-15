# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:02:12 2024

@author: CYTech Student
"""


import random
import matplotlib.pyplot as plt
import numpy as np

def relationSto(Nmc = 1, N = 10000, T =1):
    deltaT = T/N
    W = [0]
    t = [0]
    I = [0]
    fW = [0]
    for i in range(N):
        
        t.append(t[i] + deltaT)
        W.append(W[i] + (deltaT**0.5) * random.gauss(0, 1))
        I.append(I[i] + (np.exp(t[i]*0.5) * np.cos(W[i]) * (W[i+1] - W[i])))
        fW.append( np.exp(0.5*t[i+1]) * np.sin(W[i+1]) )
        
        print(I[i], fW[i])
        
    return I, fW, t

I, fW, t = relationSto()
plt.plot(t,I)
plt.plot(t,fW)

print(I[-1],fW[-1])
plt.title("Relation d'Ito")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique