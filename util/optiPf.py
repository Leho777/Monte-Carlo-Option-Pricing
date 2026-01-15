# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:23:37 2024

@author: CYTech Student
"""

import random
import matplotlib.pyplot as plt
import numpy as np

"""Mvt Stochastique par Méthode d'Euler"""

def SimuMt(N = 100, Nmc = 1000, T = 2, r =0.07, sigma = 0.5, gamma = 0.5, mu = 0.15, M0 = 500, theta = 0.5):

    all_M = []
    deltaT = T/N
    last_value = []
    
    compteur = 0
    for j in range(Nmc):
        M, t = [M0], [0]

        for i in range(N):
            t.append(t[i] + deltaT)
            
            M.append( M[i] + (r + theta*(mu - r)*M[i] - np.exp(-6*t[i]/T))*deltaT       )            ) 
            
        all_M.append(M)
        
        #if M[-1] < 800:
        #    compteur += 1
            
        last_value.append(M[-1])
        
    #last_valueUM = [ np.log(e) for e in last_value]
    #esperanceUMTheta = np.mean(last_valueUM)
    
    #print("P[Mt < 800] = ", compteur/Nmc)
        
        
    """EsperanceS = [S[0]] 
    for i in range(N):
        EsperanceS.append(S[0]*np.exp(r*t[i]))
    all_S.append(EsperanceS)
    esperance = np.mean(last_value)
    variance = np.var(last_value)
    
    print("Esperance Empirique = ", esperance)
    print("Variance Empirique = ", variance)
    print("Esperance Théorique = ", EsperanceS[-1])"""
    
    return all_M, t #, esperanceUMTheta
"""
simuXt, simuPt, simuMt, t, esp = SimuXtPtMt()

for e in simuXt:
    plt.plot(t,e)

plt.title("Mvt Xt")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique

for e in simuPt:
    plt.plot(t,e)
    
plt.title("Mvt Pt")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique

for e in simuMt:
    plt.plot(t,e)
#plt.plot(t, simuS[-1], color='red', linewidth=5, label="Dernière courbe")  # Ligne plus épaisse et rouge
    
plt.title("Mvt Mt")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique 
"""

"""
def optimisation(N = 100):
    deltaTheta = 1/N
    theta = [0]
    all_espUMTheta = []
    for i in range (N):
        
        simuXt, simuPt, simuMt, t, esperanceUMTheta = SimuXtPtMt(100, 10000, 1, 0.07, 0.3, 0.4, 2, 0, 0.12, theta[i])
        theta.append( theta[i] + deltaTheta)
        print(i)
        all_espUMTheta.append(esperanceUMTheta)
        
        
    return all_espUMTheta, theta[:-1]

all_espUMTheta, theta = optimisation(10)

plt.plot(theta, all_espUMTheta)
#plt.plot(t, simuS[-1], color='red', linewidth=5, label="Dernière courbe")  # Ligne plus épaisse et rouge
    
plt.title("Optimisation de Mt via Theta")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique"""
    
    