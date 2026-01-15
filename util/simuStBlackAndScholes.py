# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:14:28 2024

@author: CYTech Student
"""

import random
import matplotlib.pyplot as plt
import numpy as np

"""Mvt Stochastique par Méthode d'Euler"""

def SimuStBAH(N = 100, Nmc = 1000, T = 0.5, r =0.1, sigma = 0.5 ):

    all_S = []
    deltaT = T/N
    last_value = []
    for j in range(Nmc):
        S, t =[10], [0]
        W = [0]
        for i in range(N):
            t.append(t[i] + deltaT)
            W.append(W[i] + ((deltaT**0.5) * random.gauss(0, 1)))
            S.append( S[i] * (1+ r * deltaT + sigma * (W[i+1] - W[i])))
            
        all_S.append(S)
            
        last_value.append(S[-1])
    EsperanceS = [S[0]] 
    for i in range(N):
        EsperanceS.append(S[0]*np.exp(r*t[i]))
    all_S.append(EsperanceS)
    esperance = np.mean(last_value)
    variance = np.var(last_value)
    
    print("Esperance Empirique = ", esperance)
    print("Variance Empirique = ", variance)
    print("Esperance Théorique = ", EsperanceS[-1])
    
    return all_S, t

simuS, t = SimuStBAH()

for e in simuS[:-1]:
    plt.plot(t,e)
plt.plot(t, simuS[-1], color='red', linewidth=5, label="Dernière courbe")  # Ligne plus épaisse et rouge
    
plt.title("Mvt Sotchastique")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique


"""Mvt Sto par Solution Analytique"""
def SimuStBAH2(N = 100, Nmc = 1000, T = 0.5, r =0.1, sigma = 0.5 ):

    all_S = []
    deltaT = T/N
    last_value = []
    for j in range(Nmc):
        S, t =[10], [0]
        W = [0]
        for i in range(N):
            t.append(t[i] + deltaT)
            W.append(W[i] + ((deltaT**0.5) * random.gauss(0, 1)))
            S.append( S[i] * np.exp((r - 0.5*(sigma**2))*deltaT + sigma*(W[i+1]-W[i])))
            
        all_S.append(S)
        last_value.append(S[-1])
    esperance = np.mean(last_value)
    variance = np.var(last_value)
    
    EsperanceS = [S[0]] 
    for i in range(N):
        EsperanceS.append(S[0]*np.exp((r-0.5*sigma**2)*t[i]))
    all_S.append(EsperanceS)
    
    print("Esperance 2", esperance)
    print("Variance 2", variance)
    
    return all_S, t

simuS2, t = SimuStBAH2()
for e in simuS2[:-1]:
    plt.plot(t,e)

plt.plot(t, simuS2[-1], color='red', linewidth=5, label="Dernière courbe")  # Ligne plus épaisse et rouge
plt.title("Mvt Sotchastique V2")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique


"""Mvt Sto Vasicek """
def SimuRtVasicek(N = 100, Nmc = 100, T = 5, n =0.016, gamma = 0.2, omega = 0.02):

    all_R = []
    deltaT = T/N
    last_value = []
    for j in range(Nmc):
        R, t =[0.1], [0]
        W = [0]
        for i in range(N):
            t.append(t[i] + deltaT)
            W.append(W[i] + ((deltaT**0.5) * random.gauss(0, 1)))
            R.append(  (n - gamma*R[i]) * deltaT + omega*(W[i+1]-W[i]))
            
        all_R.append(R)
        last_value.append(R[-1])
    esperance = np.mean(last_value)
    variance = np.var(last_value)
    
    EsperanceR = [R[0]] 
    """for i in range(N):
        EsperanceR.append(R[0]*np.exp((r-0.5*sigma**2)*t[i]))
    all_R.append(EsperanceR)"""
    
    print("Esperance 2", esperance)
    print("Variance 2", variance)
    
    return all_R, t

simuR, t = SimuRtVasicek()
for e in simuR[:-1]:
    plt.plot(t,e)

plt.plot(t, simuR[-1], color='red', linewidth=5, label="Dernière courbe")  # Ligne plus épaisse et rouge
plt.title("Mvt Sotchastique V2")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique


