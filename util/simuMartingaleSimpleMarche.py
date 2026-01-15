# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:59:05 2024

@author: CYTech Student
"""

import matplotlib.pyplot as plt
import random as rd

# Génère un pas aléatoire de +1 ou -1
def pas():
    i = rd.random()
    if i < 0.5:
        return 1
    else:
        return -1

# Simule un processus de marche aléatoire
def processus(k):
    M = [0]
    for i in range(k):
        M.append(M[i] + pas())
    return M, M[-1]

# Fonction martingale qui simule Nmc marches aléatoires
def martingale(n, k, Nmc):
    last_value = []  # Liste pour les valeurs finales
    Mk, lastValueMk = processus(k)
    #marches = []
    for j in range(Nmc):
        
        M = [ Mk[i] for i in range(len(Mk))]
        
        # Simule la marche aléatoire de k à n
        for i in range(k, n):
            M.append(M[i] + pas())
        all_marches.append(M)   
        last_value.append(M[-1])  # Ajoute la dernière valeur de chaque marche
    
    moyenne_last_value = sum(last_value) / len(last_value)
    return moyenne_last_value, lastValueMk

all_marches = []


# Appel de la fonction martingale avec 100 étapes, k=10, et 1000 simulations
moyenne_finale, MF_k = martingale(50, 30, 10)
moyenne_finale, MF_k = martingale(50, 30, 10)
print("Moyenne des dernières valeurs :", moyenne_finale)
print("M_k", MF_k)


 # Afficher toutes les marches
for marche in all_marches:
    plt.plot(marche)  # Tracer chaque marche sur le même graphique
 
plt.title("Marches aléatoires")
plt.xlabel('Étapes')
plt.ylabel('Position')
plt.show()  # Afficher le graphique