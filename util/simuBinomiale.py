# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:32:44 2024

@author: CYTech Student
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Paramètres
sigma = 0.5
r = 0.1
T = 0.5
S0 = 10
N = 20
delta_t = T / N

xabscisse = [i*delta_t for i in range (N+1)]

print(xabscisse)
# Calcul des facteurs et de la probabilité
u = math.exp(sigma * math.sqrt(delta_t))
d = math.exp(-sigma * math.sqrt(delta_t))
p = (math.exp(r * delta_t) - d) / (u - d)

print(u, d, p)

#Fct pour construire tout l'arbre

def construireTree(S0, N, u, d):
    prices = []
    
    for i in range(N+1):
        price = []
        for j in range(i+1):
            price.append(S0*(u**j)*(d**(i-j)))
        prices.append(price)
    return prices

prices = construireTree(S0, N, u, d)

# Préparer les données pour le nuage de points
x = []
y = []
for i in range(len(prices)):
    for price in prices[i]:
        x.append(i*delta_t)      # Niveau de l'arbre
        y.append(price)  # Prix

# Fonction pour simuler une trajectoire
def simulate_binomial_tree(S0, N, u, d):
    prices = []
    price = S0
    
    for n in range(N + 1):
        prices.append(price)
        # Décider si on va "monter" ou "descendre"
        if n < N:
            if random.random() < p:
                price *= u  # Move up
            else:
                price *= d  # Move down
                
    return prices

# Simuler plusieurs trajectoires
def simulate_multiple_trajectories(S0, N, u, d, num_trajectories):
    all_prices = []
    last_value = []
    
    
    for i in range(num_trajectories):
        trajectory = simulate_binomial_tree(S0, N, u, d)
        all_prices.append(trajectory)
        last_value.append(trajectory[-1])
        
    esperance = np.mean(last_value)
    
    print("Esperance emprique: ", esperance)
    print("Espérance théorique: ", S0 * (p * u + (1 - p) * d)**N)


    print("S0", S0)
    
    return all_prices

# Nombre de trajectoires à simuler
num_trajectories = 100
trajectories = simulate_multiple_trajectories(S0, N, u, d, num_trajectories)

# Visualiser les trajectoires
plt.figure(figsize=(10, 10))
for i in range(num_trajectories):
    plt.plot(xabscisse,trajectories[i], marker='o') #, label=f'Trajectoire {i + 1}')
plt.scatter(x, y, color='blue', label='Prix possibles')
plt.title('Trajectoires des Actifs dans un Modèle Binomial')
plt.xlabel('Étapes (N)')
plt.ylabel('Prix de l’Actif')
plt.legend()
plt.grid()
plt.show()



