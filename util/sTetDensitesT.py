# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:57:05 2024

@author: CYTech Student
"""

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
Nmc = 1000  # Nombre de trajectoires
T = 2  # Temps final
S0 = 10  # Valeur initiale
sigma0 = 0.5  # Volatilité initiale
r0 = 0.8  # Taux initial
N_steps = 1000  # Nombre de pas de temps
dt = T / N_steps  # Taille d'un pas de temps

# Fonction pour r(t) et sigma(t, St)
def r(t):
    return r0 * (np.cos(t / T)) ** 2

def sigma(t, St):
    return sigma0 * (np.sin(St / S0)) ** 2

# Simulation des trajectoires
t_values = np.linspace(0, T, N_steps + 1)  # Discrétisation temporelle
S = np.zeros((Nmc, N_steps + 1))  # Stocker les trajectoires
S[:, 0] = S0  # Initialisation

for i in range(N_steps):
    t = t_values[i]
    dW = np.sqrt(dt) * np.random.randn(Nmc)  # Incréments Brownien
    S[:, i + 1] = S[:, i] + S[:, i] * (r(t) * dt + sigma(t, S[:, i]) * dW)

# Valeurs finales ST
ST = S[:, -1]

# Calcul de E[ST]
E_ST = np.mean(ST)

# Affichage du résultat
print(f"E[ST] (espérance estimée) = {E_ST}")

# Tracé de quelques trajectoires
plt.figure(figsize=(10, 6))
for i in range(10):  # Afficher 10 trajectoires
    plt.plot(t_values, S[i], alpha=0.7)
plt.title("Simulations de trajectoires de S_t")
plt.xlabel("Temps t")
plt.ylabel("S_t")
plt.grid()
plt.show()

# Densité empirique de ST
plt.figure(figsize=(10, 6))
plt.hist(ST, bins=30, density=True, color='blue', alpha=0.7, edgecolor='black')
plt.title("Densité empirique de ST")
plt.xlabel("ST")
plt.ylabel("Densité")
plt.grid()
plt.show()
