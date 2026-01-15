# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:27:23 2024

@author: CYTech Student
"""

import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Fonction gaussienne basée sur la méthode de Box-Muller
def gaussienne(mu=0, sigma=1):
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + z0 * sigma

# Générer de nombreux points
n_points = 10000  # Nombre de points à générer
points = [gaussienne(0, 1) for _ in range(n_points)]

# Tracer un histogramme
plt.hist(points, bins=50, density=True, alpha=0.6, color='blue', label="Données simulées")

# Superposer la courbe théorique pour comparaison
x = np.linspace(-4, 4, 500)  # Points pour la courbe
y = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * x**2)  # PDF de N(0,1)
plt.plot(x, y, color='red', linewidth=2, label="Courbe théorique")

# Personnalisation du graphique
plt.title("Histogramme des points générés avec une distribution normale")
plt.xlabel("Valeurs")
plt.ylabel("Densité")
plt.legend()
plt.grid()

# Affichage
plt.show()


    