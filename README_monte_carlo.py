"""
RÉSUMÉ : Pricing Monte Carlo pour Options Européennes
======================================================

Ce fichier explique l'algorithme implémenté dans monte_carlo_model.py

ALGORITHME MONTE CARLO POUR OPTIONS EUROPÉENNES
------------------------------------------------

Objectif : Calculer le prix d'une option européenne

Principe : Le prix d'une option est la valeur actuelle (actualisée) 
          de son payoff moyen à maturité.

          Prix = E[e^(-rT) × Payoff(S_T)]

Où :
- S_T = prix du sous-jacent à maturité T
- r = taux sans risque
- Payoff = max(S_T - K, 0) pour un Call
         = max(K - S_T, 0) pour un Put


ÉTAPES DE L'ALGORITHME (répéter N fois)
----------------------------------------

1. TIRER UN NOMBRE ALÉATOIRE
   u = random.uniform(0, 1)
   
   → Génère un nombre aléatoire uniformément distribué entre 0 et 1

2. CONVERTIR EN TIRAGE NORMAL
   Z = random.standard_normal()
   
   → Tire une variable aléatoire Z suivant une loi normale N(0,1)
   → Moyenne = 0, Écart-type = 1

3. CONVERTIR EN TIRAGE BROWNIEN
   W(T) = Z × √T
   
   → Le mouvement brownien à l'instant T suit : W(T) ~ N(0, T)
   → Moyenne = 0, Variance = T

4. CALCULER LE PRIX DU SOUS-JACENT À MATURITÉ
   S(T) = S₀ × exp((r - q - σ²/2)T + σW(T))
   
   Où :
   - S₀ = prix initial du sous-jacent
   - r = taux sans risque
   - q = rendement du dividende (dividend yield)
   - σ = volatilité
   - T = temps jusqu'à maturité (en années)
   
   Décomposition :
   - Drift (tendance)    : (r - q - σ²/2)T
   - Diffusion (aléa)    : σW(T)
   - Exponentielle       : transformation log-normale

5. CALCULER LE PAYOFF DE L'OPTION À MATURITÉ
   
   Pour un CALL : Payoff = max(S(T) - K, 0)
   Pour un PUT  : Payoff = max(K - S(T), 0)
   
   Où K = strike de l'option

6. ACTUALISER À AUJOURD'HUI
   Valeur_actualisée = Payoff × e^(-rT)
   
   → On ramène la valeur future en valeur présente
   → Facteur d'actualisation : e^(-rT)

7. FAIRE LA MOYENNE SUR TOUTES LES SIMULATIONS
   Prix_option = (1/N) × Σ(Valeur_actualisée_i)
   
   → C'est l'espérance empirique du payoff actualisé


CONVERGENCE ET PRÉCISION
-------------------------

Erreur standard : σ_MC / √N

Où :
- σ_MC = écart-type des payoffs actualisés
- N = nombre de simulations

→ Pour diviser l'erreur par 2, il faut multiplier N par 4
→ Convergence en O(1/√N) (loi des grands nombres)

Intervalle de confiance à 95% :
Prix ± 1.96 × (σ_MC / √N)


EXEMPLE NUMÉRIQUE
-----------------

Paramètres :
- S₀ = 100 (prix initial)
- K = 100 (strike)
- r = 5% (taux)
- σ = 20% (volatilité)
- T = 0.5 ans (6 mois)
- Type : Call européen

Une simulation :
1. Z = -1.11 (tirage normal)
2. W(T) = -1.11 × √0.5 = -0.79
3. S(T) = 100 × exp((0.05 - 0.02)×0.5 + 0.2×(-0.79)) = 86.74
4. Payoff = max(86.74 - 100, 0) = 0
5. Actualisé = 0 × exp(-0.05×0.5) = 0

Après 100,000 simulations :
Prix ≈ 6.86 € (avec erreur standard ≈ 0.03)


AVANTAGES DU MONTE CARLO
-------------------------

✓ Simplicité : algorithme facile à comprendre et implémenter
✓ Flexibilité : peut pricer des options complexes (barrières, asiatiques, etc.)
✓ Multidimensionnel : facilement extensible à plusieurs sous-jacents
✓ Parallélisable : les simulations sont indépendantes

INCONVÉNIENTS
-------------

✗ Lenteur : besoin de beaucoup de simulations pour la précision
✗ Convergence lente : O(1/√N)
✗ Moins adapté pour les options américaines (exercice anticipé)


COMPARAISON AVEC BLACK-SCHOLES
-------------------------------

Pour une option européenne standard, la formule de Black-Scholes
donne le prix exact instantanément.

Monte Carlo donne une approximation qui converge vers ce prix.

Mais Monte Carlo est plus général et peut pricer des options
que Black-Scholes ne peut pas traiter (path-dependent, etc.)


VALIDATION : PUT-CALL PARITY
-----------------------------

Pour vérifier l'implémentation, on peut tester la parité put-call :

C - P = S₀ - K × e^(-rT)

Où :
- C = prix du call
- P = prix du put
- S₀ = prix spot
- K = strike
- r = taux
- T = maturité

Cette relation doit être vérifiée (à l'erreur Monte Carlo près).


CODE PYTHON (simplifié)
-----------------------

```python
import numpy as np

def price_european_call_mc(S0, K, r, sigma, T, N):
    '''
    S0: prix initial
    K: strike
    r: taux sans risque
    sigma: volatilité
    T: maturité (années)
    N: nombre de simulations
    '''
    # Simulations
    Z = np.random.standard_normal(N)
    W_T = Z * np.sqrt(T)
    
    # Prix du sous-jacent à T
    S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*W_T)
    
    # Payoffs
    payoffs = np.maximum(S_T - K, 0)
    
    # Actualisation
    discounted = payoffs * np.exp(-r*T)
    
    # Prix = moyenne
    price = np.mean(discounted)
    
    return price
```


EXTENSIONS POSSIBLES
--------------------

1. Variables de contrôle (variance reduction)
2. Antithetic variates
3. Importance sampling
4. Quasi-Monte Carlo (Low discrepancy sequences)
5. Options path-dependent (asiatiques, lookback, etc.)
6. Options multi-sous-jacents (rainbow, basket, etc.)
7. Calcul des Greeks par différences finies
8. Parallélisation GPU


RÉFÉRENCES
----------

- Hull, J. (2018). Options, Futures, and Other Derivatives
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering
- Shreve, S. (2004). Stochastic Calculus for Finance II

"""

# Exemple d'utilisation simple
if __name__ == "__main__":
    import numpy as np
    
    print("=" * 70)
    print("EXEMPLE SIMPLE DE PRICING MONTE CARLO")
    print("=" * 70)
    
    # Paramètres
    S0 = 100      # Prix initial
    K = 100       # Strike
    r = 0.05      # Taux 5%
    sigma = 0.20  # Volatilité 20%
    T = 0.5       # 6 mois
    N = 100000    # 100k simulations
    
    print(f"\nParamètres:")
    print(f"  S0 = {S0}")
    print(f"  K = {K}")
    print(f"  r = {r*100}%")
    print(f"  σ = {sigma*100}%")
    print(f"  T = {T} ans")
    print(f"  N = {N:,} simulations")
    
    # Monte Carlo
    np.random.seed(42)
    Z = np.random.standard_normal(N)
    W_T = Z * np.sqrt(T)
    S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*W_T)
    
    # Call
    call_payoffs = np.maximum(S_T - K, 0)
    call_discounted = call_payoffs * np.exp(-r*T)
    call_price = np.mean(call_discounted)
    call_std_error = np.std(call_discounted) / np.sqrt(N)
    
    # Put
    put_payoffs = np.maximum(K - S_T, 0)
    put_discounted = put_payoffs * np.exp(-r*T)
    put_price = np.mean(put_discounted)
    put_std_error = np.std(put_discounted) / np.sqrt(N)
    
    print(f"\nRésultats:")
    print(f"  Prix Call: {call_price:.6f} ± {call_std_error:.6f}")
    print(f"  Prix Put:  {put_price:.6f} ± {put_std_error:.6f}")
    
    # Put-Call Parity
    pcp_lhs = call_price - put_price
    pcp_rhs = S0 - K * np.exp(-r*T)
    print(f"\nPut-Call Parity:")
    print(f"  C - P = {pcp_lhs:.6f}")
    print(f"  S0 - K×e^(-rT) = {pcp_rhs:.6f}")
    print(f"  Différence = {abs(pcp_lhs - pcp_rhs):.6f}")
    
    print("\n" + "=" * 70)
