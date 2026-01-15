"""
Comparaison Monte Carlo vs Black-Scholes
"""
from datetime import date
from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Formule analytique de Black-Scholes pour options européennes
    
    Parameters:
    -----------
    S : float - Prix spot du sous-jacent
    K : float - Strike
    T : float - Temps jusqu'à maturité (années)
    r : float - Taux sans risque
    sigma : float - Volatilité
    option_type : str - 'call' ou 'put'
    
    Returns:
    --------
    float : Prix de l'option
    """
    if T <= 0:
        # Option expirée
        if option_type.lower() == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def compare_methods():
    """Compare Monte Carlo et Black-Scholes"""
    
    print("=" * 80)
    print("COMPARAISON : MONTE CARLO vs BLACK-SCHOLES")
    print("=" * 80)
    
    # Configuration
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    
    T = (maturity_date - pricing_date).days / 365.0
    
    print(f"\nParamètres de marché:")
    print(f"  Spot (S0):      {S0}")
    print(f"  Strike (K):     {K}")
    print(f"  Taux (r):       {r*100}%")
    print(f"  Volatilité (σ): {sigma*100}%")
    print(f"  Maturité (T):   {T:.4f} ans ({int(T*365)} jours)")
    
    # Black-Scholes (prix exact)
    bs_call = black_scholes_price(S0, K, T, r, sigma, 'call')
    bs_put = black_scholes_price(S0, K, T, r, sigma, 'put')
    
    print("\n" + "-" * 80)
    print("BLACK-SCHOLES (Prix Analytique Exact)")
    print("-" * 80)
    print(f"  Call: {bs_call:.6f}")
    print(f"  Put:  {bs_put:.6f}")
    
    # Monte Carlo avec différents nombres de simulations
    print("\n" + "-" * 80)
    print("MONTE CARLO (Approximation)")
    print("-" * 80)
    
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    # Call
    call_option = OptionTrade(
        mat=maturity_date,
        call_put='CALL',
        ex='EUROPEAN',
        k=K
    )
    
    # Put
    put_option = OptionTrade(
        mat=maturity_date,
        call_put='PUT',
        ex='EUROPEAN',
        k=K
    )
    
    print("\nCALL Option:")
    print(f"{'N Simulations':>15} | {'Prix MC':>12} | {'Erreur vs BS':>15} | {'Erreur %':>12}")
    print("-" * 80)
    
    for n_sims in [1000, 5000, 10000, 50000, 100000, 500000]:
        mc_model = MonteCarloModel(n_sims, market, call_option, pricing_date, seed=42)
        result = mc_model.price_european()
        mc_price = result['price']
        error = abs(mc_price - bs_call)
        error_pct = (error / bs_call) * 100
        
        print(f"{n_sims:>15,} | {mc_price:>12.6f} | {error:>15.6f} | {error_pct:>11.4f}%")
    
    print("\nPUT Option:")
    print(f"{'N Simulations':>15} | {'Prix MC':>12} | {'Erreur vs BS':>15} | {'Erreur %':>12}")
    print("-" * 80)
    
    for n_sims in [1000, 5000, 10000, 50000, 100000, 500000]:
        mc_model = MonteCarloModel(n_sims, market, put_option, pricing_date, seed=42)
        result = mc_model.price_european()
        mc_price = result['price']
        error = abs(mc_price - bs_put)
        error_pct = (error / bs_put) * 100
        
        print(f"{n_sims:>15,} | {mc_price:>12.6f} | {error:>15.6f} | {error_pct:>11.4f}%")


def test_different_moneyness():
    """Test pour différents niveaux de moneyness"""
    
    print("\n\n" + "=" * 80)
    print("COMPARAISON POUR DIFFÉRENTS STRIKES (Moneyness)")
    print("=" * 80)
    
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    T = (maturity_date - pricing_date).days / 365.0
    
    S0 = 100.0
    r = 0.05
    sigma = 0.20
    
    strikes = [80, 90, 100, 110, 120]
    n_sims = 100000
    
    print(f"\nSpot = {S0}, T = {T:.4f} ans, N = {n_sims:,} simulations")
    print("\n" + "-" * 80)
    print(f"{'Strike':>8} | {'Moneyness':>12} | {'BS Call':>10} | {'MC Call':>10} | {'Erreur':>10}")
    print("-" * 80)
    
    for K in strikes:
        moneyness = S0 / K
        bs_price = black_scholes_price(S0, K, T, r, sigma, 'call')
        
        market = Market(underlying=S0, vol=sigma, rate=r, div_a=0.0, ex_div_date=None)
        option = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=K)
        mc_model = MonteCarloModel(n_sims, market, option, pricing_date, seed=42)
        mc_price = mc_model.price_european()['price']
        
        error = abs(mc_price - bs_price)
        
        if moneyness > 1.05:
            moneyness_label = "ITM"
        elif moneyness < 0.95:
            moneyness_label = "OTM"
        else:
            moneyness_label = "ATM"
        
        print(f"{K:>8.0f} | {moneyness_label:>12} | {bs_price:>10.6f} | {mc_price:>10.6f} | {error:>10.6f}")


def test_different_volatilities():
    """Test pour différentes volatilités"""
    
    print("\n\n" + "=" * 80)
    print("COMPARAISON POUR DIFFÉRENTES VOLATILITÉS")
    print("=" * 80)
    
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    T = (maturity_date - pricing_date).days / 365.0
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    
    volatilities = [0.10, 0.20, 0.30, 0.40, 0.50]
    n_sims = 100000
    
    print(f"\nSpot = {S0}, Strike = {K}, T = {T:.4f} ans, N = {n_sims:,} simulations")
    print("\n" + "-" * 80)
    print(f"{'Volatilité':>12} | {'BS Call':>10} | {'MC Call':>10} | {'Erreur':>10} | {'Erreur %':>10}")
    print("-" * 80)
    
    for sigma in volatilities:
        bs_price = black_scholes_price(S0, K, T, r, sigma, 'call')
        
        market = Market(underlying=S0, vol=sigma, rate=r, div_a=0.0, ex_div_date=None)
        option = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=K)
        mc_model = MonteCarloModel(n_sims, market, option, pricing_date, seed=42)
        mc_price = mc_model.price_european()['price']
        
        error = abs(mc_price - bs_price)
        error_pct = (error / bs_price) * 100
        
        print(f"{sigma*100:>11.0f}% | {bs_price:>10.6f} | {mc_price:>10.6f} | {error:>10.6f} | {error_pct:>9.4f}%")


if __name__ == "__main__":
    compare_methods()
    test_different_moneyness()
    test_different_volatilities()
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("- Monte Carlo converge vers Black-Scholes")
    print("- La précision augmente avec le nombre de simulations")
    print("- L'erreur diminue en 1/√N")
    print("- Monte Carlo fonctionne pour tous les payoffs (pas seulement européens)")
    print("=" * 80)
