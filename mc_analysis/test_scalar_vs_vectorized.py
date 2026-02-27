"""
Test script to compare scalar vs vectorized Monte Carlo implementations
"""
from datetime import date
from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel

def test_comparison():
    # Setup market and option
    pricing_date = date(2026, 1, 22)
    maturity_date = date(2026, 7, 22)  # 6 months
    
    market = Market(
        underlying=100.0,
        vol=0.2,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    # OptionTrade(mat, call_put, ex, k)
    option_call = OptionTrade(
        mat=maturity_date,
        call_put='CALL',
        ex='EUROPEAN',
        k=100.0
    )
    
    option_put = OptionTrade(
        mat=maturity_date,
        call_put='PUT',
        ex='EUROPEAN',
        k=100.0
    )
    
    print("=" * 70)
    print("COMPARISON: SCALAR vs VECTORIZED Monte Carlo")
    print("=" * 70)
    
    for num_sims in [1000, 10000, 100000]:
        print(f"\n{'='*70}")
        print(f"Number of simulations: {num_sims:,}")
        print("=" * 70)
        
        for option, name in [(option_call, "CALL"), (option_put, "PUT")]:
            print(f"\n--- {name} Option (K=100, S0=100, σ=20%, r=5%, T=6m) ---")
            
            for antithetic in [False, True]:
                anti_str = "WITH antithetic" if antithetic else "WITHOUT antithetic"
                print(f"\n  {anti_str}:")
                
                # Use same seed for fair comparison
                mc = MonteCarloModel(
                    num_simulations=num_sims,
                    market=market,
                    option=option,
                    pricing_date=pricing_date,
                    seed=42
                )
                
                comparison = mc.compare_scalar_vs_vectorized(antithetic=antithetic)
                
                print(f"    Scalar:     Price = {comparison['scalar']['price']:.15f} "
                      f"(±{comparison['scalar']['std_error']:.6f}), "
                      f"Time = {comparison['scalar']['time']*1000:.2f} ms")
                print(f"    Vectorized: Price = {comparison['vectorized']['price']:.15f} "
                      f"(±{comparison['vectorized']['std_error']:.6f}), "
                      f"Time = {comparison['vectorized']['time']*1000:.2f} ms")
                print(f"    Difference: {comparison['price_difference']:.15f}")
                print(f"    Speedup:    {comparison['speedup']:.1f}x")

if __name__ == "__main__":
    test_comparison()
