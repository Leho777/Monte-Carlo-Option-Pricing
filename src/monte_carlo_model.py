import random
import numpy as np
from datetime import date
from .market import Market
from .option_trade import OptionTrade


class MonteCarloModel:
    def __init__(self, num_simulations: int, market: Market, option: OptionTrade, 
                 pricing_date: date, seed=None):
        """
        Monte Carlo Model for European Option Pricing
        
        Parameters:
        -----------
        num_simulations : int
            Number of Monte Carlo paths (N)
        market : Market
            Market parameters (underlying, vol, rate, div)
        option : OptionTrade
            Option parameters (strike, maturity, call/put)
        pricing_date : date
            Valuation date
        seed : int, optional
            Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def price_european(self, antithetic=True) -> dict:
        """
        Price a European option using Monte Carlo simulation
        
        Process:
        1. Repeat N times:
           a. Draw a random number between 0 and 1
           b. Convert it to a normal draw
           c. Convert it to a Brownian draw
           d. Deduce the value of the underlying at T
           e. Calculate the value of the option at T
           f. Discount to today, this is one result
        2. Average the N results: this is the price
        
        Returns:
        --------
        dict: {'price': float, 'std_error': float, 'payoffs': list}
        """
        # Calculate time to maturity in years
        T = (self.option.mat_date - self.pricing_date).days / 365.0
        
        if T <= 0:
            # Option has expired
            return {
                'price': 0.0,
                'std_error': 0.0,
                'payoffs': []
            }
        
        # Market parameters
        S0 = self.market.underlying
        sigma = self.market.vol
        r = self.market.rate
        
        # Adjust for dividends (simple approach: reduce drift)
        # If there's a dividend before maturity, adjust the forward
        q = 0.0  # dividend yield (simplified)
        if self.market.ex_div_date and self.market.ex_div_date < self.option.mat_date:
            # Approximate dividend yield
            q = self.market.div_a / S0 if S0 > 0 else 0.0
        
        discounted_payoffs = []
        if antithetic:
            num_paths = self.num_simulations // 2
        else:
            num_paths = self.num_simulations
            
        for i in range(num_paths):
            # Step 1-2: Draw a normal random variable N(0,1)
            # (Note: We directly use standard_normal instead of uniform + inverse transform)
            Z = np.random.standard_normal()
            
            # Step 3: Convert it to a Brownian draw W(T)
            # W(T) ~ N(0, T)
            W_T = Z * np.sqrt(T)
            
            # Step 4: Deduce the value of the underlying at T
            # Using Black-Scholes formula: S(T) = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*W(T))
            S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * W_T)
            
            # Step 5: Calculate the value of the option at T (payoff)
            payoff = self.option.pay_off(S_T)
            
            # Step 6: Discount to today
            discounted_payoff = payoff * np.exp(-r * T)
            
            if antithetic:
                # Generate antithetic path using -W_T (negatively correlated)
                S_T_antithetic = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * (-W_T))
                payoff_antithetic = self.option.pay_off(S_T_antithetic)
                discounted_payoff_antithetic = payoff_antithetic * np.exp(-r * T)
                
                # KEY: Average the pair to capture variance reduction
                # Var(avg) = (Var(Y) + Var(Y_anti) + 2*Cov(Y, Y_anti)) / 4
                # Since Cov < 0, this reduces variance
                paired_avg = (discounted_payoff + discounted_payoff_antithetic) / 2
                discounted_payoffs.append(paired_avg)
            else:
                discounted_payoffs.append(discounted_payoff)
        
        # Step 7: Average the N results - this is the price
        price = np.mean(discounted_payoffs)
        # Standard error based on the actual samples (paired averages if antithetic)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
        
        return {
            'price': price,
            'std_error': std_error,
            'payoffs': discounted_payoffs
        }
    
    def price_european_vectorized(self, antithetic=True) -> dict:
        """
        Price a European option using VECTORIZED Monte Carlo simulation
        
        Same algorithm as price_european but using NumPy vectorization
        for better performance.
        
        Returns:
        --------
        dict: {'price': float, 'std_error': float, 'payoffs': ndarray}
        """
        # Calculate time to maturity in years
        T = (self.option.mat_date - self.pricing_date).days / 365.0
        
        if T <= 0:
            return {
                'price': 0.0,
                'std_error': 0.0,
                'payoffs': np.array([])
            }
        
        # Market parameters
        S0 = self.market.underlying
        sigma = self.market.vol
        r = self.market.rate
        
        # Adjust for dividends
        q = 0.0
        if self.market.ex_div_date and self.market.ex_div_date < self.option.mat_date:
            q = self.market.div_a / S0 if S0 > 0 else 0.0
        
        if antithetic:
            num_paths = self.num_simulations // 2
            
            # Step 2: Generate all normal draws at once (vectorized)
            Z = np.random.standard_normal(num_paths)
            
            # Step 3: Convert to Brownian draws W(T) ~ N(0, T)
            W_T = Z * np.sqrt(T)
            
            # Step 4: Calculate all S(T) at once (vectorized)
            drift = (r - q - 0.5 * sigma**2) * T
            S_T = S0 * np.exp(drift + sigma * W_T)
            S_T_antithetic = S0 * np.exp(drift + sigma * (-W_T))
            
            # Step 5: Calculate payoffs (vectorized)
            # Need to handle vectorized payoff calculation
            if self.option.is_a_call():
                payoffs = np.maximum(S_T - self.option.strike, 0)
                payoffs_anti = np.maximum(S_T_antithetic - self.option.strike, 0)
            else:  # put
                payoffs = np.maximum(self.option.strike - S_T, 0)
                payoffs_anti = np.maximum(self.option.strike - S_T_antithetic, 0)
            
            # Step 6: Discount to today (vectorized)
            discount_factor = np.exp(-r * T)
            discounted_payoffs = payoffs * discount_factor
            discounted_payoffs_anti = payoffs_anti * discount_factor
            
            # Average the pairs for variance reduction
            paired_averages = (discounted_payoffs + discounted_payoffs_anti) / 2
            
            # Step 7: Calculate price and standard error
            price = np.mean(paired_averages)
            std_error = np.std(paired_averages, ddof=1) / np.sqrt(len(paired_averages))
            
            return {
                'price': price,
                'std_error': std_error,
                'payoffs': paired_averages
            }
        else:
            # Step 2: Generate all normal draws at once (vectorized)
            Z = np.random.standard_normal(self.num_simulations)
            
            # Step 3: Convert to Brownian draws
            W_T = Z * np.sqrt(T)
            
            # Step 4: Calculate all S(T) at once
            drift = (r - q - 0.5 * sigma**2) * T
            S_T = S0 * np.exp(drift + sigma * W_T)
            
            # Step 5: Calculate payoffs (vectorized)
            if self.option.is_a_call():
                payoffs = np.maximum(S_T - self.option.strike, 0)
            else:  # put
                payoffs = np.maximum(self.option.strike - S_T, 0)
            
            # Step 6: Discount to today
            discount_factor = np.exp(-r * T)
            discounted_payoffs = payoffs * discount_factor
            
            # Step 7: Calculate price and standard error
            price = np.mean(discounted_payoffs)
            std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
            
            return {
                'price': price,
                'std_error': std_error,
                'payoffs': discounted_payoffs
            }
    
    def compare_scalar_vs_vectorized(self, antithetic=True) -> dict:
        """
        Compare scalar and vectorized implementations to verify they give same results
        
        Uses the same seed for both to ensure identical random draws.
        
        Returns:
        --------
        dict with both results and comparison metrics
        """
        import time
        
        # Save current seed state
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        # Time scalar version
        start_scalar = time.time()
        result_scalar = self.price_european(antithetic=antithetic)
        time_scalar = time.time() - start_scalar
        
        # Reset seed for fair comparison
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        # Time vectorized version
        start_vector = time.time()
        result_vector = self.price_european_vectorized(antithetic=antithetic)
        time_vector = time.time() - start_vector
        
        return {
            'scalar': {
                'price': result_scalar['price'],
                'std_error': result_scalar['std_error'],
                'time': time_scalar
            },
            'vectorized': {
                'price': result_vector['price'],
                'std_error': result_vector['std_error'],
                'time': time_vector
            },
            'price_difference': abs(result_scalar['price'] - result_vector['price']),
            'speedup': time_scalar / time_vector if time_vector > 0 else float('inf')
        }
    
    def run_simulation(self, model_input):
        """Legacy method for backward compatibility"""
        results = []
        for _ in range(self.num_simulations):
            result = self._simulate(model_input)
            results.append(result)
        return results

    def _simulate(self, model_input):
        """Legacy method for backward compatibility"""
        import random
        simulated_value = model_input + random.uniform(-1, 1)
        return simulated_value
    
    def calculate_average(self, results):
        return sum(results) / len(results) if results else 0
    
    def calculate_variance(self, results):
        avg = self.calculate_average(results)
        return sum((x - avg) ** 2 for x in results) / len(results) if results else 0
    