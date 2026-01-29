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
    
    def price_american_naive(self, num_steps: int = 252, antithetic=True) -> dict:
        """
        Price an American option using NAIVE Monte Carlo with backward induction (SCALAR)
        
        Process:
        1. Generate multiple paths with num_steps intermediate steps
        2. At each step, calculate S(t)
        3. Backward induction: move from T to t0
           - At T: payoff = IV(S_T)
           - At T-1: value = max(IV(S_T-1), discounted_value_T)
           - Continue backward to t=0
        4. Average across all paths
        
        Parameters:
        -----------
        num_steps : int
            Number of steps from 0 to maturity (default 252 = daily for 1 year)
        antithetic : bool
            Use antithetic variates for variance reduction
        
        Returns:
        --------
        dict: {'price': float, 'std_error': float, 'paths': list}
        """
        # Calculate time to maturity in years
        T = (self.option.mat_date - self.pricing_date).days / 365.0
        
        if T <= 0:
            return {
                'price': 0.0,
                'std_error': 0.0,
                'paths': []
            }
        
        # Market parameters
        S0 = self.market.underlying
        sigma = self.market.vol
        r = self.market.rate
        
        # Dividend adjustment
        q = 0.0
        if self.market.ex_div_date and self.market.ex_div_date < self.option.mat_date:
            q = self.market.div_a / S0 if S0 > 0 else 0.0
        
        # Time step
        dt = T / num_steps
        df = np.exp(-r * dt)  # Discount factor per step
        
        american_prices = []
        
        if antithetic:
            num_paths = self.num_simulations // 2
        else:
            num_paths = self.num_simulations
        
        # Generate paths
        for path_idx in range(num_paths):
            # Initialize path: S(0)
            S_path = [S0]
            S_path_anti = [S0]
            
            # Generate the path forward (scalar: one step at a time)
            for step in range(num_steps):
                # Normal random draw
                Z = np.random.standard_normal()
                dW = Z * np.sqrt(dt)
                
                # Next spot price
                drift = (r - q - 0.5 * sigma**2) * dt
                S_next = S_path[-1] * np.exp(drift + sigma * dW)
                S_path.append(S_next)
                
                # Antithetic path
                S_next_anti = S_path_anti[-1] * np.exp(drift + sigma * (-dW))
                S_path_anti.append(S_next_anti)
            
            # Backward induction (from T to 0)
            # Main path
            value = self.option.pay_off(S_path[-1])  # Payoff at maturity
            for step in range(num_steps - 1, -1, -1):
                continuation = value * df
                intrinsic = self.option.pay_off(S_path[step])
                value = max(intrinsic, continuation)
            
            american_price_main = value
            
            # Antithetic path
            value_anti = self.option.pay_off(S_path_anti[-1])
            for step in range(num_steps - 1, -1, -1):
                continuation = value_anti * df
                intrinsic = self.option.pay_off(S_path_anti[step])
                value_anti = max(intrinsic, continuation)
            
            american_price_anti = value_anti
            
            if antithetic:
                # Average the pair
                paired_avg = (american_price_main + american_price_anti) / 2
                american_prices.append(paired_avg)
            else:
                american_prices.append(american_price_main)
        
        # Calculate price and std error
        price = np.mean(american_prices)
        std_error = np.std(american_prices, ddof=1) / np.sqrt(len(american_prices))
        
        return {
            'price': price,
            'std_error': std_error,
            'num_steps': num_steps,
            'payoffs': american_prices
        }
    
    def price_american_naive_vectorized(self, num_steps: int = 252, antithetic=True) -> dict:
        """
        Price an American option using VECTORIZED Monte Carlo with backward induction
        
        Same algorithm as price_american_naive but vectorized for better performance.
        
        Parameters:
        -----------
        num_steps : int
            Number of steps from 0 to maturity
        antithetic : bool
            Use antithetic variates
        
        Returns:
        --------
        dict: {'price': float, 'std_error': float, 'paths': ndarray}
        """
        # Calculate time to maturity in years
        T = (self.option.mat_date - self.pricing_date).days / 365.0
        
        if T <= 0:
            return {
                'price': 0.0,
                'std_error': 0.0,
                'num_steps': num_steps,
                'payoffs': np.array([])
            }
        
        # Market parameters
        S0 = self.market.underlying
        sigma = self.market.vol
        r = self.market.rate
        
        # Dividend adjustment
        q = 0.0
        if self.market.ex_div_date and self.market.ex_div_date < self.option.mat_date:
            q = self.market.div_a / S0 if S0 > 0 else 0.0
        
        # Time step
        dt = T / num_steps
        df = np.exp(-r * dt)
        
        if antithetic:
            num_paths = self.num_simulations // 2
        else:
            num_paths = self.num_simulations
        
        # Generate entire matrix of Brownian motions at once: (num_paths, num_steps)
        Z_matrix = np.random.standard_normal((num_paths, num_steps))
        dW_matrix = Z_matrix * np.sqrt(dt)
        
        # Generate stock prices for all paths at all steps
        # Shape: (num_paths, num_steps + 1)
        S_paths = np.zeros((num_paths, num_steps + 1))
        S_paths[:, 0] = S0
        
        drift = (r - q - 0.5 * sigma**2) * dt
        
        # Forward path generation (vectorized)
        for step in range(num_steps):
            S_paths[:, step + 1] = S_paths[:, step] * np.exp(drift + sigma * dW_matrix[:, step])
        
        # Antithetic paths
        S_paths_anti = np.zeros((num_paths, num_steps + 1))
        S_paths_anti[:, 0] = S0
        
        for step in range(num_steps):
            S_paths_anti[:, step + 1] = S_paths_anti[:, step] * np.exp(drift + sigma * (-dW_matrix[:, step]))
        
        # Backward induction for main paths
        # Start at maturity
        if self.option.is_a_call():
            values = np.maximum(S_paths[:, -1] - self.option.strike, 0)
            values_anti = np.maximum(S_paths_anti[:, -1] - self.option.strike, 0)
        else:  # put
            values = np.maximum(self.option.strike - S_paths[:, -1], 0)
            values_anti = np.maximum(self.option.strike - S_paths_anti[:, -1], 0)
        
        # Move backward (vectorized)
        for step in range(num_steps - 1, -1, -1):
            continuation = values * df
            if self.option.is_a_call():
                intrinsic = np.maximum(S_paths[:, step] - self.option.strike, 0)
                intrinsic_anti = np.maximum(S_paths_anti[:, step] - self.option.strike, 0)
            else:  # put
                intrinsic = np.maximum(self.option.strike - S_paths[:, step], 0)
                intrinsic_anti = np.maximum(self.option.strike - S_paths_anti[:, step], 0)
            
            values = np.maximum(intrinsic, continuation)
            
            continuation_anti = values_anti * df
            values_anti = np.maximum(intrinsic_anti, continuation_anti)
        
        # Average the pairs if antithetic
        if antithetic:
            american_prices = (values + values_anti) / 2
        else:
            american_prices = values
        
        # Calculate price and std error
        price = np.mean(american_prices)
        std_error = np.std(american_prices, ddof=1) / np.sqrt(len(american_prices))
        
        return {
            'price': price,
            'std_error': std_error,
            'num_steps': num_steps,
            'payoffs': american_prices
        }
    
    def compare_american_naive_scalar_vs_vectorized(self, num_steps: int = 252, antithetic=True) -> dict:
        """
        Compare scalar and vectorized American pricing to verify they give same results
        
        Returns:
        --------
        dict with both results and comparison metrics
        """
        import time
        
        # Reset seed for fair comparison
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        # Time scalar version
        start_scalar = time.time()
        result_scalar = self.price_american_naive(num_steps=num_steps, antithetic=antithetic)
        time_scalar = time.time() - start_scalar
        
        # Reset seed for fair comparison
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        # Time vectorized version
        start_vector = time.time()
        result_vector = self.price_american_naive_vectorized(num_steps=num_steps, antithetic=antithetic)
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
            'speedup': time_scalar / time_vector if time_vector > 0 else float('inf'),
            'num_steps': num_steps
        }

    def price_american_longstaff_schwartz_vectorized(self, num_steps: int = 252, poly_degree: int = 3,
                                                     antithetic=True) -> dict:
        """
        Price an American option using Longstaff-Schwartz method (VECTORIZED)
        
        Uses polynomial regression on ALL paths to estimate continuation values:
        1. Generate paths forward
        2. Backward induction with regression at each step:
           - Fit polynomial: E[payoff_future | S_t] = a0 + a1*S + a2*S^2 + ...
           - Decide: max(intrinsic, estimated_continuation)
        
        Parameters:
        -----------
        num_steps : int
            Number of steps in the path
        poly_degree : int
            Degree of polynomial for regression (default 3)
        antithetic : bool
            Use antithetic variates
        
        Returns:
        --------
        dict: {'price': float, 'std_error': float}
        """
        # Calculate time to maturity in years
        T = (self.option.mat_date - self.pricing_date).days / 365.0
        
        if T <= 0:
            return {
                'price': 0.0,
                'std_error': 0.0,
                'num_steps': num_steps,
                'poly_degree': poly_degree,
                'payoffs': np.array([])
            }
        
        # Market parameters
        S0 = self.market.underlying
        sigma = self.market.vol
        r = self.market.rate
        
        # Dividend adjustment
        q = 0.0
        if self.market.ex_div_date and self.market.ex_div_date < self.option.mat_date:
            q = self.market.div_a / S0 if S0 > 0 else 0.0
        
        # Time step
        dt = T / num_steps
        df = np.exp(-r * dt)
        
        if antithetic:
            num_paths = self.num_simulations // 2
        else:
            num_paths = self.num_simulations
        
        # Generate entire matrix of Brownian motions: (num_paths, num_steps)
        Z_matrix = np.random.standard_normal((num_paths, num_steps))
        dW_matrix = Z_matrix * np.sqrt(dt)
        
        # Generate stock prices for all paths at all steps
        S_paths = np.zeros((num_paths, num_steps + 1))
        S_paths[:, 0] = S0
        
        drift = (r - q - 0.5 * sigma**2) * dt
        
        # Forward path generation (vectorized)
        for step in range(num_steps):
            S_paths[:, step + 1] = S_paths[:, step] * np.exp(drift + sigma * dW_matrix[:, step])
        
        # Antithetic paths
        S_paths_anti = np.zeros((num_paths, num_steps + 1))
        S_paths_anti[:, 0] = S0
        
        for step in range(num_steps):
            S_paths_anti[:, step + 1] = S_paths_anti[:, step] * np.exp(drift + sigma * (-dW_matrix[:, step]))
        
        # Backward induction with Longstaff-Schwartz regression
        # Initialize cash flow with payoff at maturity
        if self.option.is_a_call():
            cash_flow = np.maximum(S_paths[:, -1] - self.option.strike, 0)
            cash_flow_anti = np.maximum(S_paths_anti[:, -1] - self.option.strike, 0)
        else:  # put
            cash_flow = np.maximum(self.option.strike - S_paths[:, -1], 0)
            cash_flow_anti = np.maximum(self.option.strike - S_paths_anti[:, -1], 0)
        
        # Backward induction: move from T-1 to 0
        for step in range(num_steps - 1, -1, -1):
            # Discount future cash flows
            continuation_value = cash_flow * df
            continuation_value_anti = cash_flow_anti * df
            
            # Intrinsic value at this step
            if self.option.is_a_call():
                intrinsic = np.maximum(S_paths[:, step] - self.option.strike, 0)
                intrinsic_anti = np.maximum(S_paths_anti[:, step] - self.option.strike, 0)
            else:  # put
                intrinsic = np.maximum(self.option.strike - S_paths[:, step], 0)
                intrinsic_anti = np.maximum(self.option.strike - S_paths_anti[:, step], 0)
            
            # Regression for continuation value estimation (for in-the-money paths)
            # Only regress on paths where exercise might be optimal (intrinsic > 0)
            itm_mask = intrinsic > 0
            itm_mask_anti = intrinsic_anti > 0
            
            if np.sum(itm_mask) > poly_degree + 1:
                # Fit polynomial on ITM paths
                S_itm = S_paths[itm_mask, step]
                cf_itm = continuation_value[itm_mask]
                
                # Polynomial fit
                coeffs = np.polyfit(S_itm, cf_itm, poly_degree)
                
                # Estimate continuation for all paths
                estimated_continuation = np.polyval(coeffs, S_paths[:, step])
                
                # Ensure non-negative
                estimated_continuation = np.maximum(estimated_continuation, 0)
                
                # Exercise decision: compare intrinsic vs estimated continuation
                cash_flow = np.where(intrinsic > estimated_continuation, intrinsic, continuation_value)
            else:
                # Not enough ITM paths, use continuation directly
                cash_flow = np.where(intrinsic > continuation_value, intrinsic, continuation_value)
            
            # Same for antithetic
            if np.sum(itm_mask_anti) > poly_degree + 1:
                S_itm_anti = S_paths_anti[itm_mask_anti, step]
                cf_itm_anti = continuation_value_anti[itm_mask_anti]
                
                coeffs_anti = np.polyfit(S_itm_anti, cf_itm_anti, poly_degree)
                estimated_continuation_anti = np.polyval(coeffs_anti, S_paths_anti[:, step])
                estimated_continuation_anti = np.maximum(estimated_continuation_anti, 0)
                
                cash_flow_anti = np.where(intrinsic_anti > estimated_continuation_anti, intrinsic_anti, continuation_value_anti)
            else:
                cash_flow_anti = np.where(intrinsic_anti > continuation_value_anti, intrinsic_anti, continuation_value_anti)
        
        # Average the pairs if antithetic
        if antithetic:
            ls_prices = (cash_flow + cash_flow_anti) / 2
        else:
            ls_prices = cash_flow
        
        # Calculate price and std error
        price = np.mean(ls_prices)
        std_error = np.std(ls_prices, ddof=1) / np.sqrt(len(ls_prices)) if len(ls_prices) > 1 else 0.0
        
        return {
            'price': price,
            'std_error': std_error,
            'num_steps': num_steps,
            'poly_degree': poly_degree,
            'payoffs': ls_prices
        }
    
    def compare_american_naive_vs_ls(self, num_steps: int = 252, poly_degree: int = 3,
                                     antithetic=True) -> dict:
        """
        Compare Naive American pricing with Longstaff-Schwartz
        
        Returns:
        --------
        dict with comparison results
        """
        import time
        
        # Reset seed for fair comparison
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        # Time naive version
        start_naive = time.time()
        result_naive = self.price_american_naive_vectorized(num_steps=num_steps, antithetic=antithetic)
        time_naive = time.time() - start_naive
        
        # Reset seed for fair comparison
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        # Time LS version
        start_ls = time.time()
        result_ls = self.price_american_longstaff_schwartz_vectorized(num_steps=num_steps, 
                                                                       poly_degree=poly_degree,
                                                                       antithetic=antithetic)
        time_ls = time.time() - start_ls
        
        return {
            'naive': {
                'price': result_naive['price'],
                'std_error': result_naive['std_error'],
                'time': time_naive
            },
            'longstaff_schwartz': {
                'price': result_ls['price'],
                'std_error': result_ls['std_error'],
                'time': time_ls
            },
            'price_difference': abs(result_naive['price'] - result_ls['price']),
            'time_ratio': time_ls / time_naive if time_naive > 0 else float('inf'),
            'num_steps': num_steps,
            'poly_degree': poly_degree
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
    