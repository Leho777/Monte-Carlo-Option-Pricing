import math
from datetime import date
from scipy.stats import norm

from .market import Market
from .option_trade import OptionTrade


class BlackScholes:
    """
    Pricer analytique Black-Scholes pour options européennes.

    Utilise la même interface Market/OptionTrade que MonteCarloModel,
    ce qui permet une comparaison directe des prix.

    Dividende discret : ajustement du spot par la valeur présente du dividende
        S_eff = S0 - PV(div)  = S0 - div * exp(-r * t_div)
    (approximation «escrow» valide si div est petit devant S0)

    Formules :
        d1 = [ln(S_eff/K) + (r + σ²/2)*T] / (σ√T)
        d2 = d1 - σ√T
        Call = S_eff * N(d1) - K * e^{-rT} * N(d2)
        Put  = K * e^{-rT} * N(-d2) - S_eff * N(-d1)
    """

    def __init__(self, market: Market, option: OptionTrade, pricing_date: date):
        self.market = market
        self.option = option
        self.pricing_date = pricing_date

    # ------------------------------------------------------------------
    # Paramètres de base
    # ------------------------------------------------------------------

    def _T(self) -> float:
        return (self.option.mat_date - self.pricing_date).days / 365.0

    def _effective_spot(self) -> float:
        """S ajusté de la valeur présente du dividende discret."""
        S = self.market.underlying
        if (self.market.div_a > 0 and self.market.ex_div_date is not None
                and self.pricing_date < self.market.ex_div_date < self.option.mat_date):
            t_div = (self.market.ex_div_date - self.pricing_date).days / 365.0
            S -= self.market.div_a * math.exp(-self.market.rate * t_div)
        return max(S, 1e-10)   # éviter log(0)

    def _d1_d2(self):
        T = self._T()
        S_eff = self._effective_spot()
        K, r, sigma = self.option.strike, self.market.rate, self.market.vol
        d1 = (math.log(S_eff / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    # ------------------------------------------------------------------
    # Prix
    # ------------------------------------------------------------------

    def price(self) -> float:
        """
        Prix analytique BS.
        Retourne 0.0 si la maturité est dépassée ou si l'option n'est pas CALL/PUT.
        """
        T = self._T()
        if T <= 0:
            return max(self.option.pay_off(self.market.underlying), 0.0)

        d1, d2 = self._d1_d2()
        K, r = self.option.strike, self.market.rate
        S_eff = self._effective_spot()
        df = math.exp(-r * T)

        if self.option.is_a_call():
            return S_eff * norm.cdf(d1) - K * df * norm.cdf(d2)
        elif self.option.is_a_put():
            return K * df * norm.cdf(-d2) - S_eff * norm.cdf(-d1)
        return 0.0

    # ------------------------------------------------------------------
    # Grecques (first order)
    # ------------------------------------------------------------------

    def delta(self) -> float:
        """dV/dS"""
        d1, _ = self._d1_d2()
        if self.option.is_a_call():
            return norm.cdf(d1)
        elif self.option.is_a_put():
            return norm.cdf(d1) - 1.0
        return 0.0

    def gamma(self) -> float:
        """d²V/dS²"""
        T = self._T()
        d1, _ = self._d1_d2()
        S_eff = self._effective_spot()
        return norm.pdf(d1) / (S_eff * self.market.vol * math.sqrt(T))

    def vega(self) -> float:
        """dV/dσ  (pour 1 point de vol, i.e. +1%)"""
        T = self._T()
        d1, _ = self._d1_d2()
        return self._effective_spot() * norm.pdf(d1) * math.sqrt(T) * 0.01

    def theta(self) -> float:
        """dV/dT (par jour calendaire)"""
        T = self._T()
        d1, d2 = self._d1_d2()
        S_eff = self._effective_spot()
        K, r, sigma = self.option.strike, self.market.rate, self.market.vol
        time_decay = -(S_eff * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        if self.option.is_a_call():
            return (time_decay - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
        elif self.option.is_a_put():
            return (time_decay + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
        return 0.0

    def rho(self) -> float:
        """dV/dr (pour 1 bp = 0.01%)"""
        T = self._T()
        _, d2 = self._d1_d2()
        K, r = self.option.strike, self.market.rate
        if self.option.is_a_call():
            return T * K * math.exp(-r * T) * norm.cdf(d2) * 0.0001
        elif self.option.is_a_put():
            return -T * K * math.exp(-r * T) * norm.cdf(-d2) * 0.0001
        return 0.0

    def summary(self) -> dict:
        """Retourne prix + toutes les grecques dans un dict."""
        return {
            'price': self.price(),
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega':  self.vega(),
            'theta': self.theta(),
            'rho':   self.rho(),
        }
