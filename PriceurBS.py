# PriceurBS.py with continuous div 
import math
from datetime import date
from scipy.stats import norm


class Option:
    def __init__(self, S: float, K: float, r: float, T: float, sigma: float,
                 div_a: float = 0.0, ex_div_date: date = None, pricing_date: date = None):
        """
        Option de base compatible avec dividende discret.
        - div_a : montant du dividende (en cash)
        - ex_div_date : date du détachement du dividende (datetime.date)
        - pricing_date : date de valorisation (datetime.date)
        """
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.div_a = div_a
        self.ex_div_date = ex_div_date
        self.pricing_date = pricing_date or date.today()

    # --- Ajustement du spot si dividende à venir ---
    def effective_spot(self):
        if self.div_a > 0.0 and self.ex_div_date is not None:
            delta_days = (self.ex_div_date - self.pricing_date).days
            if 0 < delta_days < self.T * 365:  # dividende avant maturité
                t_div = delta_days / 365.0
                pv_div = self.div_a * math.exp(-self.r * t_div)
                return self.S - pv_div
        return self.S

    # --- Black–Scholes core formulas ---
    def d1(self):
        S_eff = self.effective_spot()
        return (math.log(S_eff / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * math.sqrt(self.T)
        )

    def d2(self):
        return self.d1() - self.sigma * math.sqrt(self.T)

    def vega(self):
        S_eff = self.effective_spot()
        return S_eff * norm.pdf(self.d1()) * math.sqrt(self.T)


class Call(Option):
    def price(self):
        S_eff = self.effective_spot()
        d1, d2 = self.d1(), self.d2()
        return S_eff * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)

    def delta(self):
        return norm.cdf(self.d1())

    def rho(self):
        return self.T * self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2())

    def theta(self):
        S_eff = self.effective_spot()
        d1, d2 = self.d1(), self.d2()
        return (-(S_eff * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T))
                - self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2))

    def theta_per_day(self):
        return self.theta() / 365.0


class Put(Option):
    def price(self):
        S_eff = self.effective_spot()
        d1, d2 = self.d1(), self.d2()
        return self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - S_eff * norm.cdf(-d1)

    def delta(self):
        return norm.cdf(self.d1()) - 1.0

    def rho(self):
        return -self.T * self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2())

    def rho_per_1pct(self):
        return 0.01 * self.rho()

    def theta(self):
        S_eff = self.effective_spot()
        d1, d2 = self.d1(), self.d2()
        return (-(S_eff * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T))
                + self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-d2))

    def theta_per_day(self):
        return self.theta() / 365.0


# --- Test simple ---
if __name__ == "__main__":
    from datetime import timedelta

    pricing_date = date.today()
    ex_div_date = pricing_date + timedelta(days=90)  # 3 mois avant maturité
    T = 1.0

    call = Call(S=100, K=100, r=0.05, T=T, sigma=0.2,
                div_a=5.0, ex_div_date=ex_div_date, pricing_date=pricing_date)
    put = Put(S=100, K=100, r=0.05, T=T, sigma=0.2,
              div_a=5.0, ex_div_date=ex_div_date, pricing_date=pricing_date)

    print(f"Call with dividend: {call.price():.4f}")
    print(f"Put  with dividend: {put.price():.4f}")
    print(f"Effective spot used: {call.effective_spot():.4f}")

