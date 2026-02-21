"""
main.py — Comparaison des méthodes de pricing d'options

Méthodes :
  - Black-Scholes (analytique, européen uniquement)
  - Arbre Trinomial (européen & américain)
  - Monte-Carlo Longstaff-Schwartz (européen & américain)

Options testées :
  - European Call / Put
  - American  Call / Put

Usage :
    python main.py
"""

import time
from datetime import date

import numpy as np

from src.market import Market
from src.option_trade import OptionTrade
from src.tree import Tree
from src.trinomial_model import TrinomialModel
from src.monte_carlo_model import MonteCarloModel
from src.black_scholes import BlackScholes
from src.regression import BasisType


# ══════════════════════════════════════════════════════════════════════════════
# Paramètres communs  (modifier ici)
# ══════════════════════════════════════════════════════════════════════════════
PRICING_DATE  = date(2025, 10, 29)
MATURITY      = date(2026,  8, 27)

UNDERLYING    = 102.45
STRIKE        = 100.00
VOL           = 0.28
RATE          = 0.04
DIV_AMOUNT    = 3.0
EX_DIV_DATE   = date(2026,  6,  9)   # None si pas de dividende

# Arbre trinomial
TREE_STEPS    = 500

# Monte Carlo
MC_PATHS      = 50_000
MC_STEPS_AM   = 100        # pas de temps pour l'américain LS
MC_ANTITHETIC = True
MC_SEED       = 42
MC_BASIS      = BasisType.LAGUERRE

# ══════════════════════════════════════════════════════════════════════════════
# Helpers d'affichage
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "─" * 76
SEP2 = "═" * 76

def _row(label: str, price: float, se: float = 0.0,
         ref: float = None, elapsed: float = 0.0) -> str:
    """Ligne formatée du tableau."""
    err_str = ""
    if ref is not None and ref > 0:
        err_str = f"{(price - ref) / ref * 100:+7.3f}%"
    se_str  = f"±{se:.4f}" if se > 0 else "        "
    t_str   = f"{elapsed:.2f}s" if elapsed > 0 else "     "
    return (f"  {label:<30}  {price:>8.4f}  {se_str:<10}  "
            f"{err_str:<10}  {t_str}")

def _header(title: str, ref_label: str = "") -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    if ref_label:
        print(f"  Référence : {ref_label}")
    print(SEP2)
    print(f"  {'Méthode':<30}  {'Prix':>8}  {'Err. std':<10}  "
          f"{'Err. rel':<10}  {'Temps'}")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# Constructeurs réutilisables
# ══════════════════════════════════════════════════════════════════════════════

def _build_tree(market: Market, option: OptionTrade) -> tuple:
    """Construit et retourne (TrinomialModel, prix, temps)."""
    tree = Tree(TREE_STEPS, market, option, PRICING_DATE, prunning_threshold=1e-8)
    t0 = time.perf_counter()
    tree.build_tree()
    mod = TrinomialModel(PRICING_DATE, tree)
    price = mod.price(option, "backward")
    elapsed = time.perf_counter() - t0
    return mod, price, elapsed


def _mc_european(market: Market, option: OptionTrade) -> tuple:
    """Monte Carlo européen vectorisé. Retourne (prix, se, temps)."""
    mc = MonteCarloModel(MC_PATHS, market, option, PRICING_DATE, seed=MC_SEED)
    t0 = time.perf_counter()
    res = mc.price_european_vectorized(antithetic=MC_ANTITHETIC)
    elapsed = time.perf_counter() - t0
    return res['price'], res['std_error'], elapsed


def _mc_american(market: Market, option: OptionTrade) -> tuple:
    """Monte Carlo américain Longstaff-Schwartz. Retourne (prix, se, temps)."""
    mc = MonteCarloModel(MC_PATHS, market, option, PRICING_DATE, seed=MC_SEED)
    t0 = time.perf_counter()
    res = mc.price_american_longstaff_schwartz_vectorized(
        num_steps=MC_STEPS_AM,
        poly_degree=3,
        poly_basis=MC_BASIS,
        residual_threshold=0.0,
        antithetic=MC_ANTITHETIC,
    )
    elapsed = time.perf_counter() - t0
    return res['price'], res['std_error'], elapsed


# ══════════════════════════════════════════════════════════════════════════════
# Bloc de pricing pour une option
# ══════════════════════════════════════════════════════════════════════════════

def price_option(market: Market, option: OptionTrade, label: str) -> None:
    """
    Calcule et affiche les prix par toutes les méthodes disponibles
    pour une option donnée.
    """
    is_american = option.is_american()
    is_call     = option.is_a_call()

    # ------------------------------------------------------------------
    # 1. Arbre trinomial  (référence principale, valide pour les deux types)
    # ------------------------------------------------------------------
    _, tri_price, tri_t = _build_tree(market, option)

    # ------------------------------------------------------------------
    # 2. Black-Scholes (européen uniquement)
    # ------------------------------------------------------------------
    bs_price = None
    bs_t     = 0.0
    if not is_american:
        t0 = time.perf_counter()
        bs_price = BlackScholes(market, option, PRICING_DATE).price()
        bs_t     = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 3. Monte Carlo
    # ------------------------------------------------------------------
    if is_american:
        mc_price, mc_se, mc_t = _mc_american(market, option)
    else:
        mc_price, mc_se, mc_t = _mc_european(market, option)

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------
    ref_label = f"Trinomial ({TREE_STEPS} pas)"
    _header(label, ref_label=ref_label)

    if bs_price is not None:
        print(_row("Black-Scholes (analytique)",
                   bs_price, elapsed=bs_t, ref=tri_price))

    print(_row(f"Trinomial ({TREE_STEPS} pas)",
               tri_price, elapsed=tri_t))

    mc_label = (f"MC-LS ({MC_PATHS:,} paths, {MC_BASIS.value})"
                if is_american
                else f"MC européen ({MC_PATHS:,} paths)")
    print(_row(mc_label, mc_price, se=mc_se, elapsed=mc_t, ref=tri_price))

    # Vérification parité Call-Put (si européen)
    return tri_price, bs_price, mc_price


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    market = Market(
        underlying  = UNDERLYING,
        vol         = VOL,
        rate        = RATE,
        div_a       = DIV_AMOUNT,
        ex_div_date = EX_DIV_DATE,
    )

    T = (MATURITY - PRICING_DATE).days / 365.0
    K = STRIKE

    print(f"\n{'═'*76}")
    print(f"  COMPARAISON DES MÉTHODES DE PRICING")
    print(f"{'═'*76}")
    print(f"  S₀={UNDERLYING}  K={K}  σ={VOL:.0%}  r={RATE:.0%}  "
          f"T={T:.3f}y  div={DIV_AMOUNT}@{EX_DIV_DATE}")
    print(f"  Maturité={MATURITY}  Date valuation={PRICING_DATE}")
    print(f"  MC : {MC_PATHS:,} paths  antithétique={MC_ANTITHETIC}  "
          f"base={MC_BASIS.value}  seed={MC_SEED}")

    # ── European Call ──────────────────────────────────────────────────
    eu_call = OptionTrade(mat=MATURITY, call_put='CALL', ex='EUROPEAN', k=K)
    tri_ec, bs_ec, mc_ec = price_option(market, eu_call, "EUROPEAN CALL")

    # ── European Put ───────────────────────────────────────────────────
    eu_put = OptionTrade(mat=MATURITY, call_put='PUT', ex='EUROPEAN', k=K)
    tri_ep, bs_ep, mc_ep = price_option(market, eu_put, "EUROPEAN PUT")

    # ── American Call ──────────────────────────────────────────────────
    am_call = OptionTrade(mat=MATURITY, call_put='CALL', ex='AMERICAN', k=K)
    tri_ac, _, mc_ac = price_option(market, am_call, "AMERICAN CALL")

    # ── American Put ───────────────────────────────────────────────────
    am_put = OptionTrade(mat=MATURITY, call_put='PUT', ex='AMERICAN', k=K)
    tri_ap, _, mc_ap = price_option(market, am_put, "AMERICAN PUT")

    # ── Récapitulatif ──────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  RÉCAPITULATIF")
    print(SEP2)
    print(f"  {'Option':<22}  {'Black-Scholes':>13}  {'Trinomial':>10}  {'Monte-Carlo':>12}")
    print(SEP)

    def _fmt(v): return f"{v:>12.4f}" if v is not None else f"{'N/A':>12}"

    print(f"  {'European Call':<22}  {_fmt(bs_ec)}  {tri_ec:>10.4f}  {mc_ec:>12.4f}")
    print(f"  {'European Put':<22}  {_fmt(bs_ep)}  {tri_ep:>10.4f}  {mc_ep:>12.4f}")
    print(f"  {'American Call':<22}  {_fmt(None)}  {tri_ac:>10.4f}  {mc_ac:>12.4f}")
    print(f"  {'American Put':<22}  {_fmt(None)}  {tri_ap:>10.4f}  {mc_ap:>12.4f}")

    # ── Parité Put-Call européenne ─────────────────────────────────────
    import math
    fwd_pv = UNDERLYING - K * math.exp(-RATE * T)
    # Correction dividende discret
    if DIV_AMOUNT > 0 and EX_DIV_DATE and PRICING_DATE < EX_DIV_DATE < MATURITY:
        t_div = (EX_DIV_DATE - PRICING_DATE).days / 365.0
        fwd_pv -= DIV_AMOUNT * math.exp(-RATE * t_div)

    print(f"\n{SEP}")
    print("  PARITÉ PUT-CALL EUROPÉENNE   C - P = S_eff - K·e^{-rT}")
    print(f"  Théorique (S_eff - K·e^{{-rT}}) = {fwd_pv:.4f}")
    print(f"  Tri  C-P = {tri_ec - tri_ep:.4f}   écart = {abs(tri_ec - tri_ep - fwd_pv):.4f}")
    if bs_ec and bs_ep:
        print(f"  BS   C-P = {bs_ec  - bs_ep:.4f}   écart = {abs(bs_ec  - bs_ep  - fwd_pv):.5f}")
    print(f"  MC   C-P = {mc_ec  - mc_ep:.4f}   écart = {abs(mc_ec  - mc_ep  - fwd_pv):.4f}")

    # ── Prime d'exercice anticipé ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  PRIME D'EXERCICE ANTICIPÉ   (Américain - Européen)")
    print(f"  Call : Tri={tri_ac - tri_ec:+.4f}   MC={mc_ac - mc_ec:+.4f}")
    print(f"  Put  : Tri={tri_ap - tri_ep:+.4f}   MC={mc_ap - mc_ep:+.4f}")
    print(SEP)


if __name__ == "__main__":
    main()