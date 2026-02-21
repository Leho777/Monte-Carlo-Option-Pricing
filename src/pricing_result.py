"""
PricingResult — Dataclass unifié pour tous les résultats de pricing.

Chaque méthode de MonteCarloModel (et BlackScholes) retourne un PricingResult
au lieu d'un dict ou d'un simple float, ce qui permet :
  - un affichage standardisé (str / repr)
  - le calcul d'intervalles de confiance
  - la comparaison facile avec une référence analytique
  - l'agrégation dans ConvergenceStudy
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PricingResult:
    """
    Résultat d'un calcul de prix.

    Attributs
    ---------
    price : float
        Prix de l'option estimé.
    std_error : float
        Erreur standard MC (0 pour les méthodes analytiques).
    num_paths : int
        Nombre de trajectoires utilisées (0 pour analytique).
    elapsed_s : float
        Temps de calcul en secondes.
    method : str
        Nom de la méthode (ex. 'MC-European', 'LS-American', 'Black-Scholes').
    num_steps : int
        Nombre de pas de temps (0 pour les méthodes sans discrétisation).
    extra : dict
        Métadonnées libres (poly_basis, antithetic, …).
    """

    price: float
    std_error: float = 0.0
    num_paths: int = 0
    elapsed_s: float = 0.0
    method: str = "unknown"
    num_steps: int = 0
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Intervalles de confiance
    # ------------------------------------------------------------------

    def confidence_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """
        Intervalle de confiance bilatéral au niveau (1-alpha).

        Parameters
        ----------
        alpha : float, default 0.05
            Niveau de signification (0.05 → IC à 95 %).

        Returns
        -------
        (lower, upper) : tuple[float, float]
        """
        if self.std_error <= 0 or self.num_paths <= 1:
            return (self.price, self.price)
        # Quantile normal (approximation Student → Normal pour grands n)
        from scipy.stats import norm as _norm
        z = _norm.ppf(1 - alpha / 2)
        margin = z * self.std_error
        return (self.price - margin, self.price + margin)

    # ------------------------------------------------------------------
    # Comparaison avec référence
    # ------------------------------------------------------------------

    def relative_error(self, reference: float) -> float:
        """
        Erreur relative par rapport à un prix de référence.

            rel_error = (price - reference) / reference

        Returns
        -------
        float : erreur relative (positive = surestimation)
        """
        if abs(reference) < 1e-12:
            return float('nan')
        return (self.price - reference) / reference

    def in_confidence_interval(self, reference: float, alpha: float = 0.05) -> bool:
        """Retourne True si `reference` est dans l'IC au niveau 1-alpha."""
        lo, hi = self.confidence_interval(alpha)
        return lo <= reference <= hi

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        ic = self.confidence_interval()
        parts = [
            f"[{self.method}]",
            f"price={self.price:.4f}",
        ]
        if self.std_error > 0:
            parts.append(f"se={self.std_error:.4f}")
            parts.append(f"IC95=[{ic[0]:.4f}, {ic[1]:.4f}]")
        if self.num_paths > 0:
            parts.append(f"N={self.num_paths:,}")
        if self.elapsed_s > 0:
            parts.append(f"t={self.elapsed_s:.2f}s")
        return "  ".join(parts)

    def __repr__(self) -> str:
        return (f"PricingResult(price={self.price:.4f}, se={self.std_error:.4f}, "
                f"N={self.num_paths}, method='{self.method}')")


# ------------------------------------------------------------------
# Utilitaire : timer contextuel
# ------------------------------------------------------------------

class _Timer:
    """Context manager pour mesurer un temps d'exécution."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0

    @property
    def seconds(self) -> float:
        return getattr(self, 'elapsed', 0.0)


def timed() -> _Timer:
    """
    Retourne un timer contextuel.

    Usage ::

        with timed() as t:
            price = model.price_european_mc(...)
        print(f"Done in {t.seconds:.2f}s")
    """
    return _Timer()
