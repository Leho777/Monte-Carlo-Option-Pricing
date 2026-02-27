from enum import Enum
import numpy as np


class BasisType(str, Enum):
    """
    Bases polynomiales disponibles pour la régression Longstaff-Schwartz.

    Toutes engendrent le même espace polynomial (mathématiquement équivalentes),
    mais les bases orthogonales offrent une meilleure stabilité numérique
    que la base monomiale standard (POWER).
    """
    POWER     = 'power'     # Monômes standard : 1, x, x², x³, …
    LAGUERRE  = 'laguerre'  # exp(-x/2) * L_k(x)  (base de l'article L&S)
    HERMITE   = 'hermite'   # Polynômes d'Hermite probabilistes (Hermite)
    LEGENDRE  = 'legendre'  # Polynômes de Legendre
    CHEBYSHEV = 'chebyshev' # Polynômes de Chebyshev de type 1


class Regression:
    """
    Régression polynomiale pour Longstaff-Schwartz.

    Améliorations vs régression naïve (np.polyfit) :
    - Choix de la base polynomiale (BasisType) via matrice de design explicite
    - Normalisation automatique des inputs (essentielle pour LAGUERRE/HERMITE)
    - Résolution par np.linalg.lstsq (robuste aux cas singuliers)
    - Calcul de l'écart-type résiduel après fit
    - Seuil d'exercice : exercer seulement si IV > reg + threshold * std_résidu

    Normalisation :
    - LAGUERRE  : X_norm = X / mean(X)         â†’ valeurs autour de 1 (domaine â‰¥ 0)
    - Autres    : X_norm = (X - mean) / std     â†’ z-score, valeurs autour de 0

    Toutes les bases Ã©tant des polynÃ´mes, la solution des moindres carrÃ©s est
    unique et identique quelle que soit la base (cf. cours 1/7/2026 slide 4).
    La base affecte uniquement le conditionnement numÃ©rique du systÃ¨me.
    """

    def __init__(self, degree: int = 1,
                 basis: BasisType = BasisType.POWER,
                 residual_threshold: float = 0.0,
                 normalize: bool = True):
        """
        Parameters
        ----------
        degree             : degré du polynôme (2 = quadratique comme dans L&S,
                             3 = cubique par défaut)
        basis              : base polynomiale (voir BasisType)
        residual_threshold : fraction de l'écart-type résiduel ajoutée au seuil
                             0.0 → comportement LS standard
                             0.1 → exercer si IV > reg + 0.1 * std_résidu
        normalize          : si True (défaut), normalise les inputs avant de
                             construire la matrice de design
        """
        self.degree = degree
        self.basis = BasisType(basis)
        self.residual_threshold = residual_threshold
        self.normalize = normalize
        self._coeffs: np.ndarray = None
        self._residual_std: float = 0.0
        # Paramètres de normalisation, appris dans fit()
        self._x_loc: float = 0.0
        self._x_scale: float = 1.0

    # ------------------------------------------------------------------
    # Normalisation des inputs
    # ------------------------------------------------------------------

    def _fit_normalization(self, X: np.ndarray) -> None:
        """Calcule et stocke les paramètres de normalisation sur les données d'entraînement."""
        if not self.normalize:
            self._x_loc, self._x_scale = 0.0, 1.0
            return
        mean = float(np.mean(X))
        if self.basis == BasisType.LAGUERRE:
            # Domaine doit rester positif : X_norm = X / mean â†’ moyenne = 1
            self._x_loc = 0.0
            self._x_scale = mean if mean > 0 else 1.0
        else:
            # Z-score standard
            self._x_loc = mean
            std = float(np.std(X))
            self._x_scale = std if std > 0 else 1.0

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        """Applique la normalisation apprise lors du fit."""
        return (X - self._x_loc) / self._x_scale

    # ------------------------------------------------------------------
    # Matrice de design (base polynomiale)
    # ------------------------------------------------------------------

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construit la matrice Phi de shape (n, degree+1) dans la base choisie.
        Les inputs sont normalisés avant évaluation des polynômes.

        Phi[i, k] = k-ième fonction de base évaluée en X_norm[i].
        """
        X_n = self._normalize_x(X)
        d   = self.degree + 1
        eye = np.eye(d)

        if self.basis == BasisType.POWER:
            return np.column_stack([X_n ** k for k in range(d)])

        elif self.basis == BasisType.LAGUERRE:
            # Article L&S : Ï†_k(x) = exp(-x/2) * L_k(x)
            # Avec X_n â‰ˆ 1 pour ATM, exp(-0.5) â‰ˆ 0.6 â†’ pas de dÃ©crochage
            w = np.exp(-X_n / 2)
            return np.column_stack([
                w * np.polynomial.laguerre.lagval(X_n, eye[k]) for k in range(d)
            ])

        elif self.basis == BasisType.HERMITE:
            return np.column_stack([
                np.polynomial.hermite_e.hermeval(X_n, eye[k]) for k in range(d)
            ])

        elif self.basis == BasisType.LEGENDRE:
            return np.column_stack([
                np.polynomial.legendre.legval(X_n, eye[k]) for k in range(d)
            ])

        elif self.basis == BasisType.CHEBYSHEV:
            return np.column_stack([
                np.polynomial.chebyshev.chebval(X_n, eye[k]) for k in range(d)
            ])

        raise ValueError(f"Base polynomiale inconnue : {self.basis}")

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regression":
        """
        Régession moindres-carrés dans la base choisie.
        Apprend la normalisation sur X avant de résoudre le système.
        """
        self._fit_normalization(X)
        Phi = self._design_matrix(X)
        self._coeffs, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        self._residual_std = float(np.std(y - Phi @ self._coeffs))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit E[continuation | S_t] dans la base choisie, clampé à 0.
        Utilise la normalisation apprise lors du dernier fit().
        """
        if self._coeffs is None:
            raise ValueError("Appeler fit() avant predict().")
        return np.maximum(self._design_matrix(X) @ self._coeffs, 0.0)

    # ------------------------------------------------------------------
    # Décision d'exercice (Longstaff-Schwartz)
    # ------------------------------------------------------------------

    def exercise_decision(self,
                          S_at_step: np.ndarray,
                          intrinsic: np.ndarray,
                          continuation_discounted: np.ndarray) -> np.ndarray:
        """
        Décision d'exercice optimal à un pas de temps.

        Condition d'exercice avec seuil (cf. cours 1/7/2026 forward price example) :
            Exercer si IV(S) > E[continuation | S] + residual_threshold * std_résidu

        Parameters
        ----------
        S_at_step             : prix du sous-jacent à ce step, shape (num_paths,)
        intrinsic             : valeur intrinsèque, shape (num_paths,)
        continuation_discounted : cash flow futur discounté d'un step, shape (num_paths,)
        """
        itm_mask = intrinsic > 0
        n_itm = int(np.sum(itm_mask))

        if n_itm > self.degree + 1:
            self.fit(S_at_step[itm_mask], continuation_discounted[itm_mask])
            estimated = self.predict(S_at_step)
            margin = self._residual_std * self.residual_threshold
            return np.where(intrinsic > estimated + margin,
                            intrinsic,
                            continuation_discounted)
        else:
            return np.where(intrinsic > continuation_discounted,
                            intrinsic,
                            continuation_discounted)
