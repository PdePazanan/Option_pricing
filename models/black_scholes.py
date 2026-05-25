##black_scholes

#1. black_scholes(S, K, T, r, sigma, type)  → prix de l'option
#2. greeks(S, K, T, r, sigma, type)         → dict avec Δ, Γ, ν, Θ
#3. implied_vol(market_price, S, K, T, r)   → σ implicite
#4. (bonus) plot_greeks()                   → graphe des Greeks en fonction de S
"""
Black-Scholes Model — Pricing of european options
====================================================
Implements:
- Call and put prices (analytical formula)
- The Greeks: Delta, Gamma, Vega, Theta, Rho
- Implied volatility (numerical inversion using the Brent method)
- Verification using Put-Call Parity

Black-Scholes model assumptions:
- Price follows a geometric Brownian motion
- Constant volatility over time
- No dividends
- Continuous and frictionless market
- Constant risk-free rate
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ─────────────────────────────────────────────
# FONCTIONS INTERNES
# ─────────────────────────────────────────────

def _d1(S, K, T, r, sigma):
    """
    calculate d1, the first argument of the normal distribution in the BS formula.

    Interpretation : d1 captures the "adjusted" probability that the option expires
    in the money, taking into account the expected growth of the underlying.

    Parameters
    ----------
    S     : float — Current price of the underlying asset
    K     : float — Strike price
    T     : float — Time to maturity (in years)
    r     : float — Annual risk-free rate (e.g., 0.05 for 5%)
    sigma : float — Annualized volatility (e.g., 0.20 for 20%)
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    """
    calculate d2 = d1 - sigma*sqrt(T).

    Interpretation : N(d2) is the risk-neutral probability that the option
    expires in the money (i.e., that ST > K for a call).
    """
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


# ─────────────────────────────────────────────
# PRIX DE L'OPTION
# ─────────────────────────────────────────────

def price(S, K, T, r, sigma, option_type='call'):
    """
    Prix d'une option européenne par la formule de Black-Scholes.

    Formule call : C = S*N(d1) - K*e^{-rT}*N(d2)
    Formule put  : P = K*e^{-rT}*N(-d2) - S*N(-d1)

    N(.) est la fonction de répartition de la loi normale standard.
    K*e^{-rT} est la valeur actualisée du strike.

    Paramètres
    ----------
    S           : float — Prix actuel du sous-jacent (ex: 100)
    K           : float — Strike (ex: 105)
    T           : float — Maturité en années (ex: 0.5 pour 6 mois)
    r           : float — Taux sans risque (ex: 0.05)
    sigma       : float — Volatilité (ex: 0.20)
    option_type : str   — 'call' ou 'put'

    Retourne
    --------
    float : prix de l'option
    """
    if T <= 0:
        # À maturité : on retourne le payoff directement
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")


# ─────────────────────────────────────────────
# LES GREEKS
# ─────────────────────────────────────────────

def greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calcule les 5 Greeks principaux d'une option européenne.

    Les Greeks mesurent la sensibilité du prix à chaque paramètre.
    Ils sont utilisés pour le hedging et la gestion du risque.

    Delta (Δ) : dC/dS
        Sensibilité au prix du sous-jacent.
        Un delta de 0.6 signifie que si S augmente de 1$,
        le prix de l'option augmente d'environ 0.60$.
        → Utilisé pour construire un portefeuille delta-neutre.

    Gamma (Γ) : d²C/dS²
        Sensibilité du Delta au prix du sous-jacent.
        Mesure la courbure : un gamma élevé = le delta change vite.
        → Élevé près du strike, à l'approche de la maturité.

    Vega (ν) : dC/d(sigma)
        Sensibilité à la volatilité.
        Un vega de 20 signifie que si sigma augmente de 1%,
        l'option gagne 0.20$ (vega exprimé pour 1% de var).
        → NB : Vega n'est pas une lettre grecque mais c'est l'usage.

    Theta (Θ) : dC/dT
        Sensibilité au temps (time decay).
        Generalement négatif : l'option perd de la valeur avec le temps.
        Exprimé en $/jour.

    Rho (ρ) : dC/dr
        Sensibilité au taux sans risque.
        Moins critique en pratique pour les options court terme.

    Retourne
    --------
    dict avec les clés : 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)         # densité normale en d1 (= N'(d1))
    disc = np.exp(-r * T)         # facteur d'actualisation

    # Gamma et Vega sont identiques pour call et put
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * pdf_d1 / 100    # divisé par 100 → pour 1% de vol

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * disc * norm.cdf(d2)) / 365
        rho   = K * T * disc * norm.cdf(d2) / 100

    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * disc * norm.cdf(-d2)) / 365
        rho   = -K * T * disc * norm.cdf(-d2) / 100

    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    return {
        'delta': delta,
        'gamma': gamma,
        'vega':  vega,
        'theta': theta,
        'rho':   rho,
    }


# ─────────────────────────────────────────────
# VOLATILITÉ IMPLICITE
# ─────────────────────────────────────────────

def implied_vol(market_price, S, K, T, r, option_type='call'):
    """
    Calcule la volatilité implicite par inversion numérique de Black-Scholes.

    Problème : on observe le prix de marché d'une option, mais on ne connaît
    pas la volatilité σ que le marché utilise implicitement.

    On cherche σ* tel que : BS(S, K, T, r, σ*) = market_price

    Il n'existe pas de formule analytique pour σ* → on utilise la méthode
    de Brent (scipy.optimize.brentq), qui cherche le zéro de :
        f(σ) = BS(σ) - market_price

    La méthode de Brent est hybride : combine bisection + interpolation.
    Elle converge toujours si f change de signe sur [a, b].

    Paramètres
    ----------
    market_price : float — Prix observé sur le marché
    S, K, T, r   : float — Paramètres habituels
    option_type  : str   — 'call' ou 'put'

    Retourne
    --------
    float : volatilité implicite (ex: 0.25 pour 25%)
    None  : si la convergence échoue (prix incohérent)
    """
    # Vérification : le prix de marché doit être supérieur à la valeur intrinsèque
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == 'call' else max(K * np.exp(-r * T) - S, 0)
    if market_price <= intrinsic:
        return None  # Prix arbitrable, pas de vol implicite

    # Fonction dont on cherche le zéro
    objective = lambda sigma: price(S, K, T, r, sigma, option_type) - market_price

    try:
        # Brent cherche le zéro sur [1e-6, 5.0] → vol entre 0.0001% et 500%
        return brentq(objective, 1e-6, 5.0, xtol=1e-6, maxiter=500)
    except ValueError:
        return None  # Pas de zéro trouvé dans l'intervalle


# ─────────────────────────────────────────────
# PUT-CALL PARITY — VÉRIFICATION
# ─────────────────────────────────────────────

def put_call_parity_check(S, K, T, r, sigma):
    """
    Vérifie la Put-Call Parity : C - P = S - K*e^{-rT}

    C'est une relation d'arbitrage exacte et fondamentale :
    si elle n'est pas vérifiée, il y a une opportunité d'arbitrage.

    On l'utilise ici pour valider notre implémentation.
    L'erreur doit être < 1e-10 (erreur numérique machine uniquement).

    Retourne
    --------
    dict avec 'lhs' (C-P), 'rhs' (S - Ke^{-rT}), 'error' (|lhs - rhs|)
    """
    C = price(S, K, T, r, sigma, 'call')
    P = price(S, K, T, r, sigma, 'put')
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return {
        'call':  C,
        'put':   P,
        'lhs':   lhs,
        'rhs':   rhs,
        'error': abs(lhs - rhs),
    }


# ─────────────────────────────────────────────
# DEMO RAPIDE
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Paramètres d'exemple
    S     = 100   # Prix actuel
    K     = 105   # Strike
    T     = 0.5   # 6 mois
    r     = 0.05  # 5% sans risque
    sigma = 0.20  # 20% de volatilité

    print("=" * 50)
    print("BLACK-SCHOLES — EXEMPLE")
    print("=" * 50)
    print(f"S={S}, K={K}, T={T}y, r={r:.0%}, σ={sigma:.0%}")
    print()

    call = price(S, K, T, r, sigma, 'call')
    put  = price(S, K, T, r, sigma, 'put')
    print(f"Prix Call : {call:.4f} $")
    print(f"Prix Put  : {put:.4f} $")
    print()

    g = greeks(S, K, T, r, sigma, 'call')
    print("Greeks (Call) :")
    for name, val in g.items():
        print(f"  {name.capitalize():6s} = {val:.6f}")
    print()

    parity = put_call_parity_check(S, K, T, r, sigma)
    print(f"Put-Call Parity : C - P = {parity['lhs']:.6f}")
    print(f"                  S - Ke^{{-rT}} = {parity['rhs']:.6f}")
    print(f"                  Erreur = {parity['error']:.2e}  ✓" if parity['error'] < 1e-10 else "  ✗ ERREUR")
    print()

    # Volatilité implicite
    # On prend le prix qu'on vient de calculer et on retrouve sigma
    sigma_impl = implied_vol(call, S, K, T, r, 'call')
    print(f"Vol. implicite retrouvée : {sigma_impl:.4%}")
    print(f"Vol. originale           : {sigma:.4%}")
    print(f"Erreur                   : {abs(sigma_impl - sigma):.2e}  ✓")