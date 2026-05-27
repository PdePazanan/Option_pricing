"""
Monte Carlo Simulation — Pricing of options by stochastic simulation
=======================================================================
Implement :
  - Simulation of trajectories by geometric Brownian motion (GBM)
  - Pricing call/put European by averaging the discounted payoffs
  - Variance reduction by antithetic variables
  - Confidence interval on the estimated price
  - Comparison with Black-Scholes for validation

Why Monte Carlo ?
  - Black-Scholes : closed-form formula, fast, European only
  - Binomial      : discrete, handles American options, N² in memory
  - Monte Carlo   : flexible, extensible to any exotic product,
                    the only method to handle path-dependent options
                    (barrier, Asian, lookback...)

Complexity :
  - Time   : O(N * steps)  with N = number of simulations
  - Memory : O(N)      we stock only the terminal prices
"""

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.black_scholes import price as bs_price


# ─────────────────────────────────────────────
# SIMULATION of TERMINAL PRICES
# ─────────────────────────────────────────────

def simulate_terminal_prices(S, T, r, sigma, n_sim, antithetic=False, seed=None):
    """
    Simulate the terminal prices S_T under the risk-neutral measure.

    Under Black-Scholes, S_T follows a log-normal distribution :

        S_T = S_0 * exp[(r - σ²/2)*T  +  σ*√T*Z],   Z ~ N(0,1)

    The term (r - σ²/2) is the risk-neutral drift corrected by Itô.
    Without the correction -σ²/2, we would overestimate the average price because :
        E[e^{σW_T}] = e^{σ²T/2}  (Jensen's inequality for the convexity of the exponential)

    ANTITHETIC MODE (antithetic=True) :
    ────────────────────────────────────
    For each draw Z, we also simulate -Z.
    S_T⁺ = S * exp[(r - σ²/2)*T + σ*√T*Z]
    S_T⁻ = S * exp[(r - σ²/2)*T - σ*√T*Z]

    Key property : Z and -Z are both standard normal random variables,
    so both estimators are valid. But they are negatively correlated
    (when one is large, the other is small), which reduces the variance of
    their average :

        Var[(X + Y)/2] = (Var[X] + Var[Y] + 2*Cov[X,Y]) / 4

    As Cov[X,Y] < 0 (negative correlation), the total variance decreases.
    In practice : same precision with ~40-60% fewer simulations.

    Parameters
    ----------
    S          : float — Current price
    T          : float — Maturity in years
    r          : float — Risk-free rate
    sigma      : float — Volatility
    n_sim      : int   — Number of simulations (if antithetic : n_sim/2 draws)
    antithetic : bool  — Activate antithetic variance reduction
    seed       : int   — Random seed for reproducibility (None = random)

    Returns
    -------
    np.ndarray of length n_sim containing the terminal prices S_T
    """
    rng = np.random.default_rng(seed)

    drift = (r - 0.5 * sigma**2) * T          # Risk-neutral drift (Itô correction)
    diffusion_scale = sigma * np.sqrt(T)       # Standard deviation of the random term

    if antithetic:
        # Draw n_sim/2 standard normals, then use Z and -Z
        half = n_sim // 2
        Z = rng.standard_normal(half)
        Z_full = np.concatenate([Z, -Z])       # 2*half = n_sim valeurs
    else:
        Z_full = rng.standard_normal(n_sim)

    return S * np.exp(drift + diffusion_scale * Z_full)


# ─────────────────────────────────────────────
# PRICING PRINCIPAL
# ─────────────────────────────────────────────

def price(S, K, T, r, sigma, n_sim=100_000, option_type='call',
          antithetic=True, seed=None):
    """
    Price a European option by Monte Carlo.

    Algorithm :
      1. Simulate n_sim terminal prices S_T under the risk-neutral measure
      2. Calculate the payoff for each scenario
      3. Discount the mean of the payoffs : price = e^{-rT} * mean(payoffs)

    The expectation under the risk-neutral measure Q gives the arbitrage-free price of the option :
        C = e^{-rT} * E^Q[max(S_T - K, 0)]

    Parameters
    ----------
    S           : float — Current price
    K           : float — Strike
    T           : float — Maturity in years
    r           : float — Risk-free rate
    sigma       : float — Volatility
    n_sim       : int   — Number of simulations (100 000 by default)
    option_type : str   — 'call' or 'put'
    antithetic  : bool  — Variance reduction (True by default)
    seed        : int   — Random seed (None = random)

    Returns
    -------
    float : estimated price of the option
    """
    ST = simulate_terminal_prices(S, T, r, sigma, n_sim, antithetic, seed)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(np.exp(-r * T) * np.mean(payoffs))


# ─────────────────────────────────────────────
# PRIX AVEC INTERVALLE DE CONFIANCE
# ─────────────────────────────────────────────

def price_with_ci(S, K, T, r, sigma, n_sim=100_000, option_type='call',
                  antithetic=True, confidence=0.95, seed=None):
    """
    Price a European option with a confidence interval.

    By the central limit theorem, the mean of the payoffs follows
    asymptotically a normal distribution :

        mean(payoffs) ~ N(μ, σ²/n)

    The standard error is : SE = σ_payoffs / √n

    The 95% confidence interval is : [price - 1.96*SE, price + 1.96*SE]

    Interpretation : if we repeated the simulation 100 times, the confidence interval would contain
    the true price in approximately 95 cases.

    The error decreases as 1/√n → to halve the error, we need 4× more
    simulations. This is the "curse of Monte Carlo".

    Parameters
    ----------
    confidence : float — Confidence level (0.95 = 95%)

    Returns
    -------
    dict with 'price', 'std_error', 'ci_lower', 'ci_upper', 'ci_width'
    """
    from scipy import stats

    ST = simulate_terminal_prices(S, T, r, sigma, n_sim, antithetic, seed)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    disc = np.exp(-r * T)
    mc_price   = disc * np.mean(payoffs)
    std_error  = disc * np.std(payoffs, ddof=1) / np.sqrt(n_sim)

    # Quantile z pour le niveau de confiance choisi
    z = stats.norm.ppf((1 + confidence) / 2)

    return {
        'price':     float(mc_price),
        'std_error': float(std_error),
        'ci_lower':  float(mc_price - z * std_error),
        'ci_upper':  float(mc_price + z * std_error),
        'ci_width':  float(2 * z * std_error),
        'n_sim':     n_sim,
    }


# ─────────────────────────────────────────────
# COMPARAISON STANDARD VS ANTITHÉTIQUE
# ─────────────────────────────────────────────

def antithetic_comparison(S, K, T, r, sigma, option_type='call', n_runs=20):
    """
    Compare empiriquement l'estimateur standard vs antithétique.

    On répète n_runs simulations avec chaque méthode et on mesure :
      - L'erreur moyenne par rapport à Black-Scholes
      - L'écart-type des estimations (= variance de l'estimateur)
      - Le ratio de réduction de variance

    Returns
    -------
    dict with comparative results of the two methods
    """
    n_sim = 10_000
    bs = bs_price(S, K, T, r, sigma, option_type)

    prices_std  = [price(S, K, T, r, sigma, n_sim, option_type, antithetic=False) for _ in range(n_runs)]
    prices_anti = [price(S, K, T, r, sigma, n_sim, option_type, antithetic=True)  for _ in range(n_runs)]

    std_std  = np.std(prices_std)
    std_anti = np.std(prices_anti)

    return {
        'bs_price':          bs,
        'std_mean':          float(np.mean(prices_std)),
        'std_volatility':    float(std_std),
        'anti_mean':         float(np.mean(prices_anti)),
        'anti_volatility':   float(std_anti),
        'variance_reduction': float((std_std / std_anti) ** 2),   # ratio des variances
    }


# ─────────────────────────────────────────────
# CONVERGENCE VERS BLACK-SCHOLES
# ─────────────────────────────────────────────

def convergence_vs_bs(S, K, T, r, sigma, option_type='call', seed=42):
    """
    Montre la convergence de Monte Carlo vers Black-Scholes.

    L'erreur théorique décroît en σ_payoff / √n.
    On vérifie cette loi empiriquement en calculant le prix pour
    différentes valeurs de n_sim.

    Returns
    -------
    dict with BS price and list of (n_sim, mc_price, error, ci_width)
    """
    bs = bs_price(S, K, T, r, sigma, option_type)
    steps = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    results = []
    for n in steps:
        result = price_with_ci(S, K, T, r, sigma, n, option_type, seed=seed)
        results.append({
            'n_sim':    n,
            'price':    result['price'],
            'error':    abs(result['price'] - bs),
            'ic_width': result['ci_width'],
        })
    return {'bs_price': bs, 'convergence': results}


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.20

    print("=" * 65)
    print("MONTE CARLO — DÉMONSTRATION")
    print("=" * 65)
    print(f"S={S}, K={K}, T={T}y, r={r:.0%}, σ={sigma:.0%}")

    # ── 1. Prix avec intervalle de confiance
    print(f"\n1. Price with confidence interval (100 000 simulations) :")
    for otype in ['call', 'put']:
        res = price_with_ci(S, K, T, r, sigma, n_sim=100_000,
                            option_type=otype, seed=42)
        bs  = bs_price(S, K, T, r, sigma, otype)
        print(f"\n   {otype.capitalize()} :")
        print(f"   Prix MC    : {res['price']:.4f} $  (IC 95%: [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}])")
        print(f"   Prix BS    : {bs:.4f} $")
        print(f"   Erreur     : {abs(res['price']-bs):.4f} $")
        print(f"   Std error  : {res['std_error']:.4f} $  (largeur IC = {res['ci_width']:.4f} $)")

    # ── 2. Réduction de variance antithétique
    print(f"\n2. Reduction of variance — standard vs antithetic (20 runs × 10 000 sim) :")
    comp = antithetic_comparison(S, K, T, r, sigma, 'call', n_runs=20)
    print(f"   RReference BS              : {comp['bs_price']:.4f} $")
    print(f"   Standard  — mean          : {comp['std_mean']:.4f} $  "
          f"| estimated volatility : {comp['std_volatility']:.5f}")
    print(f"   Antithetic — mean         : {comp['anti_mean']:.4f} $  "
          f"| estimated volatility : {comp['anti_volatility']:.5f}")
    print(f"   Ratio reduction variance  : {comp['variance_reduction']:.1f}×  "
          f"(antithetic is {comp['variance_reduction']:.1f}× more precise)")

    # ── 3. Convergence
    print(f"\n3. Convergence towards Black-Scholes (call, seed=42) :")
    conv = convergence_vs_bs(S, K, T, r, sigma, 'call')
    print(f"   Reference BS : {conv['bs_price']:.6f} $")
    print(f"   {'N sims':>10}  {'Prix MC':>10}  {'Error':>10}  {'CI Width':>12}")
    print(f"   {'-'*48}")
    for row in conv['convergence']:
        n_str = f"{row['n_sim']:,}".replace(',', ' ')
        print(f"   {n_str:>10}  {row['price']:>10.6f}  {row['error']:>10.4f}  {row['ic_width']:>12.4f}")