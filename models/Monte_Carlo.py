"""
Monte Carlo pricing pour options européennes.

Principe : on simule plein de trajectoires possibles du prix du sous-jacent,
on calcule le payoff pour chacune, on fait la moyenne actualisée.

Comparé à Black-Scholes (formule fermée) et au binomial (arbre), Monte Carlo
est plus lent mais beaucoup plus flexible : ça s'étend facilement à des
options exotiques (barrière, asiatique...) que les deux autres méthodes
ne peuvent pas gérer facilement.
"""

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.black_scholes import price as bs_price


def simulate_terminal_prices(S, T, r, sigma, n_sim, antithetic=False, seed=None):
    """
    Simule les prix terminaux S_T sous la mesure risque-neutre.

    Formule (GBM) : S_T = S * exp[(r - sigma^2/2)*T + sigma*sqrt(T)*Z], Z ~ N(0,1)
    Le -sigma^2/2 c'est la correction d'Itô, sans ça on biaise le prix moyen
    vers le haut (Jensen).

    Si antithetic=True : pour chaque Z tiré on utilise aussi -Z. Ça réduit la
    variance de l'estimateur (Z et -Z sont négativement corrélés) pour un
    coût de calcul quasi identique. En pratique on gagne en général 40-60%
    de simulations pour la même précision.
    """
    rng = np.random.default_rng(seed)

    drift = (r - 0.5 * sigma**2) * T
    diffusion_scale = sigma * np.sqrt(T)

    if antithetic:
        half = n_sim // 2
        Z = rng.standard_normal(half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = rng.standard_normal(n_sim)

    return S * np.exp(drift + diffusion_scale * Z_full)


def price(S, K, T, r, sigma, n_sim=100_000, option_type='call',
          antithetic=True, seed=None):
    """
    Prix d'une option européenne par Monte Carlo.

    On simule S_T, on calcule le payoff, on actualise la moyenne :
    C = e^(-rT) * E[max(S_T - K, 0)]
    """
    ST = simulate_terminal_prices(S, T, r, sigma, n_sim, antithetic, seed)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(np.exp(-r * T) * np.mean(payoffs))


def price_with_ci(S, K, T, r, sigma, n_sim=100_000, option_type='call',
                  antithetic=True, confidence=0.95, seed=None):
    """
    Même chose que price(), mais avec un intervalle de confiance sur
    l'estimation (utile pour savoir si notre nombre de simulations suffit).

    Par le théorème central limite, la moyenne des payoffs est ~normale,
    donc erreur standard = std(payoffs) / sqrt(n). On en déduit l'IC.

    """
    from scipy import stats

    ST = simulate_terminal_prices(S, T, r, sigma, n_sim, antithetic, seed)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    disc = np.exp(-r * T)
    mc_price = disc * np.mean(payoffs)
    std_error = disc * np.std(payoffs, ddof=1) / np.sqrt(n_sim)

    z = stats.norm.ppf((1 + confidence) / 2)  # ex: 1.96 pour 95%

    return {
        'price': float(mc_price),
        'std_error': float(std_error),
        'ci_lower': float(mc_price - z * std_error),
        'ci_upper': float(mc_price + z * std_error),
        'ci_width': float(2 * z * std_error),
        'n_sim': n_sim,
    }


def antithetic_comparison(S, K, T, r, sigma, option_type='call', n_runs=20):
    """
    Compare la variance de l'estimateur standard vs antithétique en relançant
    plusieurs fois la simulation (pour voir concrètement le gain).
    """
    n_sim = 10_000
    bs = bs_price(S, K, T, r, sigma, option_type)

    prices_std = [price(S, K, T, r, sigma, n_sim, option_type, antithetic=False) for _ in range(n_runs)]
    prices_anti = [price(S, K, T, r, sigma, n_sim, option_type, antithetic=True) for _ in range(n_runs)]

    std_std = np.std(prices_std)
    std_anti = np.std(prices_anti)

    return {
        'bs_price': bs,
        'std_mean': float(np.mean(prices_std)),
        'std_volatility': float(std_std),
        'anti_mean': float(np.mean(prices_anti)),
        'anti_volatility': float(std_anti),
        'variance_reduction': float((std_std / std_anti) ** 2),
    }


def convergence_vs_bs(S, K, T, r, sigma, option_type='call', seed=42):
    """Vérifie qu'on converge bien vers le prix Black-Scholes quand n_sim augmente."""
    bs = bs_price(S, K, T, r, sigma, option_type)
    steps = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    results = []
    for n in steps:
        result = price_with_ci(S, K, T, r, sigma, n, option_type, seed=seed)
        results.append({
            'n_sim': n,
            'price': result['price'],
            'error': abs(result['price'] - bs),
            'ic_width': result['ci_width'],
        })
    return {'bs_price': bs, 'convergence': results}


if __name__ == '__main__':
    S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.20

    print(f"S={S}, K={K}, T={T}y, r={r:.0%}, sigma={sigma:.0%}")

    for otype in ['call', 'put']:
        res = price_with_ci(S, K, T, r, sigma, n_sim=100_000, option_type=otype, seed=42)
        bs = bs_price(S, K, T, r, sigma, otype)
        print(f"{otype}: MC={res['price']:.4f} (IC95 [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]) vs BS={bs:.4f}")

    comp = antithetic_comparison(S, K, T, r, sigma, 'call', n_runs=20)
    print(f"\nreduction variance antithetic: {comp['variance_reduction']:.1f}x")

    conv = convergence_vs_bs(S, K, T, r, sigma, 'call')
    for row in conv['convergence']:
        print(f"n={row['n_sim']:>7}  price={row['price']:.4f}  err={row['error']:.4f}")