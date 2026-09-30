
#Monte Carlo pricing pour options européennes.



import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.black_scholes import price as bs_price


def simulate_terminal_prices(S, T, r, sigma, n_sim, antithetic=False, seed=None):
    
    #Simule les prix terminaux S_T sous la mesure risque-neutre.

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
    
   # Prix d'une option  par Monte Carlo.

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
    # compare variance et estimateur standard
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
