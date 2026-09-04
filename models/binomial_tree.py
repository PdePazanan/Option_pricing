"""
Arbre binomial CRR (Cox-Ross-Rubinstein) pour options européennes et américaines.

Contrairement à Black-Scholes (temps continu, européen only), l'arbre
discrétise le temps en N pas, ce qui permet de gérer l'exercice anticipé
des options américaines.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.black_scholes import price as bs_price


def crr_params(T, r, sigma, N):
    """
    Calcule u, d, p pour l'arbre CRR.

    u et d sont choisis pour que d = 1/u (arbre recombinant, sinon on
    aurait 2^N nœuds au lieu de N+1) et pour matcher la variance du GBM.
    p est la proba risque-neutre de monter.

    Condition de non-arbitrage : d < e^(r*dt) < u, sinon p sort de [0,1].
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    if not (0 < p < 1):
        raise ValueError(f"Condition de non-arbitrage violée : p={p:.4f}. Réduire N ou sigma.")

    return {'dt': dt, 'u': u, 'd': d, 'p': p}


def build_price_tree(S, params, N):
    """
    Matrice triangulaire des prix du sous-jacent à chaque nœud.
    price_tree[i, j] = prix au pas i après j descentes.

    Grâce à u*d=1 l'arbre est recombinant donc on peut calculer directement
    price_tree[i,j] = S * u^(i-j) * d^j sans repasser par tous les chemins.
    """
    u, d = params['u'], params['d']
    tree = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(i + 1):
            tree[i, j] = S * (u ** (i - j)) * (d ** j)

    return tree


def price(S, K, T, r, sigma, N=200, option_type='call', exercise='european'):
    """
    Prix d'une option par arbre binomial CRR.

    1. On calcule les prix terminaux (à maturité) et les payoffs associés.
    2. On remonte l'arbre pas par pas (backward induction) en actualisant
       l'espérance risque-neutre : V = e^(-r*dt) * (p*V_up + (1-p)*V_down)
    3. Pour une option américaine, à chaque nœud on compare avec la valeur
       d'exercice immédiat et on garde le max (c'est la seule différence
       avec le cas européen).
    """
    params = crr_params(T, r, sigma, N)
    dt, u, d, p = params['dt'], params['u'], params['d'], params['p']
    disc = np.exp(-r * dt)

    # prix terminaux à maturité
    j = np.arange(N + 1)
    ST = S * (u ** (N - j)) * (d ** j)

    if option_type == 'call':
        V = np.maximum(ST - K, 0)
    elif option_type == 'put':
        V = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    # backward induction : de la maturité vers t=0
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:i+1] + (1 - p) * V[1:i+2])

        if exercise == 'american':
            ST_i = S * (u ** (i - j[:i+1])) * (d ** j[:i+1])
            if option_type == 'call':
                intrinsic = np.maximum(ST_i - K, 0)
            else:
                intrinsic = np.maximum(K - ST_i, 0)
            V = np.maximum(V, intrinsic)

    return float(V[0])


def early_exercise_premium(S, K, T, r, sigma, N=200, option_type='put'):
    """
    Prime d'exercice anticipé = prix américain - prix européen (toujours >= 0).

    En pratique : quasi nulle pour un call sans dividende (jamais intéressant
    d'exercer tôt), mais significative pour un put bien dans la monnaie.
    """
    p_euro = price(S, K, T, r, sigma, N, option_type, 'european')
    p_amer = price(S, K, T, r, sigma, N, option_type, 'american')
    return {
        'european': p_euro,
        'american': p_amer,
        'premium': p_amer - p_euro,
    }


def convergence_vs_bs(S, K, T, r, sigma, option_type='call'):
    """Vérifie que l'arbre converge vers Black-Scholes quand N augmente."""
    bs = bs_price(S, K, T, r, sigma, option_type)
    steps = [5, 10, 25, 50, 100, 200, 500]
    results = []
    for N in steps:
        p_bin = price(S, K, T, r, sigma, N, option_type, 'european')
        results.append({'N': N, 'price': p_bin, 'error': abs(p_bin - bs)})
    return {'bs_price': bs, 'convergence': results}


if __name__ == '__main__':
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    print(f"S={S}, K={K}, T={T}y, r={r:.0%}, sigma={sigma:.0%}, N=200")

    params = crr_params(T, r, sigma, 200)
    print(f"u={params['u']:.4f}  d={params['d']:.4f}  p={params['p']:.4f}")

    for otype in ['call', 'put']:
        p_bin = price(S, K, T, r, sigma, 200, otype, 'european')
        p_bs = bs_price(S, K, T, r, sigma, otype)
        print(f"{otype}: binomial={p_bin:.4f}  bs={p_bs:.4f}  ecart={abs(p_bin-p_bs):.2e}")

    result = early_exercise_premium(S, K, T, r, sigma, option_type='put')
    print(f"\nput - premium exercice anticipe: {result['premium']:.4f} (euro={result['european']:.4f}, amer={result['american']:.4f})")

    result_call = early_exercise_premium(S, K, T, r, sigma, option_type='call')
    print(f"call - premium: {result_call['premium']:.6f} (devrait etre ~0, jamais interet d'exercer tot sans dividende)")

    conv = convergence_vs_bs(S, K, T, r, sigma, 'call')
    for row in conv['convergence']:
        print(f"N={row['N']:>4}  price={row['price']:.6f}  err={row['error']:.2e}")