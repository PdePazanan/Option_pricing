"""
Binomial Tree Model (Cox-Ross-Rubinstein) — Pricing d'options européennes et américaines
==========================================================================================
Implémente :
  - Arbre CRR avec N pas de temps
  - Options européennes (call et put)
  - Options américaines (call et put) — exercice anticipé possible
  - Comparaison avec Black-Scholes pour validation
  - Convergence de l'arbre vers BS quand N → ∞

Différence clé avec Black-Scholes :
  - BS suppose le temps continu → formule fermée, options européennes uniquement
  - Binomial discrétise le temps → flexible, gère l'exercice anticipé américain
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.black_scholes import price as bs_price


# ─────────────────────────────────────────────
# PARAMÈTRES CRR
# ─────────────────────────────────────────────

def crr_params(T, r, sigma, N):
    """
    Calcule les paramètres Cox-Ross-Rubinstein de l'arbre.

    Le calibrage CRR choisit u, d, p pour que :
      1. d = 1/u  → l'arbre est "recombinable" (up+down = down+up)
                    sans ça, le nombre de nœuds exploseraient en 2^N
      2. u = e^{σ√Δt} → la variance de l'arbre matche celle du GBM
      3. p risque-neutre → l'espérance de rendement = taux sans risque r

    Condition de non-arbitrage (nécessaire) : d < e^{rΔt} < u
    Si elle n'est pas vérifiée, l'arbre est mal calibré.

    Paramètres
    ----------
    T     : float — Maturité en années
    r     : float — Taux sans risque
    sigma : float — Volatilité annualisée
    N     : int   — Nombre de pas de temps

    Retourne
    --------
    dict : dt, u, d, p (probabilité risque-neutre de monter)
    """
    dt = T / N                          # Durée d'un pas
    u  = np.exp(sigma * np.sqrt(dt))    # Facteur de montée
    d  = 1.0 / u                        # Facteur de descente (= 1/u)
    p  = (np.exp(r * dt) - d) / (u - d) # Probabilité risque-neutre

    # Vérification de la condition de non-arbitrage
    if not (0 < p < 1):
        raise ValueError(
            f"Condition de non-arbitrage violée : p={p:.4f}. "
            f"Réduire N ou sigma."
        )

    return {'dt': dt, 'u': u, 'd': d, 'p': p}


# ─────────────────────────────────────────────
# CONSTRUCTION DE L'ARBRE DES PRIX
# ─────────────────────────────────────────────

def build_price_tree(S, params, N):
    """
    Construit la matrice triangulaire des prix du sous-jacent (forward pass).

    Structure de la matrice price_tree[i, j] :
      - i = numéro du pas de temps (colonne, 0 à N)
      - j = nombre de descentes depuis le nœud initial (0 = tout monté)

    Exemple pour N=3 :
      price_tree[0,0] = S          (t=0)
      price_tree[1,0] = S*u        (t=1, monté)
      price_tree[1,1] = S*d        (t=1, descendu)
      price_tree[3,0] = S*u^3      (t=3, monté 3 fois)
      price_tree[3,3] = S*d^3      (t=3, descendu 3 fois)

    Grâce à u*d = 1, les chemins qui convergent au même nœud donnent
    le même prix : S*u*d = S*d*u = S. L'arbre est "recombinable".

    On utilise une formule vectorisée :
      price_tree[i, j] = S * u^(i-j) * d^j  pour j = 0..i

    Ce qui s'écrit S * u^i * (d/u)^j = S * u^(i-2j) puisque d=1/u.

    Paramètres
    ----------
    S      : float — Prix initial
    params : dict  — Sortie de crr_params()
    N      : int   — Nombre de pas

    Retourne
    --------
    np.ndarray (N+1, N+1) — Matrice triangulaire inférieure des prix
    """
    u, d = params['u'], params['d']
    tree = np.zeros((N + 1, N + 1))

    for i in range(N + 1):           # i = pas de temps
        for j in range(i + 1):       # j = nombre de descentes
            tree[i, j] = S * (u ** (i - j)) * (d ** j)

    return tree


# ─────────────────────────────────────────────
# PRICING PRINCIPAL
# ─────────────────────────────────────────────

def price(S, K, T, r, sigma, N=200, option_type='call', exercise='european'):
    """
    Prix d'une option par l'arbre binomial CRR.

    L'algorithme en deux phases :

    PHASE 1 — Forward pass (build_price_tree) :
      Calculer tous les prix possibles du sous-jacent jusqu'à la maturité.
      On obtient une matrice triangulaire de N+1 colonnes.

    PHASE 2 — Backward induction :
      Partir des payoffs terminaux (colonne N) et remonter.
      À chaque nœud (i, j), la valeur de l'option est :

        V[i,j] = e^{-rΔt} * (p * V[i+1,j] + (1-p) * V[i+1,j+1])

      Pour une option AMÉRICAINE, on compare aussi avec l'exercice immédiat :

        V[i,j] = max(V[i,j], payoff_immediat[i,j])

      Si exercer maintenant rapporte plus → on exerce.
      Cette ligne est la seule différence entre européen et américain.

    Paramètres
    ----------
    S           : float — Prix actuel
    K           : float — Strike
    T           : float — Maturité en années
    r           : float — Taux sans risque
    sigma       : float — Volatilité
    N           : int   — Nombre de pas (200 par défaut = bonne précision)
    option_type : str   — 'call' ou 'put'
    exercise    : str   — 'european' ou 'american'

    Retourne
    --------
    float : prix de l'option
    """
    params = crr_params(T, r, sigma, N)
    dt, u, d, p = params['dt'], params['u'], params['d'], params['p']
    disc = np.exp(-r * dt)   # Facteur d'actualisation pour un pas

    # ── PHASE 1 : Prix terminaux (dernière colonne de l'arbre)
    # Prix à maturité : S * u^(N-j) * d^j pour j = 0..N
    j = np.arange(N + 1)
    ST = S * (u ** (N - j)) * (d ** j)   # Vecteur de N+1 prix terminaux

    # ── Payoffs terminaux
    if option_type == 'call':
        V = np.maximum(ST - K, 0)
    elif option_type == 'put':
        V = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    # ── PHASE 2 : Backward induction (remonter de N vers 0)
    for i in range(N - 1, -1, -1):
        # Prix du sous-jacent à ce pas (pour l'exercice américain)
        ST_i = S * (u ** (i - j[:i+1])) * (d ** j[:i+1])

        # Valeur de continuation (espérance actualisée risque-neutre)
        V = disc * (p * V[:i+1] + (1 - p) * V[1:i+2])

        # Pour une option américaine : comparer avec l'exercice immédiat
        if exercise == 'american':
            if option_type == 'call':
                intrinsic = np.maximum(ST_i - K, 0)
            else:
                intrinsic = np.maximum(K - ST_i, 0)
            V = np.maximum(V, intrinsic)

    return float(V[0])


# ─────────────────────────────────────────────
# PRIME D'EXERCICE ANTICIPÉ
# ─────────────────────────────────────────────

def early_exercise_premium(S, K, T, r, sigma, N=200, option_type='put'):
    """
    Calcule la prime d'exercice anticipé = Prix américain - Prix européen.

    Cette prime est TOUJOURS positive ou nulle :
    une option américaine vaut au moins autant qu'une européenne
    (droits supplémentaires sans coûts).

    En pratique :
    - Call américain sans dividende : prime ≈ 0 (jamais optimal d'exercer)
    - Put américain : prime > 0 (optimal d'exercer tôt si très dans la monnaie)
    - Call avec dividendes : prime > 0 (exercer juste avant le dividende)

    Retourne
    --------
    dict avec prix européen, américain, et prime
    """
    p_euro = price(S, K, T, r, sigma, N, option_type, 'european')
    p_amer = price(S, K, T, r, sigma, N, option_type, 'american')
    return {
        'european': p_euro,
        'american': p_amer,
        'premium':  p_amer - p_euro,
    }


# ─────────────────────────────────────────────
# CONVERGENCE VERS BLACK-SCHOLES
# ─────────────────────────────────────────────

def convergence_vs_bs(S, K, T, r, sigma, option_type='call'):
    """
    Montre la convergence de l'arbre binomial vers Black-Scholes.

    Quand N → ∞, le binomial converge vers BS car la loi binomiale
    converge vers la loi normale (théorème central limite).

    On calcule le prix BS une fois, puis le prix binomial pour
    N = 5, 10, 25, 50, 100, 200, 500 et on mesure l'erreur absolue.

    Retourne
    --------
    dict avec prix BS et liste de (N, prix_binomial, erreur)
    """
    bs = bs_price(S, K, T, r, sigma, option_type)
    steps = [5, 10, 25, 50, 100, 200, 500]
    results = []
    for N in steps:
        p_bin = price(S, K, T, r, sigma, N, option_type, 'european')
        results.append({
            'N':     N,
            'price': p_bin,
            'error': abs(p_bin - bs),
        })
    return {'bs_price': bs, 'convergence': results}


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    print("=" * 60)
    print("ARBRE BINOMIAL CRR — DÉMONSTRATION")
    print("=" * 60)
    print(f"S={S}, K={K}, T={T}y, r={r:.0%}, σ={sigma:.0%}, N=200")

    # ── 1. Paramètres CRR
    params = crr_params(T, r, sigma, 200)
    print(f"\nParamètres CRR (N=200) :")
    print(f"  Δt = {params['dt']:.4f} ans")
    print(f"  u  = {params['u']:.6f}  (facteur montée)")
    print(f"  d  = {params['d']:.6f}  (facteur descente = 1/u)")
    print(f"  p  = {params['p']:.6f}  (probabilité risque-neutre)")

    # ── 2. Comparaison européen vs Black-Scholes
    print(f"\nComparaison Binomial vs Black-Scholes (européen) :")
    for otype in ['call', 'put']:
        p_bin = price(S, K, T, r, sigma, 200, otype, 'european')
        p_bs  = bs_price(S, K, T, r, sigma, otype)
        print(f"  {otype.capitalize():4s} : Binomial = {p_bin:.4f}  |  BS = {p_bs:.4f}  |  Erreur = {abs(p_bin-p_bs):.2e}")

    # ── 3. Prime d'exercice anticipé (put américain)
    print(f"\nPrime d'exercice anticipé (put) :")
    result = early_exercise_premium(S, K, T, r, sigma, option_type='put')
    print(f"  Put européen  : {result['european']:.4f} $")
    print(f"  Put américain : {result['american']:.4f} $")
    print(f"  Prime         : {result['premium']:.4f} $ (+{result['premium']/result['european']*100:.2f}%)")

    # ── 4. Call américain (prime devrait être ≈ 0)
    result_call = early_exercise_premium(S, K, T, r, sigma, option_type='call')
    print(f"\nPrime d'exercice anticipé (call, sans dividende) :")
    print(f"  Call européen  : {result_call['european']:.4f} $")
    print(f"  Call américain : {result_call['american']:.4f} $")
    print(f"  Prime          : {result_call['premium']:.6f} $ (≈ 0, jamais optimal d'exercer)")

    # ── 5. Convergence vers BS
    print(f"\nConvergence vers Black-Scholes (call européen) :")
    conv = convergence_vs_bs(S, K, T, r, sigma, 'call')
    print(f"  Prix BS de référence : {conv['bs_price']:.6f}")
    print(f"  {'N':>5}  {'Prix binomial':>14}  {'Erreur':>10}")
    print(f"  {'-'*35}")
    for row in conv['convergence']:
        print(f"  {row['N']:>5}  {row['price']:>14.6f}  {row['error']:>10.2e}")