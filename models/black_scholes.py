"""
Black-Scholes model for European option pricing.

Assumes GBM price process, constant vol, no dividends, constant rate.
Gives closed-form price + greeks, plus implied vol via numerical inversion.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1(S, K, T, r, sigma):
    # standard BS d1 term
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def price(S, K, T, r, sigma, option_type='call'):
    """
    Call: C = S*N(d1) - K*e^(-rT)*N(d2)
    Put:  P = K*e^(-rT)*N(-d2) - S*N(-d1)
    """
    if T <= 0:
        # at expiry, just the payoff
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


def greeks(S, K, T, r, sigma, option_type='call'):
    """
    Delta: sensitivity to S
    Gamma: sensitivity of delta to S (same for call/put)
    Vega:  sensitivity to vol, scaled for a 1% move (same for call/put)
    Theta: time decay, per day
    Rho:   sensitivity to rate, scaled for a 1% move
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1 / 100

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * disc * norm.cdf(d2)) / 365
        rho = K * T * disc * norm.cdf(d2) / 100

    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * disc * norm.cdf(-d2)) / 365
        rho = -K * T * disc * norm.cdf(-d2) / 100

    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho,
    }


def implied_vol(market_price, S, K, T, r, option_type='call'):
    """
    Finds sigma such that price(sigma) == market_price, using Brent's method
    (no closed-form inverse for BS).
    """
    # market price has to beat intrinsic value, otherwise no valid vol exists
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == 'call' else max(K * np.exp(-r * T) - S, 0)
    if market_price <= intrinsic:
        return None

    objective = lambda sigma: price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-6, maxiter=500)  # search vol between 0.0001% and 500%
    except ValueError:
        return None  # no root in that range


def put_call_parity_check(S, K, T, r, sigma):
    """
    C - P should equal S - K*e^(-rT). Sanity check for the implementation,
    error should be near machine precision.
    """
    C = price(S, K, T, r, sigma, 'call')
    P = price(S, K, T, r, sigma, 'put')
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return {
        'call': C,
        'put': P,
        'lhs': lhs,
        'rhs': rhs,
        'error': abs(lhs - rhs),
    }


if __name__ == '__main__':
    S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.20

    print(f"S={S}, K={K}, T={T}y, r={r:.0%}, sigma={sigma:.0%}")

    call = price(S, K, T, r, sigma, 'call')
    put = price(S, K, T, r, sigma, 'put')
    print(f"call={call:.4f}  put={put:.4f}")

    g = greeks(S, K, T, r, sigma, 'call')
    print("greeks (call):", {k: round(v, 4) for k, v in g.items()})

    parity = put_call_parity_check(S, K, T, r, sigma)
    print(f"put-call parity error: {parity['error']:.2e}")

    # recover sigma from the price we just computed, should match original
    sigma_impl = implied_vol(call, S, K, T, r, 'call')
    print(f"implied vol: {sigma_impl:.4%} (original was {sigma:.4%})")