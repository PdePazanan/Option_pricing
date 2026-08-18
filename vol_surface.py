"""
Volatility Surface — Vol implicite depuis données de marché réelles (yfinance)
================================================================================
Implémente :
  - Récupération de la chaîne d'options d'un ticker via yfinance
  - Calcul de la vol implicite pour chaque (strike, maturité) avec NOTRE
    propre modèle Black-Scholes (models/black_scholes.py), pas celle
    fournie par yfinance
  - Smile de volatilité (vol implicite vs strike, à maturité fixée)
  - Surface de volatilité 3D (vol implicite vs strike vs maturité) avec Plotly

Pourquoi recalculer notre propre vol implicite plutôt qu'utiliser celle
de yfinance ?
  - yfinance renvoie souvent la vol implicite calculée par le fournisseur
    de données (méthode et conventions inconnues, parfois obsolète).
  - En la recalculant nous-mêmes via brentq sur notre price(), on est
    cohérent avec le reste du projet et on peut valider notre modèle
    black_scholes.implied_vol() sur des données réelles.

Structure attendue du projet :
    options-pricing/
    |-- models/
    |    |-- black_scholes.py   <- utilisé ici pour implied_vol()
    |-- visualisation/
    |    |-- vol_surface.py     <- ce fichier
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.black_scholes import implied_vol


# ─────────────────────────────────────────────
# RÉCUPÉRATION DES DONNÉES DE MARCHÉ
# ─────────────────────────────────────────────

def fetch_option_chain(ticker_symbol, option_type='call', r=0.05,
                        min_volume=5, max_expirations=8):
    """
    Récupère la chaîne d'options d'un ticker et calcule la vol implicite
    de chaque contrat avec notre propre modèle Black-Scholes.

    Étapes :
      1. yfinance.Ticker(ticker).options -> liste des dates de maturité
         disponibles (format 'YYYY-MM-DD')
      2. Pour chaque maturité, .option_chain(date) -> DataFrame calls/puts
      3. Pour chaque contrat, prix de marché = mid-price = (bid + ask) / 2
         (plus fiable que 'lastPrice' qui peut être une transaction ancienne)
      4. On filtre les contrats trop peu liquides (volume < min_volume ou
         bid/ask nuls) car leur prix ne reflète pas une vraie valeur de marché
      5. Inversion Black-Scholes -> vol implicite (models.black_scholes.implied_vol)

    Paramètres
    ----------
    ticker_symbol   : str   — Symbole boursier (ex: 'AAPL', 'SPY')
    option_type     : str   — 'call' ou 'put'
    r               : float — Taux sans risque utilisé pour l'inversion BS
    min_volume      : int   — Volume minimum pour garder un contrat (liquidité)
    max_expirations : int   — Nombre maximum de maturités à récupérer
                               (limite le temps de calcul ; les maturités
                               proches sont les plus liquides et les plus
                               utiles pour un smile/surface lisible)

    Retourne
    --------
    pd.DataFrame avec les colonnes :
        strike, expiration, T (années), mid_price, S (spot), iv
    """
    import yfinance as yf

    tk = yf.Ticker(ticker_symbol)
    spot = tk.fast_info['last_price']
    expirations = tk.options[:max_expirations]

    if not expirations:
        raise ValueError(f"Aucune date d'expiration trouvée pour {ticker_symbol}")

    today = pd.Timestamp.today().normalize()
    rows = []

    for exp_str in expirations:
        chain = tk.option_chain(exp_str)
        df = chain.calls if option_type == 'call' else chain.puts

        T = (pd.Timestamp(exp_str) - today).days / 365.0
        if T <= 0:
            continue  # Maturité déjà passée / expire aujourd'hui, on saute

        # Filtre de liquidité : bid/ask non nuls et volume suffisant
        df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['volume'].fillna(0) >= min_volume)]

        for _, row in df.iterrows():
            mid_price = (row['bid'] + row['ask']) / 2.0

            iv = implied_vol(mid_price, spot, row['strike'], T, r, option_type)
            if iv is None:
                continue  # Prix incohérent (arbitrable) -> on ignore ce point

            rows.append({
                'strike':     row['strike'],
                'expiration': exp_str,
                'T':          T,
                'mid_price':  mid_price,
                'S':          spot,
                'iv':         iv,
            })

    if not rows:
        raise ValueError(
            f"Aucun contrat exploitable pour {ticker_symbol} "
            f"(essayer de baisser min_volume ou un autre ticker)"
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# NETTOYAGE / FILTRAGE DES OUTLIERS
# ─────────────────────────────────────────────

def clean_iv_data(df, iv_min=0.01, iv_max=3.0, moneyness_range=(0.5, 1.5)):
    """
    Filtre les points de vol implicite aberrants avant de tracer.

    Deux sources de bruit typiques sur des données réelles :
      - Vol implicite extrême (>300% ou quasi nulle) : souvent des contrats
        très peu liquides où le mid-price bid/ask est peu fiable.
      - Strikes très éloignés du spot (moneyness = K/S hors [0.5, 1.5]) :
        options très loin de la monnaie, peu informatives pour le smile
        et souvent illiquides.

    Paramètres
    ----------
    df              : pd.DataFrame — Sortie de fetch_option_chain()
    iv_min, iv_max  : float — Bornes acceptables de vol implicite
    moneyness_range : tuple — Bornes acceptables de K/S

    Retourne
    --------
    pd.DataFrame filtré
    """
    df = df.copy()
    df['moneyness'] = df['strike'] / df['S']

    mask = (
        (df['iv'] >= iv_min) & (df['iv'] <= iv_max) &
        (df['moneyness'] >= moneyness_range[0]) &
        (df['moneyness'] <= moneyness_range[1])
    )
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────
# SMILE DE VOLATILITÉ (2D, une maturité)
# ─────────────────────────────────────────────

def plot_vol_smile(df, ticker_symbol=''):
    """
    Trace le smile de volatilité : vol implicite vs strike, une courbe
    par maturité disponible.

    Interprétation du "smile" :
      - En théorie Black-Scholes, sigma est constant (une seule courbe plate).
      - En pratique, la vol implicite est plus élevée pour les strikes loin
        de la monnaie (surtout les puts OTM = crash protection très demandée)
        -> forme de "sourire" ou de "smirk" (asymétrique sur actions/indices).
      - C'est la preuve empirique la plus visible que l'hypothèse de
        volatilité constante de Black-Scholes est fausse en pratique.

    Paramètres
    ----------
    df            : pd.DataFrame — Sortie de fetch_option_chain() (nettoyée)
    ticker_symbol : str — Utilisé uniquement pour le titre du graphe

    Retourne
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    for exp, group in df.groupby('expiration'):
        group = group.sort_values('strike')
        T_years = group['T'].iloc[0]
        fig.add_trace(go.Scatter(
            x=group['strike'],
            y=group['iv'] * 100,
            mode='lines+markers',
            name=f"{exp}  (T={T_years:.2f}y)",
        ))

    spot = df['S'].iloc[0]
    fig.add_vline(x=spot, line_dash='dash', line_color='gray',
                   annotation_text=f'Spot = {spot:.2f}')

    fig.update_layout(
        title=f"Smile de volatilité implicite — {ticker_symbol}".strip(' —'),
        xaxis_title='Strike (K)',
        yaxis_title='Vol implicite (%)',
        template='plotly_white',
        legend_title='Maturité',
    )
    return fig


# ─────────────────────────────────────────────
# SURFACE DE VOLATILITÉ (3D)
# ─────────────────────────────────────────────

def plot_vol_surface(df, ticker_symbol='', grid_points=60):
    """
    Trace la surface de volatilité implicite en 3D : vol implicite en
    fonction du strike ET de la maturité.

    Problème technique : les points (strike, T, iv) récupérés du marché
    forment un nuage IRRÉGULIER (chaque maturité n'a pas les mêmes strikes
    cotés). Pour tracer une surface continue avec Plotly (go.Surface), il
    faut d'abord interpoler ces points épars sur une grille régulière.

    On utilise scipy.interpolate.griddata (interpolation linéaire en 2D)
    pour estimer iv(K, T) sur une grille fine, à partir des points observés.
    Les zones hors de l'enveloppe convexe des points connus sont laissées
    à NaN (pas d'extrapolation hasardeuse).

    Paramètres
    ----------
    df            : pd.DataFrame — Sortie de fetch_option_chain() (nettoyée)
    ticker_symbol : str — Utilisé uniquement pour le titre du graphe
    grid_points   : int — Résolution de la grille d'interpolation

    Retourne
    --------
    plotly.graph_objects.Figure
    """
    strikes = df['strike'].values
    maturities = df['T'].values
    ivs = df['iv'].values * 100  # en %

    # Grille régulière sur laquelle interpoler
    K_grid = np.linspace(strikes.min(), strikes.max(), grid_points)
    T_grid = np.linspace(maturities.min(), maturities.max(), grid_points)
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    # Interpolation linéaire des points épars (K, T, iv) -> grille (K_mesh, T_mesh)
    IV_mesh = griddata(
        points=(strikes, maturities),
        values=ivs,
        xi=(K_mesh, T_mesh),
        method='linear',
    )

    fig = go.Figure(data=[go.Surface(
        x=K_mesh, y=T_mesh, z=IV_mesh,
        colorscale='Viridis',
        colorbar=dict(title='Vol (%)'),
    )])

    # On superpose les points de marché bruts (petits points noirs) pour
    # visualiser où se trouvent les vraies données vs la zone interpolée
    fig.add_trace(go.Scatter3d(
        x=strikes, y=maturities, z=ivs,
        mode='markers',
        marker=dict(size=2, color='black'),
        name='Données de marché',
    ))

    fig.update_layout(
        title=f"Surface de volatilité implicite — {ticker_symbol}".strip(' —'),
        scene=dict(
            xaxis_title='Strike (K)',
            yaxis_title='Maturité T (années)',
            zaxis_title='Vol implicite (%)',
        ),
        template='plotly_white',
    )
    return fig


# ─────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────

def build_and_plot(ticker_symbol, option_type='call', r=0.05, save_html=False):
    """
    Pipeline complet : récupère les données, nettoie, trace smile + surface.

    Paramètres
    ----------
    ticker_symbol : str  — Symbole boursier (ex: 'AAPL')
    option_type   : str  — 'call' ou 'put'
    r             : float — Taux sans risque
    save_html     : bool — Si True, sauvegarde les deux graphes en .html
                            dans le dossier courant

    Retourne
    --------
    dict avec 'data' (DataFrame nettoyé), 'smile_fig', 'surface_fig'
    """
    print(f"Récupération de la chaîne d'options pour {ticker_symbol}...")
    raw = fetch_option_chain(ticker_symbol, option_type=option_type, r=r)
    print(f"  {len(raw)} contrats récupérés avant nettoyage")

    df = clean_iv_data(raw)
    print(f"  {len(df)} contrats conservés après filtrage (liquidité + moneyness)")

    smile_fig = plot_vol_smile(df, ticker_symbol)
    surface_fig = plot_vol_surface(df, ticker_symbol)

    if save_html:
        smile_fig.write_html(f"{ticker_symbol}_smile.html")
        surface_fig.write_html(f"{ticker_symbol}_surface.html")
        print(f"  Graphes sauvegardés : {ticker_symbol}_smile.html, {ticker_symbol}_surface.html")

    return {'data': df, 'smile_fig': smile_fig, 'surface_fig': surface_fig}


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    TICKER = 'SPY'  # Un ETF très liquide -> bonne chaîne d'options pour tester

    print("=" * 60)
    print("VOLATILITY SURFACE — DÉMONSTRATION")
    print("=" * 60)

    result = build_and_plot(TICKER, option_type='call', save_html=True)

    df = result['data']
    print(f"\nAperçu des données ({TICKER}, calls) :")
    print(df[['expiration', 'strike', 'T', 'mid_price', 'iv']].head(10).to_string(index=False))

    result['smile_fig'].show()
    result['surface_fig'].show()
