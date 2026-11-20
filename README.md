# Option pricing
In this project, I implemented three classical option pricing methods: Black-Scholes, Binomial Tree (Cox-Ross-Rubinstein), and Monte Carlo Simulation.

The goal was to understand how these models work, implement them in Python, and compare their results.


## Overview

An option is a financial contract that gives its holder the right (not the obligation) to buy (call) or sell (put) an underlying asset at a price fixed in advance (the strike, K) at a future date (the maturity, T). 
The important question in options theory is how to price this right today, given the current price of the underlying, its volatility, and the risk-free rate.
The price of an option depends on several factors, mainly the current price of the underlying asset, its volatility, the time to maturity, and the risk-free interest rate.


## Black-Scholes model

The Black-Scholes model is one of the most used methods for pricing European options. It provides a direct mathematical formula for calculating the theoretical price of an option.

The model assumes that the underlying asset follows a Geometric Brownian Motion:

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

Where:

- $S_t$: Underlying asset price.
- $\mu$: Expected return.
- $\sigma$: Volatility.
- $W_t$: Standard Brownian motion.


Some of the main assumptions are constant volatility, a constant risk-free rate, and no transaction costs.

## Binomial tree model

The assumption for Black-Scholes is that the time is continuous. For american options where we can exercise at any time, we will use a binomial tree to discretize time because the Binomial Tree model uses a different approach by dividing the time to maturity into several steps. At each step, the underlying asset price can either go up or down.

Starting from the possible prices at maturity, the model works backwards through the tree to calculate the option price today.

One advantage of this method is that it can also be used to price American options, since it takes early exercise into account.


## Monte Carlo model 

Monte Carlo simulation uses random numbers to generate a large number of possible future price paths for the underlying asset.

For each path, the option payoff at maturity is calculated. The final option price is then estimated by taking the average of all discounted payoffs.

The accuracy of the estimation generally improves as the number of simulations increases. This method is particularly useful for pricing more complex derivatives.

```text
options-pricing/
|-- models/
|    |-- black_scholes.py
|    |-- binomial_tree.py
|    |-- monte_carlo.py
|-- visualisation/
|    |-- vol_surface.py
|-- notebooks/
|    |-- demo.ipynb
|-- README.md
|-- requirements.txt
```


## Visualisation

I also implemented a volatility surface visualisation to see how option prices evolve depending on different strike prices and volatility levels.

### Technologies

Python – Main programming language.

NumPy – Numerical computations.

Matplotlib – Data visualisation.

Jupyter Notebook – Testing and comparing the different models.


### References 

Options, Futures, and Other Derivatives — John Hull 

