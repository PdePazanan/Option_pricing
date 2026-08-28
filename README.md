# Option pricing

Implementation and comparison of three classic option pricing models: Black-Scholes, Binomial Tree (Cox-Ross-Rubinstein), and Monte Carlo simulation.

## Overview

An option is a financial contract that gives its holder the right (not the obligation) to buy (call) or sell (put) an underlying asset at a price fixed in advance (the strike, K) at a future date (the maturity, T). 
The important question in options theory is how to price this right today, given the current price of the underlying, its volatility, and the risk-free rate.




## Black-Scholes model

Black-Scholes assumes that the price of a stock follows a geometric Brownian motion, according to the formula below:

$ dS=μSdt+σSdW_t ​$


## Binomial tree model

The assumption for Black-Scholes is that the time is continuous. For american options where we can exercise at any time, we will use a binomial tree to discretize time




## Monte Carlo model 

With this method, we will simulate a lot of trajectories of the price of the option till the maturity. Then calculate the mean of all these final price.



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





### References 

Options, Futures, and Other Derivatives — John Hull 

