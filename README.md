# Option_pricing


Black-Scholes, Binomial Tree and Monte Carlo option pricing models

An option is a financial contract that gives you the right (but not the obligation) to buy or sell a stock at a price fixed in advance, called the strike price K, at a future date called maturity T.



## Black-Scholes model

Black-Scholes assumes that the price of a stock follows a geometric Brownian motion, according to the formula below:

$ dS=μSdt+σSdW_t ​$


## Binomial tree model

The assumption for Black-Scholes is that the time is continuous. For american options where we can exercise at any time, we will use a binomial tree to discretize time











```bash
options-pricing/
├── models/
│   ├── black_scholes.py
│   ├── binomial_tree.py
│   └── monte_carlo.py
├── visualisation/
│   └── vol_surface.py
├── notebooks/
│   └── demo.ipynb
├── README.md
└── requirements.txt
```


