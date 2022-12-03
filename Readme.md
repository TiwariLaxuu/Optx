# Optx

This package implements a lightweight library, written in Cython, whose purpose is to quickly evaluate option strategies. Usage examples can be found in the *examples* directory.

Among the outputs of the code, we have the payoff of the strategy, the range of stock prices for which the strategy is profitable (i.e., return greater than \$0.01), the Greeks for each leg of the strategy, the resulting debit or credit on the trading account, the maximum profit and loss within a lower and higher price of the underlying asset and an estimate of the strategy's probability of profit.

The probability of profit (PoP) of an options strategy is calculated from the distribution of estimated prices of the underlying asset on the user-defined target date.  More specifically, for the price range in the payoff for which the strategy is profitable, the PoP is the probability that the stock price will be within that range. This distribution of underlying asset prices on the target date can be lognormal, lognormal with risk neutrality (as in the Black-Scholes model) or log-Laplace, as well as obtained from simulations (e.g., Monte Carlo) or machine learning models.

Although functional, *Optx* is still in an early stage of development, including its documentation, and is provided as is. The author makes no guarantee that its results are accurate and is not responsible for any losses caused by the use of the code. If you have any questions, comments, suggestions or corrections, just [drop a message](mailto:roberto.veiga@ufabc.edu.br).