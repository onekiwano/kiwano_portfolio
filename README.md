
# Kiwano Portfolio

The purpose of this project is to provide an interface with Binance now, and more exchanges to come,
to be able to backtest, livetest and livetrade your trading algorithms on crypto-currencies.
We thought we could let you focus all your creativity and effort on your trading strategies, 
and let us handle the rest, since this part is simpler but necessary.

The code is composed of two classes, a `portfolio` manager 
that tracks the progress over time and provides useful statistics 
on the performance of the trading algorithm. And the 
inheriting class `strategy` that plays the trading algo.

## Functionalities

To understand how to use this repository, have a look at the ```main.py``` file.
The typical way to use the code is the following:
1. Create `portfolio`;
2. Add a `strategy`;
3. Run in one of the three *modes* (see below);
4. Obtain log and plots displaying the performance of your trading strategy.

The three *modes* are:
- Backtesting (or FastBacktesting if the strategy is loops already in time)
- Livetesting
- Livetrading

The workflow of `strategy` is:
1. Compute the *metrics* on data
2. Compute the first layer *strategy* to create `order`
3. Compute the second layer *strategy* to make the `order` effective in the `portfolio`

## ToDo
- [CLEAN] Code + simplify `strategy` class [DONE]
- [CODE] Add multiple crypto input [DONE]
- [CODE] multiple crypto in output

 
