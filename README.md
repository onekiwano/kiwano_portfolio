
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
- **Backtesting** (or FastBacktesting if the strategy loops already in time): will 
run your algorithm on historical data, and will provide the performance
that your algorithm would have had if it was trading in the past.
- **Livetesting**: will run your algorithm on real-time data, and will provide the performance that your algorithm would have had if it was trading live, but
without actually trading with real money.
- **Livetrading**: will run your algorithm on real-time data, and will provide the performance 
that your algorithm is having while trading live, with real money.

The workflow of `strategy` is:
1. Compute the *metrics* on data
2. Compute the first layer *strategy* to create `order`
3. Compute the second layer *strategy* to make the `order` effective in the `portfolio`

## Requirements

Install the content of the requirements.txt file with ```pip install -r requirements.txt```. 
You will then need to create an API key and secret on Binance to be able to use the code.
Go to API Management in Settings on Binance, and click on Create API. Select System generated API key,
name the API, and once you pass the authentication, you have to 
edit restriction and tick Enable Spot and Margin Trading if you want to livetrade with your algorithm.
This is not necessary if you only want to backtest or livetest. You can then copy the API key and secret
in a ```config.py```  file and define the variables as strings with the following names:

```python
api_key = 'your key as a string'
api_secret = 'your secret as a string'
```
For an example check the `config_template.py` in the `API/binance/` folder.
You can finally save the file in the ```API/binance``` folder or pass the filepath
as the `api_keys_location` argument when creating the portfolio. Then you are ready to go!



## ToDo
- ~~[CLEAN] Code + simplify `strategy` class [DONE]~~
- ~~[CODE] Add multiple crypto input [DONE]~~
- [CODE] multiple crypto in output

## Social media
Want to participate in the Kiwano adventure, join our Discord !

 
<a href="https://discord.gg/698CKv8t">
<img src="https://it.moobion.com/wp-content/uploads/2020/11/discord-logo.png" alt="https://discord.gg/698CKv8t" style="width:48px;height:48px;">
</a>
