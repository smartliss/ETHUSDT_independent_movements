import numpy as np
import pickle
import asyncio
import time
from aiohttp import ClientSession
import platform

# this line of code is needed to avoid issues with using aiohttp and Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# load the OLS model (from the task part 1)
filename = "./ols.pickle"
ols = pickle.load(open(filename, "rb"))


class IndependentMovements:

    """This class monitors prices of ETH (target symbol) and BTC (influencing symbol) within a particular time
    interval, exclude the influence of BTC to ETH, and send messages if independent ETH movements exceeded a particular
    threshold within the time interval. If the time interval is over or independent ETH movements exceeded the threshold,
    the process of monitoring begins again (without code interruption)."""

    def __init__(self, time_period: int = 3600, max_independent_change: float = 1.0):
        self.time_period = time_period  # time interval to check price changes
        self.max_change = max_independent_change  # threshold for independent price change (in %)
        self.last_influence_price = None  # previous price of the influencing symbol
        self.last_target_price = None  # previous price of the target symbol
        self.reference_time = time.time()  # start point for the time interval
        self.total_independent_change = 0  # total independent change (movements) of the target symbol

    def check_independent_movements(self, influence_price: float, target_price: float):

        """ Monitor prices and send messages if the time interval is over or independent ETH movements exceeded
        the threshold"""

        # check if it is not the first request (self.last_target_price and self.last_influence_price is not None)
        if all((self.last_influence_price, self.last_target_price)):

            # Relative price changes
            target_change = (target_price - self.last_target_price) / self.last_target_price * 100
            influence_change = (influence_price - self.last_influence_price) / self.last_influence_price * 100

            # exclude the influence from the target symbol
            target_independent_change = target_change - ols.predict(np.array([influence_change]).reshape(-1, 1))[0][0]
            self.total_independent_change += target_independent_change  # independent target change

            # time checkpoint to calculate the time interval
            now = time.time()
            time_passed = now - self.reference_time  # time that's passed since the reference_time

            # check conditions to renew monitoring
            if abs(self.total_independent_change) > self.max_change or time_passed > self.time_period:

                # check condition for printing the message
                if abs(self.total_independent_change) > self.max_change:
                    print("For the last {time} min".format(time=time_passed / 60))
                    print("independent price change = {change}%".format(change=self.total_independent_change))

                # Reset the reference time and price changes
                self.reference_time = now
                self.total_independent_change = 0

        # set (for the 1st request) or reset previous prices to the current ones
        self.last_influence_price = influence_price
        self.last_target_price = target_price


async def get_last_price(symbol: str) -> tuple:

    """Makes asynchronous requests to Binance Futures API and returns
    the latest price for the given symbol"""

    # create session for aiohttp
    async with ClientSession() as session:
        url = 'https://fapi.binance.com/fapi/v1/ticker/price'
        params = {'symbol': symbol}

        # asynchronous requests to Binance
        async with session.get(url=url, params=params) as response:
            price_json = await response.json()
            return symbol, float(price_json['price'])


async def prices(symbols: list) -> dict:

    """Wraps get_last_price function for asynchronous requests"""

    tasks = []
    for symbol in symbols:
        tasks.append(asyncio.create_task(get_last_price(symbol)))

    results = await asyncio.gather(*tasks)
    return dict(results)

# create an instance of the monitoring class
monitor = IndependentMovements(time_period=3600, max_independent_change=1.0)

influence_symbol = 'BTCUSDT'
target_symbol = 'ETHUSDT'
symbols = [influence_symbol, target_symbol]


if __name__ == '__main__':
    print('Start monitoring')
    while True:
        # get the latest prices
        price_dict = asyncio.run(prices(symbols))
        # monitor and print messages (if required)
        monitor.check_independent_movements(price_dict[influence_symbol], price_dict[target_symbol])
