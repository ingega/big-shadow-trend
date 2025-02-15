import pandas as pd
from pathlib import Path
# import numpy as np
# from decorators import timer
# import time


ticks = [["ETHUSDT", 2], ["BATUSDT", 4], ["MKRUSDT", 1], ["FLMUSDT", 4],
         ["ANKRUSDT", 5], ["MTLUSDT", 4], ["CTSIUSDT", 4],
         ["1000LUNCUSDT", 7], ["CFXUSDT", 7], ["XVSUSDT", 6]
         ]


class Bars:
    """
    This class return and works with pkl files or can be
    extended to get data from db or directly from an API
    this is the feed of the all work below
    """
    def __init__(self, ticker):
        """ pkl name files are composed by the time frame and ticker"""
        self.ticker = ticker

    def get_bars(self, minutes, days, is_list=False):
        """Retrieve historical bars from a pickle file."""
        # Get the absolute path of the current script
        script_path = Path(__file__).resolve()

        # Find the root directory (parent of the system folder)
        root_dir = script_path.parent.parent.parent

        # Construct the bars folder path dynamically
        bars_dir = root_dir / "bars"

        # Ensure bars directory exists
        if not bars_dir.exists():
            raise FileNotFoundError(f"Bars directory not found: {bars_dir}")

        # Construct file name based on the timeframe
        file_name = f"{minutes}{self.ticker}.pkl" \
            if minutes > 1 else f"{self.ticker}.pkl"
        file_path = bars_dir / file_name
        try:
            df = pd.read_pickle(file_path)
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            regs = int(days * (1440 / minutes))
            df_sliced = df[-regs:]
            # index need a reset
            df_sliced = df_sliced.reset_index()
            if is_list:
                df_sliced = df_sliced.to_dict(orient='records')
            return df_sliced
        except Exception as e:
            raise ValueError(f"an error was commited: {e}")


class Filter:
    def __init__(self, bars: Bars, minutes: int, days: int, **kwargs):
        """
        Filters market conditions to find valid entry points based on
        strategy rules.

        :param bars: Bars object containing OHLCV data.
        :param minutes: Timeframe for each bar.
        :param days: Number of days to fetch.
        :param kwargs: Filtering parameters:
            - body_size: Ratio of (abs(open - close) / open)
            - shadow_up: Ratio of (high - open) / open
            - shadow_down: Ratio of (low - open) / open
            - color: "green" (bullish) or "red" (bearish)
        """
        self.bars = bars
        self.minutes = minutes
        self.days = days
        self.params = kwargs

    def apply_filter(self):
        """
        Applies filtering conditions to identify potential entry points.

        Entry Conditions:
        - buy opportunity:
            - small body
            - large shadow_up
            - next bar is green
        - sell opportunity: the same, but for bearish
        Returns:
            pd.DataFrame: Filtered dataset with valid entry points.
        """
        df = self.bars.get_bars(self.minutes, self.days)

        # index control is necessary for further ops
        df = df.reset_index(drop=True)
        df['index'] = df.index

        # ✅ Compute necessary values for filtering
        df["body_size"] = abs(df["open"] - df["close"]) / df["open"]
        # using numpy is faster
        df["side"] = ["BUY" if c > o else "SELL"
                      for c, o in zip(df["close"], df["open"])]
        # 🧿 we can define uper_shadow, by one side,
        # 🧿 and then update the other one
        df["upper_shadow"] = (df["high"] - df["close"]) / df["close"]
        df.loc[df["side"] == "SELL", "upper_shadow"] \
            = (df["high"] - df["open"]) / df["open"]
        # 🧿 and the same for lower shadow
        df["lower_shadow"] = (df["open"] - df["low"]) / df["open"]
        df.loc[df["side"] == "SELL", "lower_shadow"] \
            = (df["close"] - df["low"]) / df["close"]
        df["next_bar_side"] = df["side"].shift(-1)

        # Apply conditions from kwargs
        filtered_df = df.loc[
            (
                    (
                            (df['side'] == 'BUY')
                            & (df['upper_shadow'] > self.params['shadow'])
                            & (df['next_bar_side'] == 'BUY')
                    )
                    |
                    (
                            (df['side'] == 'SELL')
                            & (df['lower_shadow'] > self.params['shadow'])
                            & (df['next_bar_side'] == 'red')
                    )
            )
            & (df['body_size'] < self.params['body'])
            ]

        return filtered_df.to_dict(orient='records')
