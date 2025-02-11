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
    def __init__(self, ticker):
        self.ticker = ticker

    def get_bars(self, minutes, days, is_list=False):
        """Retrieve historical bars from a pickle file."""
        # Get the absolute path of the current script
        script_path = Path(__file__).resolve()

        # Find the root directory (parent of the system folder)
        root_dir = script_path.parent.parent

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
            df['fecha'] = df['date']
            regs = int(days * (1440 / minutes))
            df_sliced = df[-regs:]
            # index need a reset
            df_sliced = df_sliced.reset_index()
            if is_list:
                df_sliced = df_sliced.to_dict(orient='records')
            return df_sliced
        except Exception as e:
            raise ValueError(f"an errror was commited: {e}")
