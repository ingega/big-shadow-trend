import numpy as np
import pandas as pd
from src.bets import single_bet
from src.functions import Bars
import time

ticks = [["ETHUSDT", 2], ["BATUSDT", 4], ["MKRUSDT", 1], ["FLMUSDT", 4],
         ["ANKRUSDT", 5], ["MTLUSDT", 4], ["CTSIUSDT", 4],
         ["1000LUNCUSDT", 7], ["CFXUSDT", 7], ["XVSUSDT", 6]
         ]


class Analysis:
    """
    This class contains every analysis of the bet
    returns: winner and all bets in different parameters
    """

    def __init__(self, ticker, minutes, list_min,
                 days, reverse=False, **kwargs):
        self.ticker = ticker
        self.minutes = minutes
        self.list_min = list_min
        self.days = days
        self.reverse = reverse
        self.bet_continue = kwargs.get("bet_continue")
        self.max_bars = kwargs.get('max_bars')

    def analysis(self):
        """
        params:
            shadow, range(0.005, 0.02)
            body, range(0.005, 0.01)
            bet, range(0.03, 0.05)
        """
        alls = []
        for shadow in np.arange(0.015, 0.0351, 0.005):
            for body in np.arange(0.01, 0.041, 0.005):
                for bet in np.arange(0.03, 0.051, 0.01):
                    s = single_bet(self.ticker,
                                   self.list_min, self.minutes, self.days,
                                   shadow=shadow,
                                   body=body, bet=bet,
                                   reverse=self.reverse,
                                   bet_continue=self.bet_continue,
                                   max_bars=self.max_bars
                                   )
                    if s is not None:
                        s['ticker'] = self.ticker
                        alls.append(s)
        final, ret = None, None
        if len(alls) > 0:
            final = pd.concat(alls)
            ret = {
                'alls': final,
                'winner': final.sort_values(
                    by='sum', ascending=False).head(1).to_dict(
                    orient='records')
            }
        return ret


def analysis_ten(minutes, days, reverse=False, **kwargs):
    """internal method for extraction of parameters"""
    bet_continue = kwargs.get('bet_continue')
    max_bars = kwargs.get('max_bars')
    acum = []  # acumulate all combinations
    for ticker in ticks:
        list_min = Bars(ticker[0]).get_bars(
            minutes=1, days=days, is_list=True)
        analysis = Analysis(
            ticker=ticker[0], minutes=minutes, days=days,
            list_min=list_min, reverse=reverse,
            bet_continue=bet_continue,
            max_bars=max_bars).analysis()
        records = analysis['alls']
        acum.append(records)
        print(analysis['winner'], time.ctime())
    if len(acum) == 0:
        return
    """parameters go here"""
    param1 = 'shadow'
    param2 = 'body'
    df = pd.concat(acum)
    # group by parameters to find the best fit
    grouped = pd.DataFrame(df.groupby([param1, param2, 'bet'])['sum'].mean())
    # order by sum, only the first 10 are intereseting
    grouped = grouped.sort_values(by='sum', ascending=False).head(10)
    ret = {
        'alls': df,
        'grouped': grouped,
    }

    return ret
