import time

import pandas as pd

from src.functions import Bars, Filter


def get_profit(side, price_in, price_out):
    if side == 'BUY':
        profit = (price_out - price_in) / price_in
    elif side == 'SELL':
        profit = (price_in - price_out) / price_in
    else:
        raise ValueError("side is bad formed or invalid")
    profit -= 0.0008  # commission

    return profit


class Bet:
    """
    This class takes a filter and then iterates it to get
    the outcome of a bet.
    Parameter reverse is for make a reverse trend opportunity
    """

    def __init__(self, list_min: list, minutes: int,
                 bet: float, sl: float, sl_max: float,
                 tp: float, reverse=False, **kwargs):
        """set the parameters and kwargs"""
        self.bet = bet
        self.minutes = minutes
        self.params = kwargs
        self.list_min = list_min
        self.sl = sl
        self.sl_max = sl_max
        self.tp = tp
        self.reverse = reverse
        self.adjust = 0  # this variable controls the loss
        self.bars_elapsed = 0
        # value by the fault -> 1440
        if kwargs.get('max_bars') is not None:
            self.maximum_bars = kwargs.get('max_bars')
        else:
            self.maximum_bars = 1440  # this can be changed by kwargs
        # continue is for loop until reach max bars without sl
        self.bet_continue = kwargs.get('bet_continue')

    def prepare_bet(self, filtered_record):
        """
        Evaluates a single bet outcome.
        Returns:
            dict: Outcome details if bet is resolved, else None.
        """
        # the bar index 1, multiply by minutes, must be added 1 bar
        # in order to get the next date possible
        pointer = (filtered_record['index'] + 1) * self.minutes
        # side depends on reverse
        if self.reverse:
            if filtered_record['side'] == 'BUY':
                side = "SELL"
            else:
                side = "BUY"
        else:
            side = filtered_record['side']
        price_in = filtered_record['close']
        date_in = filtered_record['date']
        if pointer >= len(self.list_min):
            return None
        if side == 'BUY':
            winner_price = price_in * (1 + self.bet)
            losser_price = price_in * (1 - self.bet)
        else:
            winner_price = price_in * (1 - self.bet)
            losser_price = price_in * (1 + self.bet)

        result = self.check_bet(side, pointer, winner_price, losser_price)
        outcome = None

        if result:
            outcome = {
                'side': side,
                'price_in': price_in,
                'date_in': date_in,
                'price_out': result['price_out'],
                'date_out': result['date_out'],
                'pointer_in': pointer,
                'pointer_out': result['pointer'],
                'outcome': result['outcome'],
                'way': 'direct',
            }
        return outcome

    def check_bet(self, side, pointer, winner_price,
                  losser_price, protect=False):
        """
        Checks whether the bet is won or lost based on price movements.

        Returns:
            dict: Outcome data if resolved, else None.
        """
        the_list = self.list_min
        outcome = None
        for p in range(pointer, len(the_list)):
            if ((side == 'BUY' and the_list[p]['high'] >= winner_price) or
                    (side == "SELL" and the_list[p]['low'] <= winner_price)):
                outcome = {
                    'date_out': the_list[p]['date'],
                    'price_out': winner_price,
                    'pointer': p,
                    'outcome': 'tp',  # take profit
                }
                break
            elif ((side == "BUY" and the_list[p]['low'] <= losser_price) or
                  (side == "SELL" and the_list[p]['high'] >= losser_price)):
                outcome = {
                    'date_out': the_list[p]['date'],
                    'price_out': losser_price,
                    'pointer': p,
                    'outcome': 'sl',  # stop loss
                }
                break
            self.bars_elapsed += 1
            if self.bars_elapsed >= self.maximum_bars and protect:
                # declare a tie
                outcome = {
                    'date_out': the_list[p]['date'],
                    'price_out': the_list[p]['close'],
                    'pointer': p,
                    'outcome': 'tie',  # stop loss
                    'way': "indirect",  # control adjust
                }
                # the control of self.bars is in the main bet area
                break
        return outcome

    def protect(self, pointer, price_in, side):
        """
        in this loop, the goal is acumulate the loss value
        """
        while True:
            # if kwargs continue then, the finale is until reach
            # self.max_bars without sl, or until sl or tp
            if self.bet_continue:
                # reset the elapsed bars to force reach the max
                # bars without sl
                self.bars_elapsed = 0
            if side == 'BUY':
                winner_price = price_in * (1 + self.tp + self.adjust)
                losser_price = price_in * (1 - self.sl)
            else:
                winner_price = price_in * (1 - self.tp - self.adjust)
                losser_price = price_in * (1 + self.sl)
            # remember that the maximum time is 1440 bars
            result = self.check_bet(side, pointer, winner_price, losser_price,
                                    protect=True)
            if result:
                # depends on outcome
                if result['outcome'] == 'tp':
                    """this is a final outcome"""
                    profit = self.tp - 0.0008
                    break
                elif result['outcome'] == 'tie':
                    """this is a final outcome"""
                    # step 1: get the profit
                    profit = get_profit(side, price_in, result['price_out'])
                    profit -= self.adjust
                    break
                else:  # is sl, check for sl_max
                    """ change the side, price_in and adjust"""
                    self.adjust += self.sl + 0.0008  # commission
                    if self.adjust > self.sl_max:
                        profit = -self.adjust
                        break
                    if side == "BUY":
                        side = "SELL"
                    else:
                        side = "BUY"
                    pointer = result['pointer'] + 1  # next bar star
                    price_in = result['price_out']
            else:
                return None
        outcome = {
            'date_out': result['date_out'],
            'price_out': result['price_out'],
            'pointer': result['pointer'],
            'outcome': result['outcome'],
            'way': "indirect",
            'profit': profit,
        }
        return outcome


def single_bet(ticker, list_min, minutes, days, reverse=False, **kwargs):
    """
    kwargs is used to group totals in the analysis section
    function:
    Executes a series of bets based on the given filters.
    Returns:
        df: df with bet outcomes.
    """
    bars = Bars(ticker)
    shadow = kwargs.get('shadow')
    body = kwargs.get('body')
    bet_value = kwargs.get('bet')
    bet_continue = kwargs.get('bet_continue')
    max_bars = kwargs.get('max_bars')
    sl = 0.0125  # this value can be changed by kwargs

    f = Filter(bars, minutes=minutes, days=days, shadow=shadow, body=body)
    filtered = f.apply_filter()
    bet = Bet(list_min=list_min, minutes=minutes, bet=bet_value,
              sl=sl, sl_max=0.1, tp=0.1,
              reverse=reverse, bet_continue=bet_continue,
              max_bars=max_bars
              )

    final = []
    for record in filtered:
        # now, final have the date of the last outcome
        if len(final) > 0 and final[-1]['date_out'] > record['date']:
            # this record in real life can't be reached
            continue
        outcome = bet.prepare_bet(record)
        if outcome:
            if outcome['outcome'] == 'tp':
                outcome['profit'] = bet_value - 0.0008  # commission
            else:
                # send to protect, get the side
                if record['side'] == "BUY":
                    side = "SELL"
                else:
                    side = "BUY"
                # set adjust
                bet.adjust = bet_value + 0.0008
                # well due that the date_out field is needed to control
                # next filtered_record, the indir_date will be the date
                # where the bet lost
                outcome['indir_date'] = outcome['date_out']
                # finally, in order to be sure that only bars
                # allowed will be iterating, set the bars
                bet.bars_elapsed = 0
                result = bet.protect(outcome['pointer_out'],
                                     outcome['price_out'], side)
                if result:
                    # complete the outcome using zip
                    keys = ['price_out', 'outcome', 'way',
                            'profit', 'date_out']
                    values = [result['price_out'], result['outcome'],
                              result['way'], result['profit'],
                              result['date_out']
                              ]
                    outcome.update({k: v for k, v in zip(keys, values)})
                    # also add the indirect values, the indir_pointer
                    # give us the pointer of exit
                    outcome['indir_pointer'] = result['pointer']
            if outcome:
                final.append(outcome)
    # return into a pd
    if len(final) == 0:
        return None
    df = pd.DataFrame(final)
    # the returned info must be summarized
    positives = df.loc[df['profit'] > 0]['profit'].count() / len(df)
    directs = df.loc[df['way'] == 'direct']['way'].count() / len(df)
    percentage_sl = df.loc[df['outcome'] == 'sl']['way'].count() / len(df)
    percentage_tp = (df.loc[
                        (df['outcome'] == 'tp')
                        & (df['way'] == 'indirect')]['way'].count()
                     / len(df))
    tie_avg = df.loc[df['outcome'] == 'tie']['profit'].mean()
    """add the params here, in way of duple"""
    key1, value1 = 'shadow', shadow
    key2, value2 = 'body', body
    if kwargs.get('getEntries'):
        # also the parameter must be set in True
        get_entries = df.to_dict(orient='records')  # avoid multi-df
    else:
        get_entries = False
    ret = {
        'ticker': ticker,
        'sum': round(df['profit'].sum(), 3),
        'positives': round(positives, 3),
        'records': len(df),
        'directs': round(directs, 3),
        'sl': round(percentage_sl, 3),
        'tp': round(percentage_tp, 3),
        'tie_avg': round(tie_avg, 3),
        key1: value1,
        key2: value2,
        'bet': bet_value,  # bet is always present
        'entries': get_entries,
    }
    # if getEntries, return in dict
    if ret['entries']:
        return ret
    else:
        df_final = pd.DataFrame(ret, index=[0])
        return df_final


def all_tickers(minutes, days, reverse=False, **kwargs):
    """
    This function returns the list of every ticker in
    tickers.csv, is functional because if more than 50% of
    tickers are positives, the strategy beats for too much the
    randomness test
    """
    tickers = pd.read_csv("src/tickers.csv").to_dict(orient='records')
    acumulate = []
    """change your params here"""
    shadow = kwargs.get('shadow')
    body = kwargs.get('body')
    """params always present"""
    bet = kwargs.get('bet')
    bet_continue = kwargs.get('bet_continue')
    max_bars = kwargs.get('max_bars')
    for ticker in tickers:
        list_min = Bars(ticker['ticker']).get_bars(
            minutes=1, days=days, is_list=True)
        s = single_bet(
            ticker['ticker'], list_min, minutes, days,
            shadow=shadow, body=body, bet=bet,
            reverse=reverse, bet_continue=bet_continue,
            max_bars=max_bars
        )
        if s is not None:
            acumulate.append(s)
            print(s['ticker'], s['sum'], time.ctime())
    final_df = pd.concat(acumulate)
    return final_df


def main():
    pass


if __name__ == '__main__':
    main()
