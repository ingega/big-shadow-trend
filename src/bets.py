import pandas as pd

from src.functions import Bars, Filter


class Bet:
    """
    This class takes a filter and then iterates it to get
    the outcome of a bet.
    """
    def __init__(self, list_min: list, minutes: int, bet: float, **kwargs):
        self.bet = bet
        self.minutes = minutes
        self.params = kwargs
        self.list_min = list_min

    def prepare_bet(self, filtered_record):
        """
        Evaluates a single bet outcome.

        Returns:
            dict: Outcome details if bet is resolved, else None.
        """
        pointer = filtered_record['index'] * self.minutes
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
                'outcome': result['outcome'],
                'way': 'direct',
            }
        return outcome

    def check_bet(self, side, pointer, winner_price, losser_price):
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
                    'way': 'direct',
                }
                break
            elif ((side == "BUY" and the_list[p]['low'] <= losser_price) or
                    (side == "SELL" and the_list[p]['high'] >= losser_price)):
                outcome = {
                    'date_out': the_list[p]['date'],
                    'price_out': losser_price,
                    'pointer': p,
                    'outcome': 'sl',  # stop loss
                    'way': 'direct',
                }
                break
        return outcome


def single_bet(ticker, list_min, minutes, days, **kwargs):
    """
    Executes a series of bets based on the given filters.
    Returns:
        df: df with bet outcomes.
    """
    bars = Bars(ticker)
    shadow = kwargs.get('shadow')
    body = kwargs.get('body')
    bet_value = kwargs.get('bet')

    f = Filter(bars, minutes=minutes, days=days, shadow=shadow, body=body)
    filtered = f.apply_filter()
    bet = Bet(list_min=list_min, minutes=minutes, bet=bet_value)

    final = []
    for record in filtered:
        outcome = bet.prepare_bet(record)
        if outcome:
            if outcome['outcome'] == 'tp':
                outcome['profit'] = bet_value - 0.0008  # commission
            else:
                outcome['profit'] = -bet_value - 0.0008  # commission
            final.append(outcome)
    # return into a pd
    df = pd.DataFrame(final)
    return df


def main():
    pass


if '__name__' == '__main__':
    main()
