# from src.bets import Bet, single_bet, all_tickers
import unittest


class TestBet:
    """
    in this testing class must be provided, mocked cases for
    Bet:
        check_bet, prepare_bet, protect
    and single_bet and all_tickers
    """
    def test_single_bet(self):
        """mock cases here"""
        assert 1 + 1 == 2

    def test_all_tickers(self):
        """mock cases here"""
        assert 1 * 1 == 1


if __name__ == '__main__':
    unittest.main()
