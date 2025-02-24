from src.functions import Bars, Filter
import unittest
from unittest.mock import patch
import pandas as pd
# Path is used in decorator patch, but not is visible
from pathlib import Path  # noqa: F401


class TestBars(unittest.TestCase):

    @patch('pathlib.Path.exists', return_value=True)  # Mock directory check
    @patch('pandas.read_pickle')
    def test_get_bars(self, mock_read_pickle, mock_path_exists):
        sample_data = {
            'date': pd.date_range(start='2025-01-01', periods=5, freq='D'),
            'open': ['100', '101', '102', '103', '104'],
            'high': ['105', '106', '107', '108', '109'],
            'low': ['95', '96', '97', '98', '99'],
            'close': ['102', '103', '104', '105', '106'],
            'volume': ['1000', '1100', '1200', '1300', '1400'],
        }
        mock_df = pd.DataFrame(sample_data)
        mock_read_pickle.return_value = mock_df

        # Create Bars instance
        bars = Bars(ticker='BTC')

        # Run getbars method
        result = bars.get_bars(minutes=1400, days=3)

        # Assertions to check the correctness
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # Check slicing
        self.assertEqual(result.iloc[0]['open'], 102)  # Check conversion
        # Check correct data load
        self.assertEqual(result.iloc[-1]['volume'], 1400)

    @patch('pathlib.Path.exists', return_value=True)
    @patch('pandas.read_pickle',
           side_effect=FileNotFoundError("File not found"))
    def test_get_bars_invalid_ticker(self, mock_read_pickle, mock_path_exists):
        """Test get_bars() when an invalid ticker is passed"""
        bars = Bars(ticker='INVALID_TICKER')
        with self.assertRaises(ValueError) as context:
            bars.get_bars(minutes=1440, days=3)

        self.assertIn("File not found", str(context.exception))

    @patch('pathlib.Path.exists', return_value=True)
    @patch('pandas.read_pickle', side_effect=OSError("OS error"))
    def test_get_bars_os_error(self, mock_read_pickle, mock_path_exists):
        """Test get_bars() when an OS error occurs"""
        bars = Bars(ticker='BTC')
        with self.assertRaises(ValueError) as context:
            bars.get_bars(minutes=1440, days=3)

        self.assertIn("OS error", str(context.exception))

    @patch('pathlib.Path.exists', return_value=True)
    @patch('pandas.read_pickle', side_effect=Exception("Unknown error"))
    def test_get_bars_generic_error(self, mock_read_pickle, mock_path_exists):
        """Test get_bars() when an unexpected error occurs"""
        bars = Bars(ticker='BTCUSDT')
        with self.assertRaises(ValueError) as context:
            bars.get_bars(minutes=1440, days=3)

        self.assertIn("Unknown error", str(context.exception))


class TestFilter(unittest.TestCase):
    @patch.object(Bars, "get_bars")
    def test_apply_filter_valid_entry(self, mock_get_bars):
        """
        Test apply_filter() when a valid entry condition exists
        :return
            1 case in uptrend, and 1 in downtrend
        """
        # ✅ Mock DataFrame with two valid trade signal
        sample_data = pd.DataFrame({
            'open': [100, 102, 101, 100],  # body_size = 0.004
            'high': [110, 108, 101.4, 99],  # shadow_up = 0.1
            'low': [95, 99, 97, 96],  # shadow_down = 0.04
            'close': [100.4, 106, 100.6, 99.5],  # Both bars same color
        })
        mock_get_bars.return_value = sample_data

        # ✅ Create Bars instance (Mocked)
        bars = Bars(ticker="BTCUSDT")

        # ✅ Initialize Filter with sample conditions
        filter_obj = Filter(bars, minutes=1, days=1, shadow=0.02, body=0.01)

        # ✅ Apply the filter
        result = filter_obj.apply_filter()

        # ✅ Assertions: Should return 1 row
        self.assertEqual(len(result), 2)

    @patch.object(Bars, "get_bars")
    def test_apply_filter_empty_entry(self, mock_get_bars):
        """
        Test apply_filter() when a valid entry condition exists
        :return
            1 case in uptrend, and 1 in downtrend
        """
        # ✅ Mock DataFrame with two valid trade signal
        sample_data = pd.DataFrame({
            'open': [100, 102, 101, 100],  # body_size = 0.05
            'high': [110, 108, 101.4, 99],  # shadow_up = 0.1
            'low': [95, 99, 97, 96],  # shadow_down = 0.04
            'close': [105, 106, 96, 99.5],  # Both bars same color
        })
        mock_get_bars.return_value = sample_data

        # ✅ Create Bars instance (Mocked)
        bars = Bars(ticker="BTCUSDT")

        # ✅ Initialize Filter with sample conditions
        filter_obj = Filter(bars, minutes=1, days=1, shadow=0.02, body=0.01)

        # ✅ Apply the filter
        result = filter_obj.apply_filter()

        # ✅ Assertions: Should return 1 row
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
