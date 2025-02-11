from functions import Bars
import unittest
from unittest.mock import patch
import pandas as pd


class TestBars(unittest.TestCase):

    @patch('pandas.read_pickle')
    def test_get_bars(self, mock_read_pickle):
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
        bars = Bars(ticker='BTC',)

        # Run getbars method
        result = bars.get_bars(minutes=1400, days=3)

        # Assertions to check the correctness
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # Check if slicing works correctly
        self.assertEqual(result.iloc[0]['open'], 102)  # Check data conversion
        self.assertEqual(result.iloc[-1]['volume'], 1400)  # check if charges values correctly

    @patch('pandas.read_pickle', side_effect=FileNotFoundError("File not found"))
    def test_get_bars_invalid_ticker(self, mock_read_pickle):
        """Test get_bars() when an invalid ticker is passed"""
        bars = Bars(ticker='INVALID_TICKER')
        with self.assertRaises(ValueError) as context:
            bars.get_bars(minutes=1440, days=3)

        # Ensure correct error message is raised
        self.assertIn("File not found", str(context.exception))

    @patch('pandas.read_pickle', side_effect=OSError("OS error"))
    def test_get_bars_os_error(self, mock_read_pickle):
        """Test get_bars() when an OS error occurs"""
        bars = Bars(ticker='BTC')
        with self.assertRaises(ValueError) as context:
            bars.get_bars(minutes=1440, days=3)

        self.assertIn("OS error", str(context.exception))

    @patch('pandas.read_pickle', side_effect=Exception("Unknown error"))
    def test_get_bars_generic_error(self, mock_read_pickle):
        """Test get_bars() when an unexpected error occurs"""
        bars = Bars(ticker='BTCUSDT')
        with self.assertRaises(ValueError) as context:
            bars.get_bars(minutes=1440, days=3)

        self.assertIn("Unknown error", str(context.exception))


if __name__ == '__main__':
    unittest.main()
