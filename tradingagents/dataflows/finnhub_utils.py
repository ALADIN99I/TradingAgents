import json
import os
from pathlib import Path
import logging


def get_data_in_range(ticker, start_date, end_date, data_type, data_dir, period=None):
    """
    Gets finnhub data saved and processed on disk.
    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        data_type (str): Type of data from finnhub to fetch. Can be insider_trans, SEC_filings, news_data, insider_senti, or fin_as_reported.
        data_dir (str): Directory where the data is saved.
        period (str): Default to none, if there is a period specified, should be annual or quarterly.
    """

    if period:
        data_path = os.path.join(
            data_dir,
            "finnhub_data",
            data_type,
            f"{ticker}_{period}_data_formatted.json",
        )
    else:
        data_path = os.path.join(
            data_dir, "finnhub_data", data_type, f"{ticker}_data_formatted.json"
        )

    try:
        sanitized_path = Path(str(data_path).strip())
        resolved_path = sanitized_path.resolve(strict=True)
        with open(resolved_path, "r", encoding="utf-8") as f:
            data_content = json.load(f)
    except FileNotFoundError:
        logging.error(f"Finnhub data file not found for {ticker} (type: {data_type}) at: {sanitized_path}")
        return {} # Return empty dict if file not found
    except Exception as e:
        logging.error(f"Error loading Finnhub data file '{data_path}' for {ticker} (type: {data_type}): {e}")
        return {} # Return empty dict on other errors

    # filter keys (date, str in format YYYY-MM-DD) by the date range (str, str in format YYYY-MM-DD)
    # Use data_content instead of data for filtering
    filtered_data = {}
    for key, value in data_content.items():
        if start_date <= key <= end_date and len(value) > 0:
            filtered_data[key] = value
    return filtered_data
