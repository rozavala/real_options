import os
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

SIGNALS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_signals.csv'))

def log_model_signal(contract: str, signal: str):
    """
    Logs a model signal to the model_signals.csv file.

    Args:
        contract (str): The contract month (e.g., 'KOH6').
        signal (str): The signal ('BULL', 'BEAR', 'NEUTRAL').
    """
    try:
        new_signal = pd.DataFrame({
            'timestamp': [datetime.now()],
            'contract': [contract],
            'signal': [signal]
        })

        if not os.path.exists(SIGNALS_FILE_PATH):
            logger.info(f"'{SIGNALS_FILE_PATH}' not found, creating new file.")
            new_signal.to_csv(SIGNALS_FILE_PATH, index=False)
        else:
            new_signal.to_csv(SIGNALS_FILE_PATH, mode='a', header=False, index=False)

        logger.info(f"Successfully logged signal: Contract={contract}, Signal={signal}")

    except Exception as e:
        logger.error(f"Failed to log model signal: {e}", exc_info=True)

def get_model_signals_df():
    """
    Reads the model_signals.csv file into a pandas DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the model signals, or an empty
        DataFrame if the file does not exist.
    """
    if not os.path.exists(SIGNALS_FILE_PATH):
        logger.warning(f"'{SIGNALS_FILE_PATH}' not found.")
        return pd.DataFrame()

    df = pd.read_csv(SIGNALS_FILE_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
