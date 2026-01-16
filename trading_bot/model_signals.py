import os
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

SIGNALS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_signals.csv'))

def log_model_signal(contract: str, signal: str, price: float = None, sma_200: float = None, expected_price: float = None, confidence: float = None):
    """
    Logs a model signal to the model_signals.csv file.

    Args:
        contract (str): The contract month (e.g., 'KOH6').
        signal (str): The signal ('BULL', 'BEAR', 'NEUTRAL').
        price (float, optional): The current price of the contract.
        sma_200 (float, optional): The 200-day SMA.
        expected_price (float, optional): The expected price from the prediction.
        confidence (float, optional): The confidence of the prediction.
    """
    try:
        # Clamp confidence if present
        if confidence is not None:
             confidence = max(0.0, min(1.0, float(confidence)))

        new_signal = pd.DataFrame({
            'timestamp': [datetime.now()],
            'contract': [contract],
            'signal': [signal],
            'price': [price],
            'sma_200': [sma_200],
            'expected_price': [expected_price],
            'confidence': [confidence]
        })

        if not os.path.exists(SIGNALS_FILE_PATH):
            logger.info(f"'{SIGNALS_FILE_PATH}' not found, creating new file.")
            new_signal.to_csv(SIGNALS_FILE_PATH, index=False)
        else:
            try:
                # Check for schema consistency by reading the header
                existing_df = pd.read_csv(SIGNALS_FILE_PATH, nrows=0)
                if list(existing_df.columns) != list(new_signal.columns):
                    logger.info("Schema changed. Migrating existing log file to new schema.")
                    # Read the full file to migrate
                    full_existing_df = pd.read_csv(SIGNALS_FILE_PATH)

                    # Add missing columns
                    for col in new_signal.columns:
                        if col not in full_existing_df.columns:
                            full_existing_df[col] = None

                    # Reorder columns to match new schema
                    full_existing_df = full_existing_df[new_signal.columns]

                    # Write back the migrated data + the new signal
                    combined_df = pd.concat([full_existing_df, new_signal], ignore_index=True)
                    combined_df.to_csv(SIGNALS_FILE_PATH, index=False)
                else:
                    new_signal.to_csv(SIGNALS_FILE_PATH, mode='a', header=False, index=False)
            except pd.errors.EmptyDataError:
                # File exists but is empty
                new_signal.to_csv(SIGNALS_FILE_PATH, index=False)

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
