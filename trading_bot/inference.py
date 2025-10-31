import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import json
import logging
import warnings
import os
from scipy.stats import linregress
from arch import arch_model
import pandas_ta as ta

# --- 1. Suppress Warnings & Configure Logging ---

# Suppress noisy warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow GPU/CPU logs
tf.get_logger().setLevel('ERROR')

# Configure logging to match your desired output format
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(message)s',
datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 2. Define Constants ---
TIME_STEPS = 60 # Our 60-day lookback
PREDICTION_HORIZON_DAYS = 5 # T+5
NUM_CONTRACTS = 5
MIN_HISTORY_DAYS = 100 # Min days needed in input CSV to run calculations

# Define all 10 production assets we need to load
PRODUCTION_ASSET_FILES = {
# Scalers (3)
"x_scaler_seq": "models/x_scaler_sequential.joblib",
"x_scaler_agg": "models/x_scaler_aggregated.joblib",
"y_scaler": "models/y_scaler_t5.joblib",

# Models (7)
"transformer": "models/transformer_model.keras",
"xgb_0": "models/xgboost_tuned_component_0.ubj",
"xgb_1": "models/xgboost_tuned_component_1.ubj",
"xgb_2": "models/xgboost_tuned_component_2.ubj",
"xgb_3": "models/xgboost_tuned_component_3.ubj",
"xgb_4": "models/xgboost_tuned_component_4.ubj",
"blender": "models/hybrid_blender.joblib"
}

# --- 3. Helper Functions for Feature Engineering ---

def get_garch_features(log_returns):
    """
    Fits a GARCH(1,1) model to the log returns.
    Returns a DataFrame with 'conditional_vol' and 'std_residuals'.
    """
    returns_cleaned = log_returns.dropna() * 100
    if returns_cleaned.empty:
        return pd.DataFrame(columns=['conditional_vol', 'std_residuals'])
    try:
        garch = arch_model(returns_cleaned, vol='Garch', p=1, o=0, q=1, dist='t')
        garch_fit = garch.fit(update_freq=0, disp='off')

        features_df = pd.DataFrame({
            'conditional_vol': garch_fit.conditional_volatility / 100,
            'std_residuals': garch_fit.resid / garch_fit.conditional_volatility
        }, index=returns_cleaned.index)
        return features_df
    except Exception as e:
        logging.warning(f"GARCH fitting failed: {e}. Returning NaNs.")
        return pd.DataFrame(columns=['conditional_vol', 'std_residuals'], index=log_returns.index)

def get_slope(array):
    """
    Calculates the linear regression slope of a 1D array.
    """
    y = array
    x = np.arange(len(y))
    try:
        slope = linregress(x, y)[0]
    except ValueError:
        slope = 0.0 # Handle cases with NaNs
    return slope

# --- 4. Core Pipeline Functions ---

def load_production_assets():
    """
    Loads all 10 model and scaler files from disk.
    """
    logging.info("Loading all production assets...")
    assets = {}

    # Load Scalers
    assets['x_scaler_seq'] = joblib.load(PRODUCTION_ASSET_FILES['x_scaler_seq'])
    assets['x_scaler_agg'] = joblib.load(PRODUCTION_ASSET_FILES['x_scaler_agg'])
    assets['y_scaler'] = joblib.load(PRODUCTION_ASSET_FILES['y_scaler'])

    # Load Models
    assets['tf_model'] = tf.keras.models.load_model(PRODUCTION_ASSET_FILES['transformer'])

    xgb_models = []
    for i in range(NUM_CONTRACTS):
        model = xgb.XGBRegressor()
        model.load_model(PRODUCTION_ASSET_FILES[f'xgb_{i}'])
        xgb_models.append(model)
    assets['xgb_models'] = xgb_models

    assets['blender_model'] = joblib.load(PRODUCTION_ASSET_FILES['blender'])

    logging.info("All 10 assets loaded successfully.")
    return assets

def generate_inference_features(raw_df, assets):
    """
    Takes the raw data CSV and performs all feature engineering
    (GARCH, TA, Spreads, Aggregation) to create the
    final scaled inputs for the LSTM and XGBoost models.
    """
    logging.info("Generating features from new data...")
    df = raw_df.copy()

    # --- 4a. Create derived features (GARCH, TA, Spreads) ---

    # 1. GARCH Features
    df['main_log_return'] = np.log(df['front_month_price'] / df['front_month_price'].shift(1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    garch_features = get_garch_features(df['main_log_return'])
    df = df.join(garch_features)

    # 2. TA Features
    df['ta_rsi_14d'] = ta.rsi(df['front_month_price'], length=14)
    df.ta.macd(close=df['front_month_price'], length=12, slow=26, signal=9, append=True)

    # 3. Spread Features
    df['spread_1_2'] = df['front_month_price'] - df['second_month_price']
    df['spread_2_3'] = df['second_month_price'] - df['third_month_price']
    df['spread_3_4'] = df['third_month_price'] - df['fourth_month_price']
    df['spread_4_5'] = df['fourth_month_price'] - df['fifth_month_price']

    # This is our full 30-feature set for the Transformer.
    # The order MUST match the columns from the training data EXACTLY.
    all_base_features = [
        'conditional_vol', 'std_residuals', 'ta_rsi_14d', 'MACD_12_26_9',
        'MACDh_12_26_9', 'MACDs_12_26_9', 'spread_1_2', 'spread_2_3',
        'spread_3_4', 'spread_4_5', 'brazil_minas_gerais_avg_temp',
        'brazil_minas_gerais_precipitation', 'vietnam_ho_chi_minh_avg_temp',
        'vietnam_ho_chi_minh_precipitation', 'colombia_antioquia_avg_temp',
        'colombia_antioquia_precipitation', 'indonesia_sumatra_avg_temp',
        'indonesia_sumatra_precipitation', 'cot_noncomm_net',
        'brl_usd_exchange_rate', 'oil_price_wti', 'indonesia_idr_usd',
        'mexico_mxn_usd', 'sugar_price', 'sp500_price', 'nestle_stock',
        'us_dollar_index', 'shipping_proxy', 'volatility_index', 'dgs10_yield'
    ]
    df_base_features = df[all_base_features]

    # --- 4b. Create Aggregated Features (for XGBoost) ---
    df_agg = pd.DataFrame(index=df.index)
    for col in all_base_features:
        df_agg[f'{col}_mean_{TIME_STEPS}d'] = df[col].rolling(window=TIME_STEPS).mean()
        df_agg[f'{col}_std_{TIME_STEPS}d'] = df[col].rolling(window=TIME_STEPS).std()
        df_agg[f'{col}_slope_{TIME_STEPS}d'] = df[col].rolling(window=TIME_STEPS).apply(get_slope, raw=True)
        df_agg[f'{col}_last'] = df[col]

    # --- 4c. Get LAST row and Scale ---

    # Get LSTM/Transformer input
    last_seq_data = df_base_features.iloc[-TIME_STEPS:] # Get last 60 rows
    scaled_seq_data = assets['x_scaler_seq'].transform(last_seq_data)
    lstm_input = scaled_seq_data.reshape(1, TIME_STEPS, len(all_base_features)) # (1, 60, 30)

    # Get XGBoost input
    last_agg_data = df_agg.iloc[-1:] # Get last 1 row
    scaled_agg_data = assets['x_scaler_agg'].transform(last_agg_data) # (1, 120)

    logging.info("Feature generation complete.")
    return lstm_input, scaled_agg_data


def get_hybrid_prediction(lstm_input, xgb_input, assets):
    """
    Runs the full hybrid prediction pipeline.
    """
    logging.info("Running hybrid prediction...")

    # 1. Get Transformer Prediction
    preds_tf_scaled = assets['tf_model'].predict(lstm_input)

    # 2. Get XGBoost Predictions
    xgb_preds_list = []
    for model in assets['xgb_models']:
        pred = model.predict(xgb_input)
        xgb_preds_list.append(pred[0])
    preds_xgb_scaled = np.array(xgb_preds_list).reshape(1, NUM_CONTRACTS)

    # 3. Blend Predictions
    blender_input = np.hstack([preds_tf_scaled, preds_xgb_scaled])
    final_preds_scaled = assets['blender_model'].predict(blender_input)

    # 4. Inverse-Transform to get log-returns
    final_preds_unscaled = assets['y_scaler'].inverse_transform(final_preds_scaled)

    return final_preds_unscaled[0] # Return the 1D array of 5 predictions


# --- 5. Main Execution ---

def get_model_predictions(raw_df: pd.DataFrame):
    """
    Main function to run the entire pipeline.
    """
    try:
        # 1. Load all models and scalers
        assets = load_production_assets()

        # 2. Load new raw data
        if len(raw_df) < MIN_HISTORY_DAYS:
            logging.error(f"Input data has only {len(raw_df)} rows. Need at least {MIN_HISTORY_DAYS} to calculate features.")
            return

        # 3. Generate features for the latest day
        # This will use the last 60-100 days of data
        lstm_input, xgb_input = generate_inference_features(raw_df, assets)

        # 4. Get final blended prediction
        price_changes = get_hybrid_prediction(lstm_input, xgb_input, assets)

        # 5. Format and log the output
        output_dict = {
            "price_changes": price_changes.tolist()
        }

        # Log the final JSON-formatted string
        logging.info(json.dumps(output_dict, indent=2))
        return output_dict

    except FileNotFoundError as e:
        logging.error(f"FATAL ERROR: A required asset file was not found: {e.filename}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    return None
