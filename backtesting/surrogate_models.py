"""
Level 2 Backtest: Surrogate Model Training and Inference

This module trains lightweight models to mimic the Council's decisions,
enabling rapid hyperparameter optimization without LLM costs.

Speed: ~30 minutes for 5 years (including training)
Cost: $0 per backtest run (after initial training)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SurrogateConfig:
    """Configuration for surrogate model training."""
    model_dir: str = "./data/KC/surrogate_models"
    min_training_samples: int = 100
    test_size: float = 0.2
    random_state: int = 42

    # GradientBoosting hyperparameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1


class DecisionSurrogate:
    """
    XGBoost-based surrogate for Council decisions.

    Input features:
    - Price trend (SMA crossovers)
    - Volatility (ATR, IV rank)
    - Sentiment score (aggregated)
    - Weather risk score

    Output:
    - Direction: BULLISH, BEARISH, NEUTRAL
    - Confidence: 0.0-1.0
    """

    FEATURE_COLUMNS = [
        'price_trend_5d',      # 5-day price change %
        'price_trend_20d',     # 20-day price change %
        'sma_cross',           # 1 if SMA5 > SMA20, else 0
        'atr_pct',             # ATR as % of price
        'iv_rank',             # IV percentile (0-1)
        'rsi_14',              # RSI (0-100)
        'sentiment_score',     # Aggregated sentiment (-1 to 1)
        'weather_risk',        # Weather risk score (0-1)
        'inventory_trend',     # Inventory change direction (-1, 0, 1)
        'day_of_week',         # 0-4 (Mon-Fri)
    ]

    DIRECTION_MAP = {
        'BULLISH': 0,
        'NEUTRAL': 1,
        'BEARISH': 2
    }

    DIRECTION_REVERSE = {v: k for k, v in DIRECTION_MAP.items()}

    def __init__(self, config: SurrogateConfig = None):
        self.config = config or SurrogateConfig()
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self._is_trained = False

        # Ensure model directory exists
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

    def train(self, council_history: pd.DataFrame) -> Dict:
        """
        Train surrogate model on Council decision history.

        Args:
            council_history: DataFrame with columns matching FEATURE_COLUMNS
                            plus 'master_decision' and 'master_confidence'

        Returns:
            Dict with training metrics
        """
        logger.info("Training decision surrogate model...")

        # Validate data
        if len(council_history) < self.config.min_training_samples:
            raise ValueError(
                f"Need at least {self.config.min_training_samples} samples, "
                f"got {len(council_history)}"
            )

        # Prepare features
        X = self._prepare_features(council_history)
        y = council_history['master_decision'].map(self.DIRECTION_MAP)

        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(1)  # Default to NEUTRAL

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)

        self._is_trained = True

        metrics = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(
                self.FEATURE_COLUMNS,
                self.model.feature_importances_
            ))
        }

        logger.info(f"Surrogate trained: accuracy={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

        return metrics

    def predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict Council decision using surrogate.

        Args:
            features: DataFrame with FEATURE_COLUMNS

        Returns:
            Tuple of (direction, confidence)
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self._prepare_features(features)
        X_scaled = self.scaler.transform(X.fillna(0))

        # Get prediction and probability
        direction_idx = self.model.predict(X_scaled)[0]
        probas = self.model.predict_proba(X_scaled)[0]
        confidence = probas[direction_idx]

        direction = self.DIRECTION_REVERSE[direction_idx]

        return direction, float(confidence)

    def save(self, name: str = "council_surrogate") -> str:
        """Save model to disk."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        model_path = Path(self.config.model_dir) / f"{name}.joblib"
        scaler_path = Path(self.config.model_dir) / f"{name}_scaler.joblib"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"Surrogate saved to {model_path}")
        return str(model_path)

    def load(self, name: str = "council_surrogate") -> None:
        """Load model from disk."""
        model_path = Path(self.config.model_dir) / f"{name}.joblib"
        scaler_path = Path(self.config.model_dir) / f"{name}_scaler.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self._is_trained = True

        logger.info(f"Surrogate loaded from {model_path}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from dataframe."""
        available_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]

        if not available_cols:
            raise ValueError(
                f"No feature columns found. Expected: {self.FEATURE_COLUMNS}"
            )

        return df[available_cols].copy()


def prepare_surrogate_features(
    price_data: pd.DataFrame,
    sentiment_data: Optional[pd.DataFrame] = None,
    weather_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Utility to prepare features for surrogate model.

    Args:
        price_data: OHLCV data
        sentiment_data: Optional sentiment scores by date
        weather_data: Optional weather risk scores by date

    Returns:
        DataFrame ready for surrogate training/prediction
    """
    df = price_data.copy()

    # Price features
    df['price_trend_5d'] = df['close'].pct_change(5)
    df['price_trend_20d'] = df['close'].pct_change(20)
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_cross'] = (df['sma_5'] > df['sma_20']).astype(int)

    # Volatility features
    df['atr'] = _calc_atr(df, period=14)
    df['atr_pct'] = df['atr'] / df['close']

    # RSI
    df['rsi_14'] = _calc_rsi(df['close'], period=14)

    # IV rank placeholder (would need options data)
    df['iv_rank'] = 0.5  # Default

    # Merge sentiment if provided
    if sentiment_data is not None and 'sentiment_score' in sentiment_data.columns:
        df = df.join(sentiment_data[['sentiment_score']], how='left')
    else:
        df['sentiment_score'] = 0.0

    # Merge weather if provided
    if weather_data is not None and 'weather_risk' in weather_data.columns:
        df = df.join(weather_data[['weather_risk']], how='left')
    else:
        df['weather_risk'] = 0.0

    # Inventory placeholder
    df['inventory_trend'] = 0

    # Day of week
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
    else:
        df['day_of_week'] = 0

    return df


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr


def _calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index with zero-division guard.

    FIX (MECE V2 #5): Handle division by zero when loss=0 (all gains).
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    # Guard against division by zero (all gains, no losses)
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases: fill NaN with neutral RSI
    rsi = rsi.fillna(50)

    return rsi
