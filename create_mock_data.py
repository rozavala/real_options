
import os
import random
import csv
from datetime import datetime, timedelta

def create_mock_council_history():
    """Create a mock council_history.csv for frontend verification."""
    ticker = os.environ.get("COMMODITY_TICKER", "KC")
    data_dir = os.path.join("data", ticker)
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "council_history.csv")

    if os.path.exists(file_path):
        print(f"Mock data already exists at {file_path}")
        return

    print(f"Creating mock council history at {file_path}")

    headers = [
        "timestamp", "contract", "master_decision", "master_confidence",
        "master_reasoning", "compliance_approved", "weighted_score",
        "vote_breakdown", "trigger_type", "primary_catalyst",
        "conviction_multiplier", "thesis_strength", "dissent_acknowledged",
        "meteorologist_sentiment", "macro_sentiment", "geopolitical_sentiment",
        "fundamentalist_sentiment", "sentiment_sentiment", "technical_sentiment",
        "volatility_sentiment"
    ]

    rows = []
    now = datetime.now()

    # Generate 10 sample rows
    for i in range(10):
        ts = (now - timedelta(hours=i*4)).isoformat()
        decision = random.choice(["BULLISH", "BEARISH", "NEUTRAL"])

        row = {
            "timestamp": ts,
            "contract": f"KC{chr(65+i)}25",
            "master_decision": decision,
            "master_confidence": f"{random.uniform(0.6, 0.9):.2f}",
            "master_reasoning": "Mock reasoning for frontend verification of XSS fix.",
            "compliance_approved": "True",
            "weighted_score": f"{random.uniform(-0.8, 0.8):.2f}",
            "vote_breakdown": '[{"agent": "Macro", "direction": "BULLISH", "contribution": 0.2}]',
            "trigger_type": random.choice(["scheduled", "emergency", "PriceSentinel"]),
            "primary_catalyst": "Mock Catalyst Event <script>alert('XSS')</script>",
            "conviction_multiplier": "0.85",
            "thesis_strength": random.choice(["PROVEN", "PLAUSIBLE", "SPECULATIVE"]),
            "dissent_acknowledged": "Minor dissent from Macro agent regarding yield curve.",
            "meteorologist_sentiment": "BULLISH",
            "macro_sentiment": "NEUTRAL",
            "geopolitical_sentiment": "BEARISH",
            "fundamentalist_sentiment": "BULLISH",
            "sentiment_sentiment": "NEUTRAL",
            "technical_sentiment": "BULLISH",
            "volatility_sentiment": "LOW"
        }
        rows.append(row)

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    create_mock_council_history()
