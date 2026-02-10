import pytest
from trading_bot.strategy_router import (
    calculate_agent_conflict,
    detect_imminent_catalyst,
    route_strategy,
    infer_strategy_type
)

# === TEST DATA ===

SCHEDULED_AGENTS_AGREE = {
    'macro_sentiment': 'BULLISH',
    'technical_sentiment': 'BULLISH',
    'geopolitical_sentiment': 'BULLISH',
    'sentiment_sentiment': 'BULLISH',
    'agronomist_sentiment': 'BULLISH'
}

SCHEDULED_AGENTS_CONFLICT = {
    'macro_sentiment': 'BULLISH',
    'technical_sentiment': 'BEARISH',
    'geopolitical_sentiment': 'BULLISH',
    'sentiment_sentiment': 'BEARISH',
    'agronomist_sentiment': 'NEUTRAL'
}

EMERGENCY_AGENTS_AGREE = {
    'macro': {'data': 'I am deeply BULLISH'},
    'technical': {'data': 'Market looks BULLISH'},
    'news': 'BULLISH news everywhere'
}

EMERGENCY_AGENTS_CONFLICT = {
    'macro': {'data': 'I am deeply BULLISH'},
    'technical': {'data': 'Market looks BEARISH'},
    'news': 'Mixed signals'
}

# === CONFLICT TESTS ===

def test_conflict_scheduled():
    score_low = calculate_agent_conflict(SCHEDULED_AGENTS_AGREE, mode="scheduled")
    assert score_low == 0.0

    score_high = calculate_agent_conflict(SCHEDULED_AGENTS_CONFLICT, mode="scheduled")
    assert score_high > 0.5

def test_conflict_emergency():
    score_low = calculate_agent_conflict(EMERGENCY_AGENTS_AGREE, mode="emergency")
    assert score_low == 0.0

    score_high = calculate_agent_conflict(EMERGENCY_AGENTS_CONFLICT, mode="emergency")
    assert score_high > 0.5

# === CATALYST TESTS ===

def test_catalyst_scheduled():
    # Compound keyword
    data = {'agronomist_summary': 'Active frost warning in Minas Gerais'}
    cat = detect_imminent_catalyst(data, mode="scheduled")
    assert "frost warning" in cat

    # Single keyword without urgency
    data_weak = {'agronomist_summary': 'Some drought concerns but nothing major'}
    cat_weak = detect_imminent_catalyst(data_weak, mode="scheduled")
    assert cat_weak is None

    # Single keyword WITH urgency
    data_urgent = {'agronomist_summary': 'Severe drought imminent'}
    cat_urgent = detect_imminent_catalyst(data_urgent, mode="scheduled")
    assert "Drought" in cat_urgent

    # Resolved
    data_resolved = {'agronomist_summary': 'The drought has ended and rains returned'}
    cat_resolved = detect_imminent_catalyst(data_resolved, mode="scheduled")
    assert cat_resolved is None

def test_catalyst_emergency():
    data = {'news': {'data': 'Breaking news: USDA report shows massive deficit'}}
    cat = detect_imminent_catalyst(data, mode="emergency")
    assert "USDA report" in cat

# === ROUTING TESTS ===

def test_route_iron_condor():
    # Neutral + Range Bound + Expensive Vol = Iron Condor
    routed = route_strategy(
        direction='NEUTRAL',
        confidence=0.5,
        vol_sentiment='BEARISH', # Expensive
        regime='RANGE_BOUND',
        thesis_strength='PLAUSIBLE',
        conviction_multiplier=1.0,
        reasoning='Test',
        agent_data=SCHEDULED_AGENTS_AGREE,
        mode="scheduled"
    )
    assert routed['prediction_type'] == 'VOLATILITY'
    assert routed['vol_level'] == 'LOW'
    assert infer_strategy_type(routed) == 'IRON_CONDOR'

def test_route_long_straddle():
    # Neutral + Catalyst + Cheap Vol = Long Straddle
    # Catalyst triggered by "frost warning"
    catalyst_data = {'agronomist_summary': 'Active frost warning'}

    routed = route_strategy(
        direction='NEUTRAL',
        confidence=0.5,
        vol_sentiment='BULLISH', # Cheap
        regime='RANGE_BOUND',
        thesis_strength='PLAUSIBLE',
        conviction_multiplier=1.0,
        reasoning='Test',
        agent_data=catalyst_data,
        mode="scheduled"
    )
    assert routed['prediction_type'] == 'VOLATILITY'
    assert routed['vol_level'] == 'HIGH'
    assert infer_strategy_type(routed) == 'LONG_STRADDLE'

def test_route_directional():
    routed = route_strategy(
        direction='BULLISH',
        confidence=0.8,
        vol_sentiment='NEUTRAL',
        regime='TRENDING_UP',
        thesis_strength='PROVEN',
        conviction_multiplier=1.0,
        reasoning='Test',
        agent_data={},
        mode="scheduled"
    )
    assert routed['prediction_type'] == 'DIRECTIONAL'
    assert routed['direction'] == 'BULLISH'
    assert infer_strategy_type(routed) == 'DIRECTIONAL'

def test_route_no_trade():
    # Neutral + No Catalyst + No Conflict + Cheap Vol = No Trade
    routed = route_strategy(
        direction='NEUTRAL',
        confidence=0.5,
        vol_sentiment='BULLISH',
        regime='RANGE_BOUND',
        thesis_strength='SPECULATIVE',
        conviction_multiplier=1.0,
        reasoning='Test',
        agent_data=SCHEDULED_AGENTS_AGREE, # Agreeing agents = low conflict
        mode="scheduled"
    )
    assert routed['prediction_type'] == 'DIRECTIONAL'
    assert routed['direction'] == 'NEUTRAL'
    assert infer_strategy_type(routed) == 'NONE'
