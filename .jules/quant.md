# QUANT'S JOURNAL - CRITICAL LEARNINGS

## Iron Condor Risk Calculation Flaw (2025-02-10)
- **Scenario**: Iron Condor (4 legs: Long Put, Short Put, Short Call, Long Call).
- **Issue**: The system calculated "Max Loss" using the full spread width (`Max Strike - Min Strike`), which massively overestimates the structural width of the trade (e.g., 30 points vs. true wing width of 10 points).
- **Impact**: This inflated the denominator in the `Risk Used %` calculation (`Unrealized PnL / Max Loss`), causing the risk percentage to appear artificially low (e.g., 14% instead of 50%).
- **Consequence**: Stop-loss triggers based on `Risk Used %` failed to fire, leaving positions open beyond defined risk limits.
- **Fix**: Implemented structure detection for Iron Condors to calculate Max Loss based on the wider of the two wings (`max(Put Wing, Call Wing) - Credit`).
