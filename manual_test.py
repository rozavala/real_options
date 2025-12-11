import asyncio
import logging
from config_loader import load_config
from trading_bot.logging_config import setup_logging
from trading_bot.order_manager import generate_and_queue_orders, generate_and_execute_orders

# Initialize logging so you can see the output in your console/file
setup_logging()
logger = logging.getLogger("ManualTest")

async def main():
    logger.info("--- STARTING MANUAL TEST ---")
    config = load_config()
    
    # --- MODE SELECTION ---
    
    # OPTION 1: Verify Fixes Only (Recommended)
    # This runs the Data Pull -> Model -> Signal Generation.
    # It calculates the prices and signals but STOPS before placing trades.
    # Use this to verify the "Price" and "Confidence" bugs are gone.
    logger.info("Running generation logic to verify metrics...")
    await generate_and_queue_orders(config)

    # OPTION 2: Full Execution (Use only when Option 1 is verified)
    # Uncomment the line below to actually PLACE orders to IB.
    # await generate_and_execute_orders(config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Manual test stopped by user.")
