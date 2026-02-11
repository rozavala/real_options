import sys
import os
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Add repo root to path
sys.path.append(os.getcwd())

from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole, ModelProvider
from trading_bot.router_metrics import RouterMetrics

async def test_metrics_crash():
    print("Starting reproduction test...")

    # Mock config
    config = {
        'model_registry': {},
        'gemini': {'api_key': 'fake'},
        'openai': {'api_key': 'fake'},
        'anthropic': {'api_key': 'fake'},
        'xai': {'api_key': 'fake'},
    }

    # Prevent actual file writing
    with patch.object(RouterMetrics, '_save_to_disk', return_value=None):
        router = HeterogeneousRouter(config)

        # Mock acquire_api_slot to return True
        with patch('trading_bot.heterogeneous_router.acquire_api_slot', new_callable=AsyncMock) as mock_acquire:
            mock_acquire.return_value = True

            # Mock the client generation to succeed
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=("Success response", 100, 50))

            router._get_client = MagicMock(return_value=mock_client)

            try:
                print("Attempting to route (should crash)...")
                await router.route(AgentRole.WEATHER_SENTINEL, "test prompt")
                print("FAILURE: Route succeeded unexpectedly (expected AttributeError)")
            except AttributeError as e:
                print(f"Caught expected error: {e}")
                if "'RouterMetrics' object has no attribute 'record_success'" in str(e):
                    print("SUCCESS: Reproduced expected crash")
                elif "'RouterMetrics' object has no attribute 'record_failure'" in str(e):
                    print("SUCCESS: Reproduced expected crash (failure case)")
                else:
                    print(f"FAILURE: Crashed with unexpected AttributeError: {e}")
            except Exception as e:
                print(f"FAILURE: Crashed with unexpected error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_metrics_crash())
