import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'state.json')

class StateManager:
    """
    Manages the persistence of agent reports to ensure the 'Hybrid Decision Model'
    can access the 'Last Known Opinion' of sleeping agents during triggered cycles.
    """

    @staticmethod
    def save_state(reports: dict):
        """
        Saves the provided agent reports to the state file.

        Args:
            reports (dict): A dictionary where keys are agent names and values are their report strings.
        """
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

            data = {
                "timestamp": datetime.now().isoformat(),
                "reports": reports
            }

            with open(STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info("Agent state successfully saved.")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")

    @staticmethod
    def load_state() -> dict:
        """
        Loads the last known agent reports from the state file.

        Returns:
            dict: A dictionary of agent reports. Returns an empty dict if no state exists.
        """
        if not os.path.exists(STATE_FILE):
            logger.warning("No previous state file found. Returning empty state.")
            return {}

        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)

            # Basic validation
            if 'reports' in data:
                logger.info(f"Agent state loaded (Timestamp: {data.get('timestamp')}).")
                return data['reports']
            else:
                logger.warning("State file found but missing 'reports' key.")
                return {}

        except json.JSONDecodeError:
            logger.error("State file corrupted. Returning empty state.")
            return {}
        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            return {}
