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
        Saves the provided agent reports to the state file using atomic writes.
        Stores each report with a timestamp.
        Merges new reports with existing state.

        Args:
            reports (dict): A dictionary where keys are agent names (or special keys like 'latest_ml_signals')
                            and values are their contents.
        """
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

            # Load existing state first to merge
            current_state = {}
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, 'r') as f:
                        data = json.load(f)
                        if 'reports' in data:
                            current_state = data['reports']
                except Exception:
                    logger.warning("Could not load existing state for merge. Starting fresh.")

            current_time = datetime.now().isoformat()

            # Merge new reports into current state
            for key, content in reports.items():
                if key == "latest_ml_signals":
                    # Special handling for ML signals: Just store the list directly with timestamp wrapper?
                    # Or just as a raw object? Let's wrap it to match the schema 'data', 'timestamp'
                    # But content is a list of dicts.
                    # Let's verify if we need it serialized. json.dump handles list of dicts fine.
                    current_state[key] = {
                        "data": content, # List of dicts
                        "timestamp": current_time
                    }
                elif isinstance(content, dict) and 'data' in content:
                    # Already formatted (e.g., loading from previous state and passing back)
                    current_state[key] = content
                else:
                    # New or string content -> Stamp with NOW
                    current_state[key] = {
                        "data": str(content), # Ensure serialization for text reports
                        "timestamp": current_time
                    }

            final_data = {
                "last_update_global": current_time,
                "reports": current_state
            }

            # Atomic Write
            temp_file = STATE_FILE + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(final_data, f, indent=2, default=str) # default=str handles objects like datetime inside ML signals

            os.replace(temp_file, STATE_FILE)

            logger.info("Agent state successfully saved (Atomic Merge).")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")

    @staticmethod
    def load_state() -> dict:
        """
        Loads the last known agent reports from the state file.

        Returns:
            dict: A dictionary of agent reports (dicts with 'data' and 'timestamp').
                  Returns an empty dict if no state exists.
        """
        if not os.path.exists(STATE_FILE):
            logger.warning("No previous state file found. Returning empty state.")
            return {}

        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)

            if 'reports' in data:
                logger.info(f"Agent state loaded (Last Global Update: {data.get('last_update_global')}).")
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
