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

        Args:
            reports (dict): A dictionary where keys are agent names and values are their report strings.
        """
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

            # Format data with timestamps for each agent
            current_time = datetime.now().isoformat()

            # If we are updating partial state, we should probably load existing first?
            # The current architecture passes "final_reports" which is Cached + Fresh.
            # So overwriting is fine as long as "final_reports" is complete.
            # But wait, run_specialized_cycle loads cached, updates one, and saves back.
            # So 'reports' here is the FULL state.

            formatted_reports = {}
            for agent, content in reports.items():
                # Handle if content is already a dict (from previous load) or string (fresh)
                if isinstance(content, dict) and 'data' in content:
                    formatted_reports[agent] = content # Preserve existing timestamp if not updated?
                    # Wait, if we pass it back from run_specialized_cycle, we might want to update timestamp only for FRESH.
                    # This method receives a dict of STRINGS or DICTS?
                    # In run_specialized_cycle: final_reports[active] = fresh_string. Others are dicts (from load).
                    # We need to standardize.
                    pass
                else:
                    # New or string content -> Stamp with NOW
                    formatted_reports[agent] = {
                        "data": str(content), # Ensure serialization
                        "timestamp": current_time
                    }

            data = {
                "last_update_global": current_time,
                "reports": formatted_reports
            }

            # Atomic Write
            temp_file = STATE_FILE + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            os.replace(temp_file, STATE_FILE)

            logger.info("Agent state successfully saved (Atomic).")
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
