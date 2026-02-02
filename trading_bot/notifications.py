from enum import Enum

class NotificationTier(Enum):
    """Notification routing tier based on severity. Commodity-agnostic."""
    LOG_ONLY = "log_only"       # severity 0-4: Log only, no external notification
    DASHBOARD = "dashboard"     # severity 5-6: Log + dashboard state (future Slack)
    PUSHOVER = "pushover"       # severity 7-8: Pushover notification (throttled)
    CRITICAL = "critical"       # severity 9-10: Pushover + emergency escalation

def get_notification_tier(severity: int) -> NotificationTier:
    """Map severity to notification tier. Commodity-agnostic."""
    if severity >= 9:
        return NotificationTier.CRITICAL
    elif severity >= 7:
        return NotificationTier.PUSHOVER
    elif severity >= 5:
        return NotificationTier.DASHBOARD
    return NotificationTier.LOG_ONLY
