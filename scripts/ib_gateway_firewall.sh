#!/usr/bin/env bash
# ib_gateway_firewall.sh — Restrict IB Gateway port to localhost + Tailscale
#
# Usage: sudo bash scripts/ib_gateway_firewall.sh [PORT]
#   PORT defaults to 4001 (IB Gateway live trading)
#
# Idempotent: flushes existing rules for the port before applying new ones.
# Run on the PRODUCTION droplet to allow dev connections via Tailscale only.

set -euo pipefail

PORT="${1:-4001}"
TAILSCALE_IF="tailscale0"

echo "=== IB Gateway Firewall Setup ==="
echo "Port: $PORT"
echo "Tailscale interface: $TAILSCALE_IF"

# Verify tailscale0 exists
if ! ip link show "$TAILSCALE_IF" &>/dev/null; then
    echo "ERROR: Interface $TAILSCALE_IF not found. Is Tailscale running?"
    exit 1
fi

# Flush any existing rules for this port (idempotent)
echo "Flushing existing rules for port $PORT..."
iptables -S INPUT 2>/dev/null | grep -- "--dport $PORT" | while read -r rule; do
    # Convert -A to -D for deletion
    delete_rule="${rule/-A INPUT/-D INPUT}"
    iptables $delete_rule 2>/dev/null || true
done

# Allow localhost (loopback)
echo "Allowing localhost..."
iptables -A INPUT -i lo -p tcp --dport "$PORT" -j ACCEPT

# Allow Tailscale interface
echo "Allowing Tailscale ($TAILSCALE_IF)..."
iptables -A INPUT -i "$TAILSCALE_IF" -p tcp --dport "$PORT" -j ACCEPT

# Drop everything else to this port
echo "Dropping all other traffic to port $PORT..."
iptables -A INPUT -p tcp --dport "$PORT" -j DROP

echo "=== Firewall rules applied ==="
echo ""
echo "Current rules for port $PORT:"
iptables -L INPUT -n -v | head -2
iptables -L INPUT -n -v | grep "$PORT"
echo ""
echo "NOTE: These rules are NOT persistent across reboots."
echo "To persist: apt install iptables-persistent && netfilter-persistent save"
