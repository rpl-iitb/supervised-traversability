#!/usr/bin/env bash
# Finds the Jackal's current IP (it changes: DHCP on the lab Wi-Fi). Runs on the Mac.
# The robot runs Foxglove bridge on port 8765 and Cockpit on 9090, so we list
# hosts with both open. Confirm with `ssh robot@<ip> hostname` (cpr-j100-0952).
#
#     bash hardware/find_robot.sh                 # scans 10.129.152.0/24
#     bash hardware/find_robot.sh 10.129.153      # another /24
#
# Verified 2026-10-21: found the robot at 10.129.152.239 in 4 s from a Mac on 10.51.x
# (a different IITB subnet), so this also works away from the lab.

PREFIX="${1:-10.129.152}"
echo "Scanning $PREFIX.1-254 for Foxglove (8765) + Cockpit (9090)..." >&2
FOUND=$(seq 1 254 | xargs -P 64 -I{} sh -c "nc -z -G 2 $PREFIX.{} 8765 2>/dev/null && nc -z -G 2 $PREFIX.{} 9090 2>/dev/null && echo $PREFIX.{}")
if [ -n "$FOUND" ]; then
  echo "$FOUND"
  exit 0
fi
ALIVE=$(seq 1 254 | xargs -P 64 -I{} sh -c "ping -c 1 -t 1 $PREFIX.{} >/dev/null 2>&1 && echo x" | wc -l | tr -d ' ')
if [ "$ALIVE" -gt 0 ]; then
  echo "Robot not found, but $ALIVE other devices in $PREFIX.x answered, so the network is reachable." >&2
  echo "The robot is probably switched off or not on Wi-Fi (ERTS_5G). Ask someone in the lab, or try another subnet." >&2
else
  echo "Nothing in $PREFIX.x answered at all: this network can't reach that subnet (try from the lab or IITB Wi-Fi)." >&2
fi
exit 1
