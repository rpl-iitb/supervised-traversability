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
seq 1 254 | xargs -P 64 -I{} sh -c "nc -z -G 1 $PREFIX.{} 8765 2>/dev/null && nc -z -G 1 $PREFIX.{} 9090 2>/dev/null && echo $PREFIX.{}"
