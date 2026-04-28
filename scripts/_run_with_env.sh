#!/bin/bash
# AutoDL wrapper: set PATH (miniconda), enable network_turbo proxy, then exec argv.
# Used by tmux-launched data-gen / training jobs so they have HF/GitHub access
# and the right Python on PATH regardless of whether ~/.bashrc was sourced.
export PATH=/root/miniconda3/bin:$PATH
[ -f /etc/network_turbo ] && source /etc/network_turbo
# Disable HF xethub backend — its CAS server returns 401 through the AutoDL proxy.
# Forces the legacy download path which works reliably.
export HF_HUB_DISABLE_XET=1
exec "$@"
