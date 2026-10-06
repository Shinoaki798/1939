#!/usr/bin/env bash
# Like science_ia_more.sh but runs at once, in parallel with science_ia.sh (two streams to archive.org).
NOWAIT=1 exec bash "$HOME/1939/scripts/science_ia_more.sh" "$@"
