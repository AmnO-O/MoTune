"""Pipeline-wide constants."""

from __future__ import annotations

# Label range. Real labels span [0.0, 5.0]; ordinal bins and prediction
# clipping must cover 0.0 .. 5.0 (starting at 1.0 would clamp sub-1.0 gold).
SCORE_MIN = 0.0
SCORE_MAX = 5.0