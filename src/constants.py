MODEL_NAME = 'answerdotai/ModernBERT-base'
MAX_LENGTH = 128
MAX_CONTEXT_LENGTH = 256

# Label score range. Real labels span [0.0, 5.0] (en-nn ModAvg min=0.0), so the
# ordinal bins (and any prediction clipping) must cover 0.0 .. 5.0; starting at
# 1.0 would clamp every sub-1.0 gold score to the lowest bin.
SCORE_MIN = 0.0
SCORE_MAX = 5.0

# Span markers injected around the modifier / head spans. The model is
# resized to include these tokens; they teach the encoder which words are being
# scored ("<mod> account </mod> <head> book </head>"). The whole MWE is always
# the contiguous mod+head pair, so no separate <mwe> markers are needed.
MARKER_TOKENS = ['<mod>', '</mod>', '<head>', '</head>']