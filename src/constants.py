MODEL_NAME = 'answerdotai/ModernBERT-base'
MAX_LENGTH = 128
MAX_CONTEXT_LENGTH = 256

# Span markers injected around the modifier / head / MWE spans. The model is
# resized to include these tokens; they teach the encoder which words are being
# scored ("<mod> account </mod> <head> book </head>").
MARKER_TOKENS = ['<mod>', '</mod>', '<head>', '</head>', '<mwe>', '</mwe>']