"""mmBERT compositionality pipeline (rebuild).

New architecture: no span-markers (spans located via tokenizer offsets),
compound-aware MLM warmup through LoRA (merged into the base), then an
attention-pool scoring head on top of a fresh LoRA adapter.
"""