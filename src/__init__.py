"""mmBERT compositionality pipeline (gauss-only rebuild).

Trimmed architecture: no span-markers (spans located via tokenizer offsets),
no reg/softmax heads and no MLM warmup phase. A Gaussian (mu, sigma) head per
role reads span + role-aware whole-sentence context features on top of a LoRA
adapter; ``train80`` is the only wired training mode.
"""