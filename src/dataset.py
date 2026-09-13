import torch
from torch.utils.data import Dataset

from src.matching import fallback_marked_text, mark_compound, span_text_offsets


class NNDataset(Dataset):
    """Tokenizer wrapper producing ONE marked sentence per row.

    The compound's modifier / head spans are wrapped in ``<mod>`` / ``<head>``
    markers (whole MWE in ``<mwe>``) inside the real context sentence. Each
    item carries ``input_ids``, ``attention_mask`` and one boolean span mask
    per role, so the model pools hidden states over the marked words only.

    ``max_context_length`` caps the marked sentence (the context sentences are
    longer than the old standalone Mod/Head/Compound items).
    """

    def __init__(self, df, tokenizer, max_length=128, max_context_length=256,
                 is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_context_length
        self.is_test = is_test

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        marked = mark_compound(
            str(row['Context']), str(row['Mod']), str(row['Head'])
        )
        if marked is None:
            # Rare: compound not found verbatim in the sentence. Fall back to a
            # deterministic marked compound + the (unmarked) context.
            marked = fallback_marked_text(
                str(row['Mod']), str(row['Head']), str(row['Context'])
            )

        encoded = self.tokenizer(
            marked,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        offsets = encoded['offset_mapping'].squeeze(0).tolist()

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mod_span_mask': self._span_mask(marked, 'mod', offsets),
            'head_span_mask': self._span_mask(marked, 'head', offsets),
            'mwe_span_mask': self._span_mask(marked, 'mwe', offsets),
        }

        if not self.is_test and 'ModAvg' in row and 'HeadAvg' in row:
            item['mod_avg'] = torch.tensor(float(row['ModAvg']), dtype=torch.float)
            item['head_avg'] = torch.tensor(float(row['HeadAvg']), dtype=torch.float)
            if 'ModStd' in row:
                item['mod_std'] = torch.tensor(float(row['ModStd']), dtype=torch.float)
            if 'HeadStd' in row:
                item['head_std'] = torch.tensor(float(row['HeadStd']), dtype=torch.float)

        return item

    def _span_mask(self, marked, tag, offsets):
        """Boolean mask over tokens whose character window lies inside the span."""
        span = span_text_offsets(marked, tag)
        mask = torch.zeros(len(offsets), dtype=torch.bool)
        if span is None:
            return mask
        start, end = span
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end > start and tok_start < end:
                mask[i] = True
        return mask