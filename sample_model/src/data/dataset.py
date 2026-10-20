"""
Dataset and DataLoader utilities for TinyStories training.
Streams or loads tokenized stories into fixed context length blocks (e.g. 256 tokens).
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset

class TinyStoriesDataset(Dataset):
    """
    Dataset that loads tokenized stories and packs them into contiguous blocks of max_seq_len.
    """
    def __init__(self, data_file: str, tokenizer_path: str, max_seq_len: int = 256, max_tokens: int = 5_000_000):
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        print(f"Loading and tokenizing corpus from {data_file}...")
        all_token_ids = []
        with open(data_file, "r", encoding="utf-8") as f:
            lines_chunk = []
            for i, line in enumerate(f):
                lines_chunk.append(line)
                if len(lines_chunk) >= 5000:
                    encoded = self.tokenizer.encode("".join(lines_chunk))
                    all_token_ids.extend(encoded.ids)
                    lines_chunk = []
                    if max_tokens and len(all_token_ids) >= max_tokens:
                        break
            if lines_chunk and (not max_tokens or len(all_token_ids) < max_tokens):
                encoded = self.tokenizer.encode("".join(lines_chunk))
                all_token_ids.extend(encoded.ids)

        if max_tokens and len(all_token_ids) > max_tokens:
            all_token_ids = all_token_ids[:max_tokens]

        self.tokens = torch.tensor(all_token_ids, dtype=torch.long)
        print(f"Total tokens in dataset: {len(self.tokens):,}")
        
        # Calculate number of full sequences
        self.num_samples = max(0, (len(self.tokens) - 1) // self.max_seq_len)
        print(f"Total training chunks ({self.max_seq_len} tokens each): {self.num_samples:,}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * self.max_seq_len
        chunk = self.tokens[start : start + self.max_seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloader(
    data_file: str = "data/tinystories_corpus.txt",
    tokenizer_path: str = "data/tinystories_tokenizer.json",
    batch_size: int = 32,
    max_seq_len: int = 256,
    max_tokens: int = 5_000_000,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    dataset = TinyStoriesDataset(
        data_file=data_file,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_tokens=max_tokens
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

if __name__ == "__main__":
    loader = get_dataloader(batch_size=4, max_seq_len=64, max_tokens=10000)
    for x, y in loader:
        print(f"Sample Batch - x: {x.shape}, y: {y.shape}")
        print(f"x[0][:10]: {x[0][:10]}")
        print(f"y[0][:10]: {y[0][:10]}")
        break
