import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
from datasets import load_dataset, load_from_disk

class ConcatFiles:
    def __init__(self, files):
        self.files = files
    
    def __iter__(self):
        for file_path in self.files:
            with open(file_path) as f:
                # skip first line and '\n'
                next(f)
                assert next(f) == '\n'
                for line in f:
                    yield line

class StreamDataset(IterableDataset):
    def __init__(self, data, tokenizer, seq_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.it = None
        
    def get_sequence(self):
        buffer_tokens = [self.tokenizer.bos_token_id]
        for x in self.data:
            curr_tokens = self.tokenizer(x['text'])['input_ids']
            buffer_tokens += curr_tokens
            while len(buffer_tokens) >= self.seq_length:
                tokens = buffer_tokens[:self.seq_length]
                buffer_tokens = [self.tokenizer.bos_token_id] + buffer_tokens[self.seq_length:]
                input_ids = torch.tensor(tokens)
                yield {
                    'text': input_ids, 
                    'input_ids': input_ids,
                    'attention_mask': torch.ones_like(input_ids),
                    'idx': 0, # streaming data do not have idx
                }
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def len(self):
        return len(self.data)
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
        
    
def get_openwebtext_train_data_loader(args, tokenizer, num_workers=0):
    if os.path.exists("datasets/train/openwebtext"):
        data = load_from_disk("datasets/train/openwebtext")
    else:
        data = load_dataset('stas/openwebtext-10k', split='train')
    
    stream_dataset = StreamDataset(data, tokenizer, args.seq_length)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    if args.warmup_steps is None:
        args.warmup_steps = stream_dataset.len() // args.batch_size
    args.total_steps = args.warmup_steps * args.n_epochs
    return train_data_loader
    