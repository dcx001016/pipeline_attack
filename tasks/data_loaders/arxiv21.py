import os
import re
import torch
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from communication.comm_utils import *

    
def get_arxiv21_train_data_loader(args, tokenizer, num_workers=0):
    if os.path.exists("datasets/train/arxiv21"):
        data = load_from_disk("datasets/train/arxiv21")
    else:
        data = load_dataset("ds3lab/ac-sgd-arxiv21", split="train")
    encodings = tokenizer("\n\n".join(
        [t.strip() for t in data["abstract"]]
    ), return_tensors="pt")

    input_ids_list = []
    stride = args.seq_length
    for i in tqdm(range(0, encodings.input_ids.size(1)-stride, stride)):
        begin_loc = i
        end_loc = min(i+stride, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        input_ids_list.append(input_ids)
    input_ids = torch.cat(input_ids_list, 0)

    train_set = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'idx': list(range(len(input_ids))),
    })
    
    train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    train_set.set_format(
        type='torch', columns=[
            'text', 'input_ids', 'attention_mask', 'idx',
        ])
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_sampler = torch.utils.data.RandomSampler(train_set, generator=generator)
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    if args.warmup_steps is None:
        args.warmup_steps = len(train_data_loader)
    args.total_steps = args.warmup_steps * args.n_epochs
    return train_data_loader

    
def get_arxiv21_test_data_loader(args, tokenizer, num_workers=0):
    if os.path.exists("datasets/test/arxiv21"):
        data = load_from_disk("datasets/test/arxiv21")
    else:
        data = load_dataset("ds3lab/ac-sgd-arxiv21", split="test")
    encodings = tokenizer("\n\n".join(
        [t.strip() for t in data["abstract"]]
    ), return_tensors="pt")
    
    input_ids_list = []
    stride = args.seq_length
    # TODO: last stride is dropped
    for i in tqdm(range(0, encodings.input_ids.size(1)-stride, stride)):
        begin_loc = i
        end_loc = min(i+stride, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        input_ids_list.append(input_ids)
    input_ids = torch.cat(input_ids_list, 0)
    
    test_set = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'idx': list(range(len(input_ids))),
    })
    
    test_set = test_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    test_set.set_format(
        type='torch', columns=[
            'text', 'input_ids', 'attention_mask', 'idx',
        ])
    
    # TODO: let drop_last be False
    test_data_loader = torch.utils.data.DataLoader(test_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
        
    return test_data_loader