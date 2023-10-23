from MEGABYTE_pytorch import MEGABYTE

import io
import os
import sys
import re
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# constants

EXP_NAME = sys.argv[1]
ZIPPED = True
NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LEN = 100
SEQ_LEN = 8192
ENABLE_DP = True

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# logging
class Logger(object):
    def __init__(self, f):
        self.terminal = sys.stdout
        self.log = open(f, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass
os.makedirs(f"outputs/{EXP_NAME}", exist_ok=True)
sys.stdout = Logger(f"outputs/{EXP_NAME}/log.txt")

# instantiate GPT-like decoder model

model = MEGABYTE(
    num_tokens = 256,
    dim = (768, 512, 256),
    depth = (6, 4, 2),
    max_seq_len = (512, 4, 4),
    flash_attn = True
).cuda()

# prepare data
split = "train"
dataset = load_dataset("pg19")

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, zipped=False):
        super().__init__()
        self.zip_multiplier = 8 if zipped else 2
        self.data = data
        self.seq_len = seq_len
        self.doc_lengths = np.array([len(doc["text"]) for doc in data])
        self.doc_lengths[self.doc_lengths <= self.seq_len * self.zip_multiplier] = 0
        self.zipped = zipped

    def __getitem__(self, index):
        rand_doc = np.random.choice(len(self.doc_lengths), p=self.doc_lengths/self.doc_lengths.sum())
        text = self.data[rand_doc]["text"]
        # sample a longer piece of text if we are zipping
        text_seq_len = self.seq_len * self.zip_multiplier
        rand_start = torch.randint(0, len(text) - text_seq_len, (1,))
        bytes = re.sub(r'[^\x00-\x7F]+', ' ', text[rand_start: rand_start + text_seq_len]).encode("ascii")
        if self.zipped:
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                f.write(bytes)
            bytes = buffer.getvalue()
        if len(bytes) < self.seq_len:
            return self[index]
        full_seq = torch.LongTensor(np.frombuffer(bytes[:self.seq_len], dtype=np.uint8).copy())
        return full_seq.cuda()

    def __len__(self):
        return self.doc_lengths.sum() // self.seq_len

train_dataset = TextSamplerDataset(dataset["train"], SEQ_LEN, zipped=ZIPPED)
val_dataset   = TextSamplerDataset(dataset["validation"], SEQ_LEN, zipped=ZIPPED)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# multi-gpu training
if ENABLE_DP:
    devices = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.generate = model.module.generate

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        if ENABLE_DP:
            loss = loss.mean()
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            if ENABLE_DP:
                loss = loss.mean()
            print(f'validation loss: {loss.item()}')

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime_inp = inp[:PRIME_LEN]
        prime = decode_tokens(prime_inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(prime_inp[None, :])
        sample = sample.flatten(1)

        output_str = decode_tokens(sample[0][PRIME_LEN:])
        print(output_str)

torch.save(model, f"outputs/{EXP_NAME}/model.pt")
