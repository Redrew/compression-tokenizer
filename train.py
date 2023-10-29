import relaxed_gzip as rgzip
from MEGABYTE_pytorch import MEGABYTE

import os
import sys
import re
import random
import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from datasets import load_dataset
from transformers import GPT2Tokenizer

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_tokens(tokens, tokenizer="bytes"):
    if tokenizer == "bytes":
        return ''.join(list(map(lambda token: str(chr(max(32, token))), tokens)))
    elif tokenizer == "gzip":
        bytes = np.array(tokens, dtype=np.uint8).tobytes()
        assert np.all(np.frombuffer(bytes, dtype=np.uint8) == np.array(tokens, dtype=np.uint8)), "int to byte string does not match byte string to int conversion"
        try:
            decoding = rgzip.decompress(bytes).decode("ascii")
        except:
            decoding = ''.join(list(map(lambda token: str(chr(max(32, token))), tokens)))
        return decoding
    elif tokenizer == "bpe":
        bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        return bpe_tokenizer.decode(tokens)

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

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, tokenizer="bytes", zip_multiplier=2):
        super().__init__()
        self.zip_multiplier = zip_multiplier
        self.data = data
        self.seq_len = seq_len
        self.doc_lengths = np.array([len(doc["text"]) for doc in data])
        self.doc_lengths[self.doc_lengths <= self.seq_len * self.zip_multiplier] = 0
        self.tokenizer = tokenizer
        if self.tokenizer == "bpe":
            self.bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def __getitem__(self, index):
        rand_doc = np.random.choice(len(self.doc_lengths), p=self.doc_lengths/self.doc_lengths.sum())
        text = self.data[rand_doc]["text"]
        # sample a longer piece of text if we are zipping
        text_seq_len = self.seq_len * self.zip_multiplier
        rand_start = torch.randint(0, len(text) - text_seq_len, (1,))
        text_slice = text[rand_start: rand_start + text_seq_len]
        if self.tokenizer == "bytes":
            bytes = re.sub(r'[^\x00-\x7F]+', ' ', text_slice).encode("ascii")
            token_ids = np.frombuffer(bytes, dtype=np.uint8).copy()
        elif self.tokenizer == "gzip":
            bytes = re.sub(r'[^\x00-\x7F]+', ' ', text_slice).encode("ascii")
            bytes = rgzip.compress(bytes)
            token_ids = np.frombuffer(bytes, dtype=np.uint8).copy()
        elif self.tokenizer == "bpe":
            token_ids = self.bpe_tokenizer.encode(text_slice)
        
        if len(token_ids) < self.seq_len:
            return self[index]
        full_seq = torch.LongTensor(token_ids[:self.seq_len])
        return full_seq.cuda()

    def __len__(self):
        return self.doc_lengths.sum() // self.seq_len

if __name__ == "__main__":
    # config
    VALIDATE_EVERY = 100
    GENERATE_EVERY = 500
    PRIME_LEN = 100
    config = OmegaConf.load(sys.argv[1])

    os.makedirs(f"outputs/{config.exp_name}", exist_ok=True)
    sys.stdout = Logger(f"outputs/{config.exp_name}/log.txt")

    # instantiate GPT-like decoder model

    model = MEGABYTE(
        num_tokens = config.num_tokens,
        dim = (768, 512, 256),
        depth = (6, 4, 2),
        max_seq_len = (512, 4, 4),
        flash_attn = True
    ).cuda()

    # prepare data
    dataset = load_dataset(config.dataset)

    train_dataset = TextSamplerDataset(dataset["train"], config.seq_len, tokenizer=config.tokenizer, zip_multiplier=config.zip_multiplier)
    val_dataset   = TextSamplerDataset(dataset["validation"], config.seq_len, tokenizer=config.tokenizer, zip_multiplier=config.zip_multiplier)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = config.batch_size))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = config.batch_size))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # multi-gpu training
    if config.enable_dp:
        devices = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=devices)
        model.generate = model.module.generate

    # training

    for i in tqdm.tqdm(range(config.num_batches), mininterval=10., desc='training'):
        model.train()

        for __ in range(config.gradient_accumulate_every):
            loss = model(next(train_loader), return_loss = True)
            if config.enable_dp:
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
                if config.enable_dp:
                    loss = loss.mean()
                print(f'validation loss: {loss.item()}')

        if i != 0 and i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime_inp = inp[:PRIME_LEN]
            prime = decode_tokens(prime_inp.cpu(), config.tokenizer)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(prime_inp[None, :])
            sample = sample.flatten(1)

            output_str = decode_tokens(sample[0].cpu(), config.tokenizer)[len(prime):]
            print(output_str)
            torch.save(model, f"outputs/{config.exp_name}/model-latest.pt")

    torch.save(model, f"outputs/{config.exp_name}/model.pt")
