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
from transformers import GPT2Tokenizer, BertTokenizer
from bwt import BBWT, BBWT_inv
import fastlz

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def RLE(seq):
    ret_seq = []
    i = 0
    while i < len(seq):
        j = i + 1
        while j - i < 255 and j < len(seq) and seq[j] == seq[i]:
            j += 1
        ret_seq.extend([seq[i], j - i])
        i = j
    return ret_seq

def MTF(seq):
    ret_seq = []
    alphabet = list(range(256))
    for token in seq:
        ret_seq.append(alphabet.index(token))
        alphabet.pop(alphabet.index(token))
        alphabet.insert(0, token)
    return ret_seq

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
    elif tokenizer == "lz77":
        bytes = np.array(tokens, dtype=np.uint8).tobytes()
        try:
            decoding = fastlz.decompress(bytes).decode("ascii")
        except:
            decoding = ''.join(list(map(lambda token: str(chr(max(32, token))), tokens)))
        return decoding
    elif tokenizer == "bpe":
        bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        return bpe_tokenizer.decode(tokens)
    elif tokenizer == "wordpiece":
        word_piece_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return word_piece_tokenizer.decode(tokens)
    elif tokenizer == "rle":
        final_bytes = []
        for i in range(0, len(tokens) // 2 * 2, 2):
            final_bytes.extend([tokens[i]] * tokens[i+1])
        return ''.join(list(map(lambda token: str(chr(max(32, token))), final_bytes)))
    elif tokenizer == "mtf-rle":
        expanded_bytes = []
        for i in range(0, len(tokens), 2):
            expanded_bytes.extend([tokens[i]] * tokens[i+1])
        text = ''.join(list(map(lambda token: str(chr(max(32, token))), expanded_bytes)))
        return BBWT_inv(text)
    else:
        return ''.join(list(map(lambda token: str(chr(max(32, token))), tokens)))

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

class MNISTDataset(Dataset):
    def __init__(self, dataset, tokenizer="bytes", device="cuda", pad_id=None):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.data = []
        for row in tqdm.tqdm(dataset):
            image = np.array(row["image"]).flatten()
            
            if tokenizer == "rle":
                encoded_bytes = RLE(image) 
                if pad_id is not None:
                    encoded_bytes += [pad_id] * (784 - len(encoded_bytes))
                image = np.array(encoded_bytes, dtype=np.uint8).copy()
            self.data.append(image)

    def __getitem__(self, index):
        tokens = self.data[index]
        return torch.LongTensor(tokens).to(self.device)

    def __len__(self):
        return len(self.data)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, tokenizer="bytes", zip_multiplier=2, device="cuda", pad_id=-1, sep_id=-1):
        super().__init__()
        self.device = device
        self.zip_multiplier = zip_multiplier
        self.data = data
        self.seq_len = seq_len
        self.doc_lengths = np.array([len(doc["text"]) for doc in data])
        self.doc_lengths[self.doc_lengths <= int(self.seq_len * self.zip_multiplier)] = 0
        self.tokenizer = tokenizer
        if self.tokenizer == "bpe":
            self.bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer == "wordpiece":
            self.word_piece_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad_id = pad_id
        self.sep_id = sep_id

    def __getitem__(self, index):
        rand_doc = np.random.choice(len(self.doc_lengths), p=self.doc_lengths/self.doc_lengths.sum())
        text = self.data[rand_doc]["text"]
        # sample a longer piece of text if we are zipping
        text_seq_len = int(self.seq_len * self.zip_multiplier)
        rand_start = torch.randint(0, len(text) - text_seq_len, (1,))
        text_slice = text[rand_start: rand_start + text_seq_len]
        if self.tokenizer == "bytes":
            bytes = re.sub(r'[^\x00-\x7F]+', ' ', text_slice).encode("ascii")
            token_ids = np.frombuffer(bytes, dtype=np.uint8).copy()
        elif self.tokenizer == "lz77":
            bytes = fastlz.compress(re.sub(r'[^\x00-\x7F]+', ' ', text_slice))
            token_ids = np.frombuffer(bytes, dtype=np.uint8).copy()
        elif self.tokenizer == "gzip":
            bytes = re.sub(r'[^\x00-\x7F]+', ' ', text_slice).encode("ascii")
            bytes = rgzip.compress(bytes)
            token_ids = np.frombuffer(bytes, dtype=np.uint8).copy()
        elif self.tokenizer == "bpe":
            token_ids = self.bpe_tokenizer.encode(text_slice)
        elif self.tokenizer == "wordpiece":
            token_ids = self.word_piece_tokenizer.encode(text_slice)
        elif self.tokenizer == "rle":
            bytes = re.sub(r'[^\x00-\x7F]+', ' ', text_slice).encode("ascii")
            encoded_bytes = RLE(bytes)
            token_ids = np.array(encoded_bytes, dtype=np.uint8).copy()
        elif self.tokenizer == "mtf-rle":
            bytes = re.sub(r'[^\x00-\x7F]+', ' ', text_slice).encode("ascii")
            encoded_bytes = RLE(MTF(bytes))
            token_ids = np.array(encoded_bytes, dtype=np.uint8).copy()
        elif self.tokenizer == "bwt-rle":
            text = re.sub(r'[^\x00-\x7F]+', ' ', text_slice)
            encoded_bytes = RLE(BBWT(text).encode("ascii"))
            token_ids = np.array(encoded_bytes, dtype=np.uint8).copy()
        elif self.tokenizer == "gzip-uncompression":
            bytes = re.sub(r'[^\x00-\x7F]+', ' ', text_slice).encode("ascii")
            encoded_bytes = rgzip.compress(bytes)
            token_ids = np.concatenate([
                np.frombuffer(encoded_bytes, dtype=np.uint8).copy().astype(np.uint16),
                [self.sep_id],
                np.frombuffer(bytes, dtype=np.uint8).copy().astype(np.uint16),
            ])
        elif self.tokenizer == "lz77-uncompression":
            bytes = re.sub(r'[^\x00-\x7F]+', ' ', text_slice).encode("ascii")
            encoded_bytes = fastlz.compress(bytes)
            token_ids = np.concatenate([
                np.frombuffer(encoded_bytes, dtype=np.uint8).copy().astype(np.uint16),
                [self.sep_id],
                np.frombuffer(bytes, dtype=np.uint8).copy().astype(np.uint16),
            ])
        
        # if len(token_ids) < self.seq_len:
        #     return self[index]
        full_seq = torch.LongTensor(token_ids[:self.seq_len])
        if len(full_seq) < self.seq_len:
            full_seq = torch.cat([full_seq, torch.LongTensor([self.pad_id] * (self.seq_len - len(full_seq)))])
        return full_seq.to(self.device)

    def __len__(self):
        return self.doc_lengths.sum() // self.seq_len

class MixedDataset(Dataset):
    def __init__(self, text_dataset, image_dataset, seq_len, pad_id, sample_many_images=True):
        super().__init__()
        self.text_dataset = text_dataset
        self.image_dataset = image_dataset
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.sample_many_images = sample_many_images

    def __len__(self):
        return len(self.text_dataset) + len(self.image_dataset)

    def __getitem__(self, index):
        if np.random.rand() < 0.5:
            seq = self.text_dataset[np.random.randint(len(self.text_dataset))]
        else:
            if not self.sample_many_images:
                seq = self.image_dataset[np.random.randint(len(self.image_dataset))]
            else:
                total_len = 0
                seq_elems = []
                while True:
                    new_image = self.image_dataset[np.random.randint(len(self.image_dataset))]
                    new_image = torch.concat([new_image, torch.LongTensor([self.pad_id]).to(new_image.device)])
                    if total_len + len(new_image)> self.seq_len:
                        break
                    total_len += len(new_image)
                    seq_elems.append(new_image)
                seq = torch.concat(seq_elems + [torch.LongTensor([self.pad_id] * (self.seq_len - total_len)).to(new_image.device)])
            
        if len(seq) < self.seq_len:
            seq = torch.cat([seq, torch.LongTensor([self.pad_id] * (self.seq_len - len(seq))).to(seq.device)])
        return seq

if __name__ == "__main__":
    # config
    VALIDATE_EVERY = 100
    GENERATE_EVERY = np.inf
    PRIME_LEN = 100
    config = OmegaConf.load(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert config.dataset in ["pg19", "mnist", "pg19-mnist"]

    os.makedirs(f"outputs/{config.exp_name}", exist_ok=True)
    sys.stdout = Logger(f"outputs/{config.exp_name}/log.txt")

    # instantiate GPT-like decoder model

    pad_id = config.num_tokens
    sep_id = config.num_tokens - 1 if "uncompression" in config.tokenizer else None
    model = MEGABYTE(
        num_tokens = config.num_tokens + 1,
        dim = (768, 512, 256),
        depth = (6, 4, 2),
        max_seq_len = (512, 4, 4),
        flash_attn = True,
        pad_id = pad_id,
        sep_id = sep_id,
    ).to(device)

    # prepare dataset
    if config.dataset == "pg19":
        dataset = load_dataset(config.dataset)
        train_dataset = TextSamplerDataset(dataset["train"], config.seq_len, tokenizer=config.tokenizer, zip_multiplier=config.zip_multiplier, device=device, pad_id=pad_id, sep_id=sep_id)
        val_dataset   = TextSamplerDataset(dataset["validation"], config.seq_len, tokenizer=config.tokenizer, zip_multiplier=config.zip_multiplier, device=device, pad_id=pad_id, sep_id=sep_id)
        train_loader  = cycle(DataLoader(train_dataset, batch_size = config.batch_size))
        val_loader    = cycle(DataLoader(val_dataset, batch_size = config.batch_size))
    if config.dataset == "mnist":
        dataset = load_dataset(config.dataset)
        print("Loading MNIST dataset")
        train_dataset = MNISTDataset(dataset["train"], tokenizer=config.tokenizer, device=device, pad_id=pad_id)
        val_dataset   = MNISTDataset(dataset["test"], tokenizer=config.tokenizer, device=device, pad_id=pad_id)
        train_loader  = cycle(DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True))
        val_loader    = cycle(DataLoader(val_dataset, batch_size = config.batch_size, shuffle=True))
    if config.dataset == "pg19-mnist":
        print("Loading both PG19 and MNIST dataset")
        text_dataset = load_dataset("pg19")
        image_dataset = load_dataset("mnist")
        train_dataset = MixedDataset(
            TextSamplerDataset(text_dataset["train"], config.seq_len, tokenizer=config.tokenizer, zip_multiplier=config.zip_multiplier, device=device, pad_id=pad_id, sep_id=sep_id),
            MNISTDataset(image_dataset["train"], tokenizer=config.tokenizer, device=device, pad_id=None),
            config.seq_len,
            pad_id,
        )
        val_dataset   = MixedDataset(
            TextSamplerDataset(text_dataset["validation"], config.seq_len, tokenizer=config.tokenizer, zip_multiplier=config.zip_multiplier, device=device, pad_id=pad_id, sep_id=sep_id),
            MNISTDataset(image_dataset["test"], tokenizer=config.tokenizer, device=device, pad_id=None),
            config.seq_len,
            pad_id,
        )
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
