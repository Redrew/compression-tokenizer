"""
Calculate BLEU score for trained model
"""
import sys
import torch
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader

from train import TextSamplerDataset, decode_tokens, cycle, Logger

EXP_NAME = "pg19-bytes"
TOKENIZER = EXP_NAME.split("-")[1]
BATCH_SIZE = 16
PRIME_LEN = 100
SEQ_LEN = 8192

# logging
sys.stdout = Logger(f"outputs/{EXP_NAME}/evaluation-log.txt")

# load and evaluate model
model_path = f"outputs/{EXP_NAME}/model-latest.pt"
model = torch.load(model_path).module
class GenerateWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, text):
        return self.model.generate(text)
devices = list(range(torch.cuda.device_count()))
model = torch.nn.DataParallel(GenerateWrapper(model), device_ids=devices)
dataset = load_dataset("pg19")
val_dataset = TextSamplerDataset(dataset["validation"], SEQ_LEN, tokenizer=TOKENIZER)
val_loader = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))
bleu = evaluate.load("bleu")

text = next(val_loader)
text = text[:, :-1]
prime = text[:, :PRIME_LEN]
sample = model(prime).flatten(1)
sampled_text = [decode_tokens(line) for line in sample[:, PRIME_LEN:]]
ref_text = [decode_tokens(line) for line in text[:, PRIME_LEN:]]
results = bleu.compute(predictions=sampled_text, references=ref_text)
print(results)
for p, r, s in zip([decode_tokens(line) for line in prime], ref_text, sampled_text):
    print(f"---{p}---")
    print(f"REFERENCE: {r}")
    print(f"SAMPLE: {s}")
