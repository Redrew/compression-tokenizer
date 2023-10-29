"""
Calculate BLEU score for trained model
"""
import sys
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from train import TextSamplerDataset, decode_tokens, cycle, Logger

config = OmegaConf.load(sys.argv[1])
BATCH_SIZE = 16
PRIME_LEN = 100
SAMPLE_LEN = (128, 4, 4) # reduce sample length

# logging
sys.stdout = Logger(f"outputs/{config.exp_name}/evaluation-log.txt")

# load and evaluate model
model_path = f"outputs/{config.exp_name}/model-latest.pt"
model = torch.load(model_path).module
model.max_seq_len = SAMPLE_LEN
class GenerateWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, text):
        return self.model.generate(text)
devices = list(range(torch.cuda.device_count()))
model = torch.nn.DataParallel(GenerateWrapper(model), device_ids=devices)
dataset = load_dataset(config.dataset)
val_dataset = TextSamplerDataset(
    dataset["validation"],
    seq_len=np.prod(SAMPLE_LEN),
    tokenizer=config.tokenizer,
    zip_multiplier=config.zip_multiplier
)
val_loader = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))
bleu = evaluate.load("bleu")

text = next(val_loader)
text = text[:, :-1]
sample = model(text[:, :PRIME_LEN]).flatten(1)
decode_tokens(sample[0].cpu(), config.tokenizer)
prime_texts = [decode_tokens(line.cpu(), config.tokenizer) for line in text[:, :PRIME_LEN]]
sampled_texts = [decode_tokens(line.cpu(), config.tokenizer)[len(prime):] for line, prime in zip(sample, prime_texts)]
ref_texts = [decode_tokens(line.cpu(), config.tokenizer)[len(prime):] for line, prime in zip(text, prime_texts)]
results = bleu.compute(predictions=sampled_texts, references=ref_texts)
print(results)
for prime_text, ref_text, sampled_text in zip(prime_texts, ref_texts, sampled_texts):
    print(f"---{prime_text}---")
    print(f"REFERENCE: {ref_text}")
    print(f"SAMPLE: {sampled_text}")

# TODO calculate perplexity per character
