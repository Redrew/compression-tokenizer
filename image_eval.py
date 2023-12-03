import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import os
import sys
from datasets import load_dataset
from train import MNISTDataset
import torch
from tqdm import tqdm

config = OmegaConf.load(sys.argv[1])
IMAGES_TO_SAVE = 20
PERCENT_EXISTS = 0.5


def reverse_rle(bytes):
    result = []
    for i in range(0, len(bytes), 2):
        result.extend([bytes[i]] * bytes[i+1])
    return torch.tensor(result, dtype=torch.uint8)

def convert_to_image(data):
    if len(data) < 784:
        data = np.pad(data, (0, 784 - len(data)), "constant")
    else:
        data = data[:784]
    return data.reshape(28, 28)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = f"outputs/{config.exp_name}/model-latest.pt"
model = torch.load(model_path)
pad_id = config.num_tokens

dataset = load_dataset("mnist")
mnist = MNISTDataset(dataset["test"], device=device)
pixels_to_keep = int(784 * PERCENT_EXISTS)

if IMAGES_TO_SAVE > 0:
    os.makedirs(f"outputs/{config.exp_name}/images", exist_ok=True)
    if PERCENT_EXISTS == 0.0:
        raise ValueError("PERCENT_EXISTS cannot be 0.0")
    else:
        for i in range(IMAGES_TO_SAVE):
            set_pixels = mnist.__getitem__(i)[:pixels_to_keep]
            model_output = model(set_pixels, return_loss=False) # Need to check this line
            autocompleted_image = reverse_rle(model_output)
            full_image =  torch.concatenate((set_pixels, autocompleted_image))
            image = convert_to_image(full_image)
            plt.imsave(f"outputs/{config.exp_name}/images/{PERCENT_EXISTS}_{i}.png", image, cmap="gray")

mse = 0.0

for i in tqdm(range(10_000)):
    target_image = mnist.__getitem__(i)
    set_pixels = mnist.__getitem__(i)
    model_output = model(set_pixels, return_loss=False)
    autocompleted_image = reverse_rle(model_output)
    full_image =  torch.concatenate((set_pixels, autocompleted_image))
    image = convert_to_image(full_image)
    mse += np.mean((image - set_pixels).float()**2)

print(f"Average MSE: {mse / 10_000}")
print(f"Average PSNR: {20 * np.log10(255) - 10 * np.log10(mse / 10_000)}")
import math
print(f"Average PSNR (not copilot written)", 10 * math.log10(255**2 / (mse / 10_000)))