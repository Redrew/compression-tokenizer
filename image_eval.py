import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import os
import sys
from datasets import load_dataset
from train import MNISTDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

config = OmegaConf.load(sys.argv[1])
IMAGES_TO_SAVE = 20
PERCENT_EXISTS = 0.5


def reverse_rle(bytes):
    result = []
    for i in range(0, len(bytes), 2):
        result.extend([bytes[i]] * int(bytes[i+1]))
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
mnist = MNISTDataset(dataset["test"], device=device, tokenizer="rle")

if IMAGES_TO_SAVE > 0:
    os.makedirs(f"outputs/{config.exp_name}/images", exist_ok=True)
    if PERCENT_EXISTS == 0.0:
        raise ValueError("PERCENT_EXISTS cannot be 0.0")
    else:
        for i in range(IMAGES_TO_SAVE):
            pixels_to_keep = int(mnist[i] != mnist.pad_id * PERCENT_EXISTS)
            set_pixels = mnist[i][:pixels_to_keep]
            model_output = model.generate(set_pixels.unsqueeze(0)).flatten(1)
            autocompleted_image = reverse_rle(model_output[0])
            image = convert_to_image(autocompleted_image)
            plt.imsave(f"outputs/{config.exp_name}/images/{PERCENT_EXISTS}_{i}.png", image, cmap="gray")
            breakpoint()

mse = 0.0

breakpoint()
dataloader = DataLoader(mnist, batch_size=config.batch_size, shuffle=True)
for images in dataloader:
    model_output = model(images, return_loss=False)
    breakpoint()
    autocompleted_image = reverse_rle(model_output)
    full_image =  torch.concatenate((set_pixels, autocompleted_image))
    image = convert_to_image(full_image)
    mse += np.mean((image - set_pixels).float()**2)

print(f"Average MSE: {mse / 10_000}")
print(f"Average PSNR: {20 * np.log10(255) - 10 * np.log10(mse / 10_000)}")
import math
print(f"Average PSNR (not copilot written)", 10 * math.log10(255**2 / (mse / 10_000)))