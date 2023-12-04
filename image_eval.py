import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import os
import sys
from datasets import load_dataset
from train import MNISTDataset, RLE
import torch
from tqdm import tqdm

config = OmegaConf.load(sys.argv[1])
IMAGES_TO_SAVE = 0
PERCENT_EXISTS = 0.5
MAX_TOKENS = 784

def reverse_rle(bytes):
    result = []
    for i in range(len(bytes)//2):
        result.extend([bytes[i*2]] * bytes[(i*2)+1])
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

with torch.no_grad():
    if IMAGES_TO_SAVE > 0:
        os.makedirs(f"outputs/{config.exp_name}/images", exist_ok=True)
        if PERCENT_EXISTS == 0.0:
            raise ValueError("PERCENT_EXISTS cannot be 0.0")
        else:


            for i in tqdm(range(IMAGES_TO_SAVE)):
                set_pixels = mnist.__getitem__(i)[:pixels_to_keep]
                set_pixels = torch.tensor(RLE(set_pixels)).reshape(1, -1).to(device)
                
                for _ in range(MAX_TOKENS - len(set_pixels)):
                    model_output = model(set_pixels)
                    model_predictions = model_output.argmax(dim=-1)[0][-1]
                    set_pixels = torch.cat((set_pixels, torch.tensor([[model_predictions]]).to(device)), dim=1)

                autocompleted_image = reverse_rle(set_pixels[0].cpu())
                
                image = convert_to_image(autocompleted_image)
                plt.imsave(f"outputs/{config.exp_name}/images/{PERCENT_EXISTS}_{i}.png", image, cmap="gray")

    mse = 0.0

    BATCH_SIZE = 20
    for i in tqdm(range(10_000/BATCH_SIZE)):
        target_image = mnist.__getitem__(i).cpu().numpy()
        set_pixels = mnist.__getitem__(i)[:pixels_to_keep]
        set_pixels = torch.tensor(RLE(set_pixels)).reshape(1, -1).to(device)
        
        for _ in range(MAX_TOKENS - len(set_pixels)):
            model_output = model(set_pixels)
            model_predictions = model_output.argmax(dim=-1)[0][-1]
            set_pixels = torch.cat((set_pixels, torch.tensor([[model_predictions]]).to(device)), dim=1)

        set_pixels = set_pixels[0].cpu().numpy()
        autocompleted_image = reverse_rle(set_pixels)
        
        image = convert_to_image(autocompleted_image)
        
        mse += np.mean((image.flatten() - target_image.flatten()).astype(np.float16)**2)*2
        break

    print(f"Average MSE: {mse / 10_000}")
    print(f"Average PSNR: {20 * np.log10(255) - 10 * np.log10(mse / 10_000)}")