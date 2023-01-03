import torch
import clip
from PIL import Image
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
opt_model = torch.compile(model)


def timed(model, printer):
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)
    #start.record()
    start = time.time()
    result = action(model)
    end = time.time()
    #end.record()
    torch.cuda.synchronize()
    #time_cnt = start.elapsed_time(end) / 1000
    time_cnt = end - start
    print(f"{printer}, time: {time_cnt}")
    return result, time_cnt

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)


def action(model_to_use):
    all_features = []
    all_labels = []   
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(cifar100, batch_size=100)):
            features = model_to_use.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)
    return all_features
    
    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    # for i in range(1000):
    #     image, class_id = cifar100[i]
    #     image_input = preprocess(image).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         image_features = model_to_use.encode_image(image_input)
    #         text_features = model_to_use.encode_text(text_inputs)
    # return text_inputs
    
def time_calculate():
    time_default = []
    time_opt = []

    for i in range(10):
        _, time = timed(model, "model")
        time_default.append(time)
    
    for i in range(10):
        _, time = timed(opt_model, "opt_model")
        time_opt.append(time)
    
    time_median_default = np.median(time_default)
    time_median_opt = np.median(time_opt)
    time_mean_default = np.mean(time_default)
    time_mean_opt = np.mean(time_opt)
    print(f"default model: {time_default}")
    print(f"opt model: {time_opt}")
    print("--------------")
    print(f"default model median time: {time_median_default}")
    print(f"opt model median time: {time_median_opt}")
    print("--------------")
    print(f"default model mean time: {time_mean_default}")
    print(f"opt model mean time: {time_mean_opt}")

time_calculate()