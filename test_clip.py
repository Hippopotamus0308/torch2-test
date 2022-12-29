# Test clip model
import torch
import torchvision
import numpy as np
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# without torch.compile
def time_without_opt():
    times = []
    for i in range(10):
        outputs, time = timed(model(**inputs))
        times.append(time)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return times

# with torch.compile
def time_with_opt():
    times = []
    for i in range(10):
        opt_model = torch.compile(model)
        outputs, time = timed(opt_model(**inputs))
        times.append(time)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return times

def time_compare(former_time, opt_time):
    former_med = np.median(former_time)
    opt_med = np.median(opt_time)
    print(f"median time without opt: {former_med}")
    print(f"median time with opt: {opt_med}")
    speedup = former_med / opt_med
    print(f"speed-up: {speedup}")

def test():
    former = time_without_opt()
    torch._dynamo.reset()
    opt = time_with_opt()
    time_compare(former,opt)