import torch
from models import model
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from train import device, get_transform
print(f"device is {device}")

eval_transform = get_transform(train=True)

def predict(imgpath):
    img = read_image(imgpath)
    x = eval_transform(img)
    x = x[:3, ...].to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    model.to(device)
    predictions = model([x, ])
    pred = predictions[0]

if __name__ == "__main__":
    img = 'test.png'
    pred = predict(img)
    print(pred)

