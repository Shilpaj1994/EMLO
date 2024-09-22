import json
import time
import random
import torch
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from model import Net


def infer(model, dataset, save_dir, num_samples=5):
    model.eval()
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image, _ = dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True).item()

        img = Image.fromarray(image.squeeze().numpy() * 255).convert("L")
        img.save(results_dir / f"{pred}.png")


def main():
    save_dir = "./"
    
    # init model and load checkpoint here
    model = Net()
    model.load_state_dict(torch.load("/opt/mount/model/mnist_cnn.pt"))

	# create transforms and test dataset for mnist
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    train_data = datasets.MNIST('/opt/mount/data', train=True, download=True, transform=train_transforms)

    infer(model, train_data, save_dir)
    print("Inference completed. Results saved in the 'results' folder.")


if __name__ == "__main__":
    main()
