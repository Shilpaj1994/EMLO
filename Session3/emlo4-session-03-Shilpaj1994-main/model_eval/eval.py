import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model import Net


def test_epoch(model, device, data_loader):
    # write code to test this epoch
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    out = {"Test loss": test_loss, "Accuracy": accuracy}
    print(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # create MNIST test dataset and loader
    test_data = datasets.MNIST('/opt/mount/data/', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    # create model and load state dict
    model = Net()
    model.load_state_dict(torch.load("/opt/mount/model/mnist_cnn.pt"))

    # test epoch function call
    eval_results = test_epoch(model, device, test_loader)

    with (Path(args.save_dir) / "eval_results.json").open("w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
