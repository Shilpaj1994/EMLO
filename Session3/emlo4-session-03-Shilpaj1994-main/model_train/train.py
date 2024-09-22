import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch.multiprocessing as mp
from model import Net


# Train one epoch
def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()  # Set model to training mode
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f"Process {pid} [{batch_idx * len(data)}/{len(data_loader.dataset)} "
                  f"({100. * batch_idx / len(data_loader):.0f}%)]  Loss: {loss.item():.6f}")
            if args.dry_run:
                break


# Train function for each process
def train(rank, args, model, dataset, dataloader_kwargs, device):
    # Set a different seed for each process to ensure they don't get the same batches
    torch.manual_seed(args.seed + rank)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    # Define optimizer (SGD)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train for the given number of epochs
    for epoch in range(1, args.epochs + 1):
        print(f"Process {rank}, Epoch {epoch} starting")
        train_epoch(epoch, args, model, device, train_loader, optimizer)


def main():
    parser = argparse.ArgumentParser(description="MNIST Training Script")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        metavar="N",
        help="how many training processes to use (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        metavar='N',
        help='number of epochs to train (default: 1)'
    )
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    mp.set_start_method('spawn', force=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Selected Device: {device}")

    # create model and setup mp
    model = Net().to(device)
    print("Model loaded")
    
    model.share_memory()
    print("Shared memory of the model")

    # If checkpoint, resume training (model_checkpoint.pth)
    # If trained model, skip training (mnist_cnn.pt)
    if os.path.isfile("./model/mnist_cnn.pt"):
        model.load_state_dict(torch.load("./model/mnist_cnn.pt"))
        print("Loaded the model from mnist_cnn.pt Skipping Training!")
    else:
        print("mnist_cnn.pt not found, training model...")
        if not os.path.isfile("./model/model_checkpoint.pth"):
            print("No checkpoint found to resume from.")
        else:
            print("Resuming training from checkpoint...")
            model.load_state_dict(torch.load("./model/model_checkpoint.pth"))

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }

    # create mnist train dataset
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    print("Training Dataset created")

    # mnist hogwild training process
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model,
                                           train_data, kwargs, device))
        p.start()                   # We first train the model across `num_processes` processes
        processes.append(p)
    print("Processes Created")

    for p in processes:
        p.join()

    # save model ckpt
    model_dir = './model/'

    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # File location
    file_location = os.path.join(model_dir, "mnist_cnn.pt")
    torch.save(model.state_dict(), file_location)

if __name__ == "__main__":
    main()
