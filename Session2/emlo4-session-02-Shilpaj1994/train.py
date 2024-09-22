import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Net(torch.nn.Module):
    def __init__(self):
        """
        Constructor
        """
        super(Net, self).__init__()

        # Define model architecture here
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Forward pass for model training
        :param x: Input layer
        :return: Output of the model
        """
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # Implement the training loop here
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f"    [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]    Loss: {loss.item():.6f}")
            if args.dry_run:
                break


def test_epoch(model, device, data_loader):
    # Implement the testing loop here
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
    print(f"    Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%)")


def save_checkpoint(model, optimizer, epoch, file_path='model_checkpoint.pth'):
    """
    Function to save the model checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch} to {file_path}")



def load_checkpoint(model, optimizer, checkpoint_path="model_checkpoint.pth"):
    """
    Load model checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    print(f"Keys available in the saved model: {checkpoint.keys()}")
    
    # Check that the required keys exist in the checkpoint
    if 'model_state_dict' not in checkpoint or 'optimizer_state_dict' not in checkpoint or 'epoch' not in checkpoint:
        raise KeyError("The checkpoint does not contain the required keys: 'model_state_dict', 'optimizer_state_dict', and 'epoch'")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    print(f"Checkpoint trained till {epoch} epochs. Starting next epoch")
    return epoch + 1


def main():
    # Parser to get command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    
    # Define your command line arguments here
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from previous checkpoint')
    
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the MNIST dataset for training and testing
    print("Downloading datasets....")
    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1325,), (0.3104,))
        ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    
    print("Loading Model")
    model = Net().to(device)

    # Choose and define the optimizer here
    # Optimization algorithm to update the weights
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Add a way to load the model checkpoint if 'resume' argument is True
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer)

        # Ignore the training if already trained for specified epochs
        if (start_epoch >= args.epochs):
            print(f"Model already trained for {args.epochs}. Skipping the model training")
            exit(0)

    # Scheduler to change the learning rate after specific number of epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
    
    # Implement the training and testing cycles
    print(f"Training Model for {args.epochs} epochs")

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        print("    ")
        print(f"Epoch {epoch}:")
        print(f"    Training step")
        train_epoch(epoch, args, model, device, train_loader, optimizer)

        print(f"    Test step")
        test_epoch(model, device, test_loader)

        # Save the model after each epoch
        print(f"Saving Checkpoint")
        save_checkpoint(model, optimizer, epoch)

        # Update the learning rate
        scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), "mnist.pth")


if __name__ == "__main__":
    main()
