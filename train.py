import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import argparse
import math
import time
import os
import numpy as np
from torch.utils.data import Subset, DataLoader

#from IPython.display import display, clear_output
import threading
import time

import models

oneway_models = ['fc', 'conv', 'lstm']
residual_models = ['fc_res', 'conv_res', 'lstm_res']
mix_models = ['lstm_mix']

def compare_reconstructed_patch(target_tensor,reconstructed_patches):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns, and optionally set a figure size

    # Plot target patch
    target_img = target_tensor[0].permute(1, 2, 0).detach().numpy()  # Convert the first patch to numpy array
    axs[0].imshow(target_img)
    axs[0].set_title("Target Patch")
    axs[0].axis('off')  # Optionally remove the axis

    # Plot reconstructed patch
    reconstructed_img = reconstructed_patches[0].permute(1, 2,
                                                         0).detach().numpy()  # Convert the first reconstructed patch to numpy array
    axs[1].imshow(reconstructed_img)
    axs[1].set_title("Reconstructed")
    axs[1].axis('off')  # Optionally remove the axis

    plt.show()
def init_loss_plot():

    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], 'r-', label='Training Loss')  # Initialize an empty line
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    return ax,line

def plot_losses(losses,ax,line):
    ax.set_xlim(0, len(losses))  # Set the x-axis limits
    ax.set_ylim(min(losses), max(losses) + 0.1 * (max(losses) - min(losses))+1e-6)  # Adjust the y-axis limits dynamically

    line.set_data(range(len(losses)), losses)  # Update the line data
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot


def evaluate(model, data_loader, criterion):
    """

    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation
        for data in data_loader:
            imgs = data[0]
            # Depending on your model, you might need targets here as well
            patches = to_patches(imgs, args.patch_size)

            # Assuming a similar processing as in training but without backpropagation
            for patch in patches:
                # Transform the tensor into Variable, if necessary
                v_patch = Variable(patch)
                reconstructed_patches = model(v_patch)
                loss = criterion(reconstructed_patches, v_patch)  # Adapt based on your loss calculation
                total_loss += loss.item() * v_patch.size(0)
                total_samples += v_patch.size(0)

    average_loss = total_loss / total_samples
    return average_loss

def main(args):
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.panel import Panel

    console = Console()

    seed = 42
    torch.manual_seed(seed)

    # Create the model directory if does not exist
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Normalize function so given an image of range [0, 1] transforms it into a Tensor range [-1. 1]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR LOADER
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # Generate random indices to select a subset
    indices = torch.randperm(len(trainset)).tolist()[:10000]
    # Create a smaller dataset from the full dataset
    trainset_small = Subset(trainset, indices)



    """________________________________________
     work on a smaller dataset to test the model
    """
    # Create a DataLoader for the smaller dataset
    train_loader_small = DataLoader(trainset_small, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # show an image from the dataset
    # import matplotlib.pyplot as plt

    # Load the model:
    model = models.setup(args)

    if args.from_pretrained:
        pretrained = args.from_pretrained
        model.load_state_dict(torch.load(pretrained))


    # Define the LOSS and the OPTIMIZER
    criterion = nn.MSELoss()
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # ::::::::::::::::::::::::::::::::
    #   TRAIN----------------------
    # ::::::::::::::::::::::::::::::::

    # change for trainloader size:
    num_steps = len(train_loader)
    start = time.time()
    total_losses = []
    # Divide the input 32x32 images into num_patches patch_sizexpatch_size patchs
    num_patches = (32//args.patch_size)**2

    current_losses = []
    ax, line = init_loss_plot()

    for epoch in range(args.num_epochs):

        running_loss = 0.0


        for i, data in enumerate(train_loader, 0):

            # Get the images
            imgs = data[0]
            # Transform into patches
            patches = to_patches(imgs, args.patch_size)

            # TODO: Do this thing more polite!! :S
            if args.model in oneway_models:
                for patch in patches:
                    # Transform the tensor into Variable
                    v_patch = Variable(patch)
                    target_tensor = Variable(torch.zeros(v_patch.size()), requires_grad=False)
                    losses = []
                    # Set gradients to Zero
                    optimizer.zero_grad()
                    reconstructed_patches = model(v_patch)
                    loss = criterion(reconstructed_patches, target_tensor)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            elif args.model in residual_models:
                for patch in patches:
                    # Transform the tensor into Variable
                    v_patch = Variable(patch)
                    target_tensor = Variable(torch.zeros(v_patch.size()), requires_grad=False) #⁉️⁉️
                    # target_tensor = Variable(patch, requires_grad=True)
                    losses = []
                    # Set gradients to Zero
                    optimizer.zero_grad()

                    for p in range(args.num_passes):
                        # Forward + Backward + Optimize
                        reconstructed_patches = model(v_patch, p)

                        # losses.append(criterion(reconstructed_patches, target_tensor)) #⁉️⁉️
                        losses.append(criterion(reconstructed_patches, target_tensor) / args.num_passes)

                        v_patch = reconstructed_patches

                    reconstructed_image_patch = model.sample(input_patch=patch) # DISPLAY IMAGE

                # compare_reconstructed_patch(patch, reconstructed_image_patch)

                    loss = sum(losses)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.data

            else:
                model.reset_state()
                losses = []
                # Set gradients to Zero
                optimizer.zero_grad()
                for patch in patches:
                    # Transform the tensor into Variable
                    v_patch = Variable(patch)
                    reconstructed_patches = model(v_patch)
                    current_loss = criterion(reconstructed_patches, v_patch)
                    losses.append(current_loss)
                loss = sum(losses)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]

            # STATISTICS:
##
            if (i+1) % args.log_step == 0:

                loss_value = running_loss / args.log_step / num_patches
                if loss_value < 1e-5:
                    loss_str = f"{loss_value:.1e}"  # Scientific notation for very small numbers
                else:
                    loss_str = f"{loss_value:.4f}"  # Fixed-point notation for larger numbers

                panel = Panel(f"[bold green]Step: {i + 1}/{num_steps}\n"
                              f"[bold green]Epoch: {epoch+1}/{args.num_epochs}\n"
                              f"[bold yellow]Loss: {loss_str}\n"
                              f"[bold cyan]Time: {timeSince(start, ((epoch * num_steps + i + 1.0) / (args.num_epochs * num_steps))):s}"
                              ,
                              title="[bold cyan]Training Progress",

                              expand=False,
                              padding=(1, 8))
                console.log(panel)

                current_losses.append(running_loss/args.log_step/num_patches)
                plot_losses(current_losses, ax, line)
                running_loss = 0.0


            # SAVE:
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.model_path, args.model+'-p%d_b%d-%d_%d.pkl' %
                                        (args.patch_size, args.coded_size, epoch + 1, i + 1)))


        total_losses.append(current_losses)
        torch.save(model.state_dict(),
                   os.path.join(args.model_path,
                                args.model + '-p%d_b%d-%d_%d.pkl' % (args.patch_size, args.coded_size, epoch + 1, i + 1)))
    plt.plot(current_losses)
    plt.show()
    print('__TRAINING DONE=================================================')


#==============================================
# - CUSTOM FUNCTIONS
#==============================================

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (ETA: %s)' % (asMinutes(s), asMinutes(rs))


def to_patches(x, patch_size):
    num_patches_x = 32//patch_size
    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            patch = x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.contiguous())
    return patches


"""
#=============================================================================
# - PARAMETERS
#=============================================================================
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==================================================================================================================
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='fc',
                        help='name of the model to be used: fc, fc_rec, conv, conv_rec, lstm ')
    parser.add_argument('--residual', type=bool, default=False,
                        help='Set True if the model is residual, otherwise False')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size')
    parser.add_argument('--coded_size', type=int, default=4,
                        help='number of bits representing the encoded patch')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='size for the encoded subdivision of the input image')
    parser.add_argument('--num_passes', type=int, default=16,
                        help='number of passes for recursive architectures')
    parser.add_argument('--from_pretrained', type=str, default='',
                        help='specify the path of a pretrained model to further train')

    # ==================================================================================================================
    # OPTIMIZATION
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='number of iterations where the system sees all the data')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help= 'default is ')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)

    # ==================================================================================================================
    # SAVING & PRINTING
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model_path', type=str, default='./saved_models/',
                        help='path were the models should be saved')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing the log info')
    parser.add_argument('--save_step', type=int, default=5000,
                        help='step size for saving the trained models')


    #__________________________________________________________________________________________________________________
    args = parser.parse_args()
    print(args)
    main(args)

#%%
