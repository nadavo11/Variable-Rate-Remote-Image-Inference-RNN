import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import models


def main(args):
    # Create the model directory if does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Normalize the input images
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    dataiter = iter(testloader)

    print('Loading Models')
    # Initialize the models
    model = models.setup(args)

    # Load the SAVED model
    # path_to_model = os.path.join(args.model_path, args.model+'_res-p%d_b%d-%d_%d.pkl' %
    #                              (args.patch_size, args.coded_size, args.load_epoch, args.load_iter))

    path_to_model = args.model_path
    model.load_state_dict(torch.load(path_to_model))

    print('Starting eval:::::::::::::::::')

    # Iterate through the dataset batches
    for i in range(args.num_samples//args.batch_size):
        # fixed from dataiter.next() ❌❌

        #load the data
        imgs, _ = next(dataiter)
        imsave(torchvision.utils.make_grid(imgs), 'prova_'+str(i))

        # divide the images to patches:
        patches = to_patches(imgs, args.patch_size)
        r_patches = []  # Reconstructed Patches

        # If the model is not a residual model, reset its state for a fresh start
        if args.residual is None:
            model.reset_state()

        # Process each patch through the model
        for p in patches:

            # If the model is a residual model, use the model's sampling method
            if args.residual:
                outputs = model.sample(Variable(p))

            # Otherwise, simply forward the patch through the model
            else:
                outputs = model(Variable(p))
            # Collect the model's output patches for reconstruction
            r_patches.append(outputs)
        # Transform the patches into the image
        outputs = reconstruct_patches(r_patches)
        imsave(torchvision.utils.make_grid(outputs), 'prova_'+str(i)+'_decoded')
        imshow(torchvision.utils.make_grid(outputs),"decoded")
        print(f'saved to {args.output_path}!')


def evaluate(model, imgs =None, dataiter = None, num_samples =20,batch_size=4):
    for i in range(num_samples//batch_size):
        print(i)
        #load the datasd
        if not imgs:
            imgs, _ = next(dataiter)
        imsave(torchvision.utils.make_grid(imgs), 'prova_'+str(i))

        # divide the images to patches:
        patches = to_patches(imgs, args.patch_size)
        r_patches = []  # Reconstructed Patches

        # If the model is not a residual model, reset its state for a fresh start
        if args.residual is None:
            model.reset_state()

        # Process each patch through the model
        for p in patches:

            # If the model is a residual model, use the model's sampling method
            if args.residual:
                outputs = model.sample(Variable(p))

            # Otherwise, simply forward the patch through the model
            else:
                outputs = model(Variable(p))
            # Collect the model's output patches for reconstruction
            r_patches.append(outputs)
        # Transform the patches into the image
        outputs = reconstruct_patches(r_patches)
        imsave(torchvision.utils.make_grid(outputs), 'prova_'+str(i)+'_decoded')


#==============================================
# - CUSTOM FUNCTIONS
#==============================================

def imsave(img, name):
    img = img / 2 + 0.5     # unnormalize
    saving_path = os.path.join(args.output_path, name+'.png')
    print(f"saving to {saving_path }")
    torchvision.utils.save_image(img, saving_path)

def imshow(img, name):
    img = img / 2 + 0.5     # unnormalize
    img = np.transpose(img,(1,2,0))
    plt.figure()
    plt.imshow(img)
    plt.show()

def to_patches(x, patch_size):
    """
    Splits an input tensor into square patches of a specified size.

    :param x: (Tensor) The input tensor to be split into patches. Expected shape is (N, C, H, W).
    :param patch_size:(int) The size of one side of the square patches to be extracted.
    :return:patches - List[Tensor]: A list of the extracted patches, each of which is a tensor with shape (N, C, patch_size, patch_size).
    """
    # Calculate the number of patches along one dimension
    num_patches_x = 32//patch_size

    patches = []

    # Iterate over each patch position in the input tensor
    for i in range(num_patches_x):
        for j in range(num_patches_x):

            # Extract the patch
            patch = x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

            # Ensure the patch is contiguous in memory and add it to the list
            patches.append(patch.contiguous())
    return patches


def reconstruct_patches(patches):
    """
    :param patches:
    :return:
    """
    batch_size = patches[0].size(0)
    patch_size = patches[0].size(2)
    num_patches_x = 32//patch_size
    reconstructed = torch.zeros(batch_size, 3, 32, 32)
    p = 0
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            reconstructed[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[p].data
            p += 1
    return reconstructed


#=============================================================================
# - PARAMETERS
#=============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==================================================================================================================
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='fc',
                        help='name of the model to be used: fc, conv, lstm ')
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

    # ==================================================================================================================
    # OPTIMIZATION
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='number of iterations where the system sees all the data')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)

    # ==================================================================================================================
    # SAVING & PRINTING
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model_path', type=str, default='./saved_models/main_model.pkl',
                        help='path were the models should be saved')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing the log info')
    parser.add_argument('--save_step', type=int, default=5000,
                        help='step size for saving the trained models')
    parser.add_argument('--output_path', type=str, default='./test_imgs/')

    parser.add_argument('--load_iter', type=int, default=100,
                        help='iteration which the model to be loaded was saved')
    parser.add_argument('--load_epoch', type=int, default=1,
                        help='epoch in which the model to be loaded was saved')


    parser.add_argument('--num_samples', type=int, default=20,
                        help='number of pictures to be plotted')
    # __________________________________________________________________________________________________________________
    args = parser.parse_args()
    print(args)
    main(args)

#%%
