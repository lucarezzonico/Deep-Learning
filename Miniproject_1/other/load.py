import torch
from torch.utils.data import (DataLoader,)  # Gives easier dataset managment and creates mini batches
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import matplotlib.pyplot as plt

noisy_imgs_train_1, noisy_imgs_train_2 = torch.load('../../miniproject_dataset/train_data.pkl')
noisy_imgs_valid, clean_imgs_valid = torch.load('../../miniproject_dataset/val_data.pkl')

print('noisy_imgs_train_1', noisy_imgs_train_1.size(), 'noisy_imgs_train_2', noisy_imgs_train_2.size())
print('noisy_imgs_valid', noisy_imgs_valid.size(), 'clean_imgs_valid', clean_imgs_valid.size())

def plot_images(img):
    for i in range(img.size(dim=0)):
        plt.imshow(img[i, :, :, :].permute((1, 2, 0)), interpolation='spline16')
        plt.show()

def plot_pair_of_images(img1, img2):
    _, axes = plt.subplots(1, 2)
    for i in range(img1.size(dim=0)):
        axes[0].imshow(img1[i, :, :, :].permute((1, 2, 0)), interpolation='spline16')
        axes[1].imshow(img2[i, :, :, :].permute((1, 2, 0)), interpolation='spline16')
        plt.show()

plot_pair_of_images(noisy_imgs_train_1[0:4, :, :, :], noisy_imgs_train_2[0:4, :, :, :])
plot_pair_of_images(noisy_imgs_valid[0:4, :, :, :], clean_imgs_valid[0:4, :, :, :])


# transform data
my_transforms = transforms.Compose(
    [   # Compose makes it possible to have many transforms
        # transforms.ToPILImage(),
        transforms.Resize((36, 36)),  # Resizes (32,32) to (36,36)
        transforms.RandomCrop((32, 32)),  # Takes a random (32,32) crop
        transforms.ColorJitter(brightness=0.5),  # Change brightness of image
        transforms.RandomRotation(degrees=45),  # Perhaps a random rotation from -45 to 45 degrees
        transforms.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with probability 0.5
        transforms.RandomVerticalFlip(p=0.05),  # Flips image vertically with probability 0.05
        transforms.RandomGrayscale(p=0.2),  # Converts to grayscale with probability 0.2
        # transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Note: these values aren't optimal
    ]
)

transformed_imgs = my_transforms(noisy_imgs_train_1[0:4, :, :, :])
# plot_images(transformed_imgs)

