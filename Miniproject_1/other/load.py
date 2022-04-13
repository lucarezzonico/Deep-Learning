import torch
import matplotlib.pyplot as plt

noisy_imgs_train_1, noisy_imgs_train_2 = torch.load('../../miniproject_dataset/train_data.pkl')
noisy_imgs_valid, clean_imgs_valid = torch.load('../../miniproject_dataset/val_data.pkl')

print('noisy_imgs_train_1', noisy_imgs_train_1.size(), 'noisy_imgs_train_2', noisy_imgs_train_2.size())
print('noisy_imgs_valid', noisy_imgs_valid.size(), 'clean_imgs_valid', clean_imgs_valid.size())

def plot_noisy_images(noisy_imgs_train_1, noisy_imgs_train_2):
    _, axes = plt.subplots(noisy_imgs_train_1.size(dim=0), 2)
    for i in range(noisy_imgs_train_1.size(dim=0)):
        axes[i,0].imshow(noisy_imgs_train_1[i, :, :, :].permute((1, 2, 0)), interpolation='spline16')
        axes[i,1].imshow(noisy_imgs_train_2[i, :, :, :].permute((1, 2, 0)), interpolation='spline16')
    plt.show()

plot_noisy_images(noisy_imgs_train_1[0:2, :, :, :], noisy_imgs_train_2[0:2, :, :, :])
plot_noisy_images(noisy_imgs_valid[0:2, :, :, :], clean_imgs_valid[0:2, :, :, :])
