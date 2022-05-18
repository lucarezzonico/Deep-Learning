import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, nb_input=3, nb_output=3):
        # initialize deep-learning network
        super(Net, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels=nb_input, out_channels=48, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1))
            # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1))
            # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1))
            # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(in_channels=96 + nb_input, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=nb_output, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.1))

        # Initialize weights
        # self._init_weights()

    # def _init_weights(self):
    #     """Initializes weights using He et al. (2015)."""
    #
    #     for m in self.modules():
    #         if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data)
    #             m.bias.data.zero_()

    def forward(self, input):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(input)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, input), dim=1)
        output = self._block6(concat1)

        # Final activation
        return output

class Net2(nn.Module):
    def __init__(self, nb_input=3, nb_output=3):
        # initialize deep-learning network
        super(Net2, self).__init__()

        self.model = nn.Sequential(
            # encoder
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=4, stride=1, padding=0),
            # decoder
            nn.ConvTranspose2d(8, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=0)
        )

    def forward(self, input):
        return self.model(input)

class Net3(nn.Module):
    def __init__(self, nb_input=3, nb_output=3):
        # initialize deep-learning network
        super(Net3, self).__init__()

        self.model = nn.Sequential(
            # encoder
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=4, stride=1, padding=0),
            # decoder
            nn.ConvTranspose2d(8, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=0)
        )

    def forward(self, input):
        return self.model(input)