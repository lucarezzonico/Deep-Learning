import torch
import torch.nn as nn
from torch import optim
# from Miniproject_2.other.net import *
from Miniproject_2.other.modules import Conv2d, TransposeConv2d, NearestUpsampling, ReLU, Sigmoid, Sequential, MSE, SGD
# model.py will be imported by the testing pipeline

torch.set_grad_enabled(False)

class Model():
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need

        # self.model = Net()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=128, kernel_size=2, stride=2, padding=0),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=0),
            ReLU(),
            TransposeConv2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=0),
            ReLU(),
            TransposeConv2d(in_channels=128, out_channels=3, kernel_size=2, stride=2, padding=0),
            Sigmoid()
        )

        # self.model = Sequential(
        #     Conv2d(in_channels=3, out_channels=128, kernel_size=2, stride=2, padding=0),
        #     ReLU(),
        #     Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=0),
        #     ReLU(),
        #     NearestUpsampling(scale_factor=2),
        #     Conv2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=0),
        #     ReLU(),
        #     NearestUpsampling(scale_factor=2),
        #     Conv2d(in_channels=128, out_channels=3, kernel_size=2, stride=2, padding=0),
        #     Sigmoid()
        # )

        self.learning_rate = 10

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), lr=self.learning_rate)

    def save_model(self) -> None:
        ## This saves the parameters of the model into bestmodel.pth
        torch.save(self.model.param(), 'Miniproject_2/bestmodel.pth')
        # print('model saved to bestmodel.pth')

    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        m_state_dict = torch.load('Miniproject_2/bestmodel.pth')

        # print('m_state_dict : ', m_state_dict[1][0])

        # self.model.blocks[0].weight.data = m_state_dict[0][0]
        # self.model.blocks[0].d_weight.data = m_state_dict[0][1]
        # self.model.blocks[0].bias.data = m_state_dict[1][0]
        # self.model.blocks[0].d_bias.data = m_state_dict[1][1]
        # self.model.blocks[2].weight.data = m_state_dict[2][0]
        # self.model.blocks[2].d_weight.data = m_state_dict[2][1]
        # self.model.blocks[2].bias.data = m_state_dict[3][0]
        # self.model.blocks[2].d_bias.data = m_state_dict[3][1]
        # self.model.blocks[5].weight.data = m_state_dict[4][0]
        # self.model.blocks[5].d_weight.data = m_state_dict[4][1]
        # self.model.blocks[5].bias.data = m_state_dict[5][0]
        # self.model.blocks[5].d_bias.data = m_state_dict[5][1]
        # self.model.blocks[8].weight.data = m_state_dict[6][0]
        # self.model.blocks[8].d_weight.data = m_state_dict[6][1]
        # self.model.blocks[8].bias.data = m_state_dict[7][0]
        # self.model.blocks[8].d_bias.data = m_state_dict[7][1]

        for i in range(0, len(self.model.blocks), 2):
            self.model.blocks[i].weight.data = m_state_dict[i][0]
            self.model.blocks[i].d_weight.data = m_state_dict[i][1]
            self.model.blocks[i].bias.data = m_state_dict[i+1][0]
            self.model.blocks[i].d_bias.data = m_state_dict[i+1][1]
        # print('model loaded')

    def train(self, train_input, train_target, num_epochs=1) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.

        train_input = train_input.type(torch.FloatTensor).div(255)
        train_target = train_target.type(torch.FloatTensor).div(255)

        mini_batch_size = 20

        for e in range(num_epochs):
            for b in range(0, train_input.size(dim=0), mini_batch_size):
                print('batch {:d}'.format(b))
                self.optimizer.zero_grad()
                # forward pass
                output = self.model.forward(train_input.narrow(dim=0, start=b, length=mini_batch_size)) # takes time
                loss = self.criterion.forward(output, train_target.narrow(dim=0, start=b, length=mini_batch_size))

                # backward pass
                d_loss_d_y = self.criterion.backward()
                self.model.backward(d_loss_d_y)

                self.optimizer.step()
            print('epoch {:d}/{:d}'.format(e + 1, num_epochs), 'training loss = {:.2f}'.format(loss))

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)

        # predicted_output =  torch.empty_like(test_input)

        # test_output = test_input.view(test_input.size(dim=0), -1)
        # predicted_output = torch.argmax(nn.softmax(test_output).data, 1)

        test_input = test_input.type(torch.FloatTensor).div(255)

        predicted_output = self.model.forward(test_input)

        return predicted_output
