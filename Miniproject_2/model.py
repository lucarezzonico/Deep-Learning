import torch
import torch.nn as nn
from torch import optim
# from Miniproject_2.other.net import *
from Miniproject_2.other.modules import Conv2d, NearestUpsampling, ReLU, Sigmoid, Sequential, MSE, SGD
# model.py will be imported by the testing pipeline


class Model():
    def __init__(self, lr=10) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need

        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=128, kernel_size=2, stride=2, padding=0),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            ReLU(),
            NearestUpsampling(scale_factor=2),
            Conv2d(in_channels=256, out_channels=128, kernel_size=2, stride=1, padding=1),
            ReLU(),
            NearestUpsampling(scale_factor=2),
            Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=0),
            Sigmoid()
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # self.model.to(self.device)

        self.learning_rate = lr

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), lr=self.learning_rate)

    def save_model(self, path='Miniproject_2/bestmodel.pth') -> None:
        ## This saves the parameters of the model into bestmodel.pth
        torch.save(self.model.param(), path)
        # print('model saved to bestmodel.pth')

    def load_pretrained_model(self, path='Miniproject_2/bestmodel.pth') -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        m_state_dict = torch.load(path)

        # print('m_state_dict : ', m_state_dict[1][0])

        self.model.blocks[0].weight.data = m_state_dict[0][0].to(self.device)
        self.model.blocks[0].d_weight.data = m_state_dict[0][1].to(self.device)
        self.model.blocks[0].bias.data = m_state_dict[1][0].to(self.device)
        self.model.blocks[0].d_bias.data = m_state_dict[1][1].to(self.device)
        self.model.blocks[2].weight.data = m_state_dict[2][0].to(self.device)
        self.model.blocks[2].d_weight.data = m_state_dict[2][1].to(self.device)
        self.model.blocks[2].bias.data = m_state_dict[3][0].to(self.device)
        self.model.blocks[2].d_bias.data = m_state_dict[3][1].to(self.device)
        self.model.blocks[5].weight.data = m_state_dict[4][0].to(self.device)
        self.model.blocks[5].d_weight.data = m_state_dict[4][1].to(self.device)
        self.model.blocks[5].bias.data = m_state_dict[5][0].to(self.device)
        self.model.blocks[5].d_bias.data = m_state_dict[5][1].to(self.device)
        self.model.blocks[8].weight.data = m_state_dict[6][0].to(self.device)
        self.model.blocks[8].d_weight.data = m_state_dict[6][1].to(self.device)
        self.model.blocks[8].bias.data = m_state_dict[7][0].to(self.device)
        self.model.blocks[8].d_bias.data = m_state_dict[7][1].to(self.device)

    # print('model loaded')

    def train(self, train_input, train_target, num_epochs=7, mini_batch_size=4, scheduler_gamma=1, lambda_l2=0) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.

        torch.set_grad_enabled(False)

        train_input, train_target = train_input.to(self.device), train_target.to(self.device)

        train_input = train_input.float().div(255)
        train_target = train_target.float().div(255)

        for e in range(num_epochs):

            # self.learning_rate = self.learning_rate * scheduler_gamma
            # self.optimizer = SGD(self.model.param(), lr=self.learning_rate)

            for b in range(0, train_input.size(dim=0), mini_batch_size):
                self.optimizer.zero_grad()

                # forward pass
                output = self.model.forward(train_input.narrow(dim=0, start=b, length=mini_batch_size)) # takes time
                loss = self.criterion.forward(output, train_target.narrow(dim=0, start=b, length=mini_batch_size))

                # # L2 penalty term (to avoid overfitting the training data for an increasing number of epochs)
                # for p in self.model.param():
                #     loss += lambda_l2 * ((p[0]**2).sum() + (p[1]**2).sum())

                # backward pass
                d_loss_d_y = self.criterion.backward().to(self.device)
                self.model.backward(d_loss_d_y)

                self.optimizer.step()
            print('epoch {:d}/{:d}'.format(e + 1, num_epochs), 'training loss = {:.5f}'.format(loss))

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)

        test_input = test_input.to(self.device)

        test_input = test_input.float().div(255)
        predicted_output = self.model.forward(test_input)
        predicted_output = predicted_output.mul(255).to(torch.uint8)

        return predicted_output
