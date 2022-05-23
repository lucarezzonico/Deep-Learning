import torch
import torch.nn as nn
from torch import optim
from Miniproject_1.other.net import *
# model.py will be imported by the testing pipeline

class Model():
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need

        self.model = Net()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model.to(self.device)

        self.learning_rate = 1e-4

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_model(self) -> None:
        ## This saves the parameters of the model into bestmodel.pth
        torch.save(self.model.state_dict(), 'Miniproject_1/bestmodel.pth')
        # print('model saved to bestmodel.pth')

    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        m_state_dict = torch.load('Miniproject_1/bestmodel.pth')
        self.model.load_state_dict(m_state_dict)
        # print('model loaded')

    def train(self, train_input, train_target, num_epochs=1) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.

        train_input, train_target = train_input.to(self.device), train_target.to(self.device)

        train_input = train_input.type(torch.FloatTensor).div(255)
        train_target = train_target.type(torch.FloatTensor).div(255)

        mini_batch_size = 20

        for e in range(num_epochs):
            for b in range(0, train_input.size(dim=0), mini_batch_size):
                output = self.model(train_input.narrow(dim=0, start=b, length=mini_batch_size)) # takes time
                loss = self.criterion(output, train_target.narrow(dim=0, start=b, length=mini_batch_size))
                self.optimizer.zero_grad()
                loss.backward() # takes time
                self.optimizer.step()
            print('epoch {:d}/{:d}'.format(e + 1, num_epochs), 'training loss = {:.2f}'.format(loss))
        pass

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)

        # predicted_output =  torch.empty_like(test_input)

        # test_output = test_input.view(test_input.size(dim=0), -1)
        # predicted_output = torch.argmax(nn.softmax(test_output).data, 1)

        test_input = test_input.to(self.device)

        test_input = test_input.type(torch.FloatTensor).div(255)

        predicted_output = self.model(test_input)

        predicted_output = predicted_output.cpu()

        return predicted_output
