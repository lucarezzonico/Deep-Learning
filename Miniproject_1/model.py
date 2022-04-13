import torch
import torch.nn as nn
from torch import optim
# model.py will be imported by the testing pipeline

class Model():
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.model = nn.Sequential(
                         nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0, dilation=1),  # w1 = (32 x 3 x 5 x 5), b1 = (32 x 1)
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.ReLU(),
                         nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1),
                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0, dilation=1),  # w2 = (64 x 32 x 5 x 5), b2 = (64 x 1)
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.ReLU(),
                         nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
                         nn.Linear(in_features=64 * 5 * 5, out_features=1200),
                         nn.Linear(in_features=1200, out_features=n_output)
                     )

    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model

        # torch.save(m.state_dict(), 'bestmodel.pth')
        m_state_dict = torch.load('bestmodel.pth')
        new_m = Model()
        new_m.load_state_dict(m_state_dict)

        pass

    def train(self, train_input, train_target) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.

        learning_rate = 1e-1
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        nb_epochs = 250
        mini_batch_size = 200

        for e in range(nb_epochs):
            for b in range(0, train_input.size(dim=0), mini_batch_size):
                output = self.model(train_input.narrow(dim=0, start=b, length=mini_batch_size))
                loss = self.criterion(output, train_target.narrow(dim=0, start=b, length=mini_batch_size))
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            # print(e, loss)
        pass

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)

        # predicted_output =  torch.empty_like(test_input)

        test_output = test_input.view(test_input.size(dim=0), -1)
        predicted_output = torch.argmax(nn.softmax(test_output).data, 1)

        return predicted_output
