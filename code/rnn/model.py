import torch
import torch.nn as nn
from torch.autograd import Variable


# don't touch this file... no reason to.

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()

    def forward(self, input, hidden):
        # print(input)
        # print("~~~~~")
        # print(hidden)
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.rand(1, self.hidden_size))

    def evaluate(self, line_tensor):
        hidden = self.init_hidden()
        output, hidden = self(line_tensor, hidden)
        
        return output

    def predict(self, line_tensor):
        output = self.evaluate(line_tensor)
        topv, topi = output.data.topk(1, 1, True)
        return topi
        
