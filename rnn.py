import torch.nn as nn
import torch


class RNN(nn.Module):
    """
    Recurrent neural network for predicting the country of a city based on the
    ascii respresentation of a city's name. See rnn.ipynb.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.nl = nn.Tanh()

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.init_weights()

    def forward(self, input, hidden):
        hidden = self.nl(self.i2h(input) + self.h2h(hidden))
        scores = self.h2o(hidden)
        log_probs = self.log_softmax(scores)

        return log_probs, hidden

    def init_weights(self):
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2h.weight)
        nn.init.xavier_uniform_(self.h2o.weight)

    def get_init_hidden(self):
        return torch.zeros((1, self.hidden_size))