import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Using LSTM instead of GRU
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden_cell):
        # hidden_cell is a tuple (h, c)
        embedded = self.embedding(input).view(1, 1, -1)
        output, next_hidden_cell = self.lstm(embedded, hidden_cell)
        return output, next_hidden_cell

    def initHidden(self):
        # LSTM needs two tensors: hidden state and cell state
        h0 = torch.zeros(1, 1, self.hidden_size, device=device)
        c0 = torch.zeros(1, 1, self.hidden_size, device=device)
        return (h0, c0)

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_cell):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, next_hidden_cell = self.lstm(output, hidden_cell)
        output = self.softmax(self.out(output[0]))
        return output, next_hidden_cell
