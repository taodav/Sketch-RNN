from torch import nn


class SketchRNN(nn.Module):
    def __init__(self, hidden_size, bidirectional=True):
        super(SketchRNN, self).__init__()

        self.hidden_size = hidden_size
        enc_size = self.hidden_size // 2 if bidirectional else hidden_size

        self.encoder = nn.GRU(2, enc_size, bidirectional=bidirectional)
        self.std_dev = nn.Linear(self.hidden_size, 1)
        self.mean = nn.Linear(self.hidden_size, 1)


class SketchRNNDecoderStep(nn.Module):
    def __init__(self, hidden_size):
        super(SketchRNNDecoderStep, self).__init__()

        self.hidden_size = hidden_size
