"""
Implementation of recurrent neural network for 1-D learning.

by Christopher M. Sandino (sandino@stanford.edu), 2020.

"""

import torch
from torch import nn

class RNN(nn.Module):
    """
    Prototype for long short-term memory (LSTM) network.
    """

    def __init__(self, in_chans, hidden_size, num_layers, bidirectional=True):
        """

        """
        super().__init__()

        num_directions = 2 if bidirectional is True else 1

        self.rnn = nn.LSTM(
            input_size=in_chans,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # So that that the output has the same number of channels as the input
        # TODO: make this multi-layer?
        self.resample = nn.Linear(hidden_size*num_directions, in_chans)


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, time, in_chans]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, time, in_chans]
        """

        dims = tuple(input.size())  # [N, H, C]

        # Convert input to real representation
        input = torch.view_as_real(input)
        input = input.reshape(dims[0], dims[1], dims[2]*2)

        # Forward pass through the network
        input, _ = self.rnn(input, None)  # None represents zero initial hidden state

        # Squash channel dimension back to original number of channels
        input = self.resample(input)

        # Convert back to complex representation
        input = input.reshape(dims[0], dims[1], dims[2], 2)
        input = torch.view_as_complex(input)

        return input