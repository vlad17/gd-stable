"""
Basic MLP module, half stolen from Vitchyr's rlkit
https://github.com/vitchyr/rlkit
"""

from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with no bells or whistles."""

    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            hidden_activation=F.elu,
            output_activation=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = fc(h)
            h = self.hidden_activation(h)
        output = self.last_fc(h)
        if self.output_activation:
            output = self.output_activation(output)
        return output
