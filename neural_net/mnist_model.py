import torch.nn as nn


class Model(nn.Module):

    def __init__(self):

        super().__init__()
        self.input_size = 28 * 28  # TODO: Check image size instead of hardcoding
        self.ndim = 10  # TODO: Check number of labels
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.ndim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
