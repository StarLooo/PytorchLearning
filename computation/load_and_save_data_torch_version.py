# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


if __name__ == '__main__':
    x = torch.arange(4)
    torch.save(x, "x-file")
    x_load = torch.load("x-file")
    print(x_load == x)

    y = torch.zeros(4)
    torch.save([x, y], "(x,y)-files")
    x_load, y_load = torch.load("(x,y)-files")
    print((x_load, y_load))

    my_dict = {'x': x, 'y': y}
    torch.save(my_dict, "my_dict")
    my_dict_load = torch.load("my_dict")
    print(my_dict_load)

    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)
    torch.save(net.state_dict(), "mlp.params")

    clone = MLP()
    print(clone)
    clone.load_state_dict(torch.load('mlp.params'))
    clone.eval()
    Y_clone = clone(X)
    print(Y_clone == Y)
