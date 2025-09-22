import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.z_config import Config

config = Config()


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 65)

    def forward(self, batch, device):
        bert_reps = batch['bert']
        binder_reps = batch['binder']

        output = self.linear(bert_reps)
        output_from_0_to_1 = torch.sigmoid(output)
        output_from_0_to_6 = output_from_0_to_1 * 6

        loss = F.mse_loss(output_from_0_to_6, binder_reps)
        return loss

    def forward_eval(self, batch, device):
        bert_rep = batch['bert']
        binder_rep = batch['binder']

        with torch.no_grad():
            output = self.linear(bert_rep)
            output_from_0_to_1 = torch.sigmoid(output)
            output_from_0_to_6 = output_from_0_to_1 * 6

            loss = F.mse_loss(output_from_0_to_6, binder_rep)
            return loss.item()

    def forward_rep(self, batch, device):
        with torch.no_grad():
            output = self.linear(batch)
            output_from_0_to_1 = torch.sigmoid(output)
            output_from_0_to_6 = output_from_0_to_1 * 6
            return output_from_0_to_6


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 300)
        self.linear2 = nn.Linear(300, 200)
        self.linear3 = nn.Linear(200, 100)
        self.linear4 = nn.Linear(100, 50)
        self.linear5 = nn.Linear(50, 65)

    def forward(self, batch, device):
        bert_reps = batch['bert']
        binder_reps = batch['binder']

        output = F.relu(self.linear1(bert_reps))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = F.relu(self.linear4(output))
        output = F.sigmoid(self.linear5(output)) * 6

        loss = F.mse_loss(output, binder_reps)
        return loss

    def forward_eval(self, batch, device):
        bert_rep = batch['bert']
        binder_rep = batch['binder']

        with torch.no_grad():
            output = F.relu(self.linear1(bert_rep))
            output = F.relu(self.linear2(output))
            output = F.relu(self.linear3(output))
            output = F.relu(self.linear4(output))
            output = F.sigmoid(self.linear5(output)) * 6

            loss = F.mse_loss(output, binder_rep)
            return loss.item()

    def forward_rep(self, batch, device):
        with torch.no_grad():
            output = F.relu(self.linear1(batch))
            output = F.relu(self.linear2(output))
            output = F.relu(self.linear3(output))
            output = F.relu(self.linear4(output))
            output = F.sigmoid(self.linear5(output)) * 6
            return output
