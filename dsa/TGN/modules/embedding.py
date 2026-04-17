import torch
import torch.nn as nn


class embedding(nn.Module):

    def __init__(self, memory_dim=64, message_dim=64, emb_dim=64):
        super(embedding, self).__init__()

        input_dim = memory_dim + message_dim

        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def nodeEmbed(self, node_memory, message):

        x = torch.cat(
            [node_memory.view(-1), message.view(-1)],
            dim=0
        )

        embedding = self.linear(x)

        embedding = torch.nan_to_num(embedding, nan=0.0)

        return embedding