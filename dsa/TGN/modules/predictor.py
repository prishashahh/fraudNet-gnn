import torch
import torch.nn as nn


class predictFraud(nn.Module):

    def __init__(self, emb_dim=64, edge_dim=10):

        super(predictFraud, self).__init__()

        input_dim = emb_dim * 2 + edge_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def predict(self, sender_emb, receiver_emb, edge_features):

        edge_features = torch.as_tensor(
            edge_features,
            dtype=torch.float32,
            device=sender_emb.device
        )

        x = torch.cat([
            sender_emb,
            receiver_emb,
            edge_features
        ], dim=0)

        x = torch.nan_to_num(x, nan=0.0)

        logits = self.model(x)

        return logits.squeeze()