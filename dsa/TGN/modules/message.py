import torch
import torch.nn as nn


class messagE(nn.Module):

    def __init__(self,
                 memory_dim=64,
                 edge_dim=10,
                 time_dim=16,
                 message_dim=64):

        super(messagE, self).__init__()

        input_dim = memory_dim * 2 + time_dim + edge_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, message_dim)
        )


    def compute_message(self,
                        sender_mem,
                        receiver_mem,
                        event_features,
                        time_encoding):

        # ensure tensor + device consistency
        event_features = torch.as_tensor(
            event_features,
            dtype=torch.float32,
            device=sender_mem.device
        )

        # concatenate inputs
        x = torch.cat([
            sender_mem,
            receiver_mem,
            time_encoding,
            event_features
        ], dim=0)

        # prevent NaN propagation
        x = torch.nan_to_num(x, nan=0.0)

        message = self.mlp(x)

        return message