import torch
import torch.nn as nn


class timEncoding(nn.Module):

    def __init__(self, time_dim=16):

        super(timEncoding, self).__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([1.0, 10.0, 100.0])
        )

        self.linear = nn.Linear(6, time_dim)


    def encode(self, delta_t):

        delta_t = torch.as_tensor(
            delta_t,
            dtype=torch.float32,
            device=self.frequencies.device
        )

        # normalize time scale (important)
        delta_t = delta_t / 1000.0

        delta_t = delta_t.unsqueeze(0)

        time_encod = []

        for w in self.frequencies:

            time_encod.append(torch.sin(w * delta_t))
            time_encod.append(torch.cos(w * delta_t))

        time_vec = torch.cat(time_encod).view(1, -1)

        time_vec = torch.nan_to_num(time_vec, nan=0.0)

        time_embedding = self.linear(time_vec).squeeze(0)

        return time_embedding