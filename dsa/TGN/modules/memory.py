import torch
import torch.nn as nn


class Memory(nn.Module):

    def __init__(self, memory_dim=64):
        super(Memory, self).__init__()

        self.memory_dim = memory_dim

        self.memory = {}
        self.last_update = {}

        self.gru = nn.GRUCell(memory_dim, memory_dim)

        # learnable decay rate
        self.decay = nn.Parameter(torch.tensor(0.1))


    def init_node(self, node_id):

        if node_id not in self.memory:

            self.memory[node_id] = torch.zeros(
                self.memory_dim,
                device=self.decay.device
            )

            self.last_update[node_id] = 0


    def get_memory(self, node_id):

        if node_id not in self.memory:
            self.init_node(node_id)

        return self.memory[node_id]


    def update_memory(self, node_id, message, timestamp):

        self.init_node(node_id)

        old_mem = self.memory[node_id]
        last_time = self.last_update[node_id]

        delta_t = max(0, timestamp - last_time)

        delta_t = torch.tensor(
            float(delta_t),
            device=self.decay.device
        )

        decay_factor = torch.exp(
            -torch.clamp(self.decay, 0, 5) * delta_t
        )

        decayed_mem = old_mem * decay_factor

        new_mem = self.gru(
            message.unsqueeze(0),
            decayed_mem.unsqueeze(0)
        ).squeeze(0)

        new_mem = torch.tanh(new_mem)

        self.memory[node_id] = new_mem.detach()

        self.last_update[node_id] = timestamp