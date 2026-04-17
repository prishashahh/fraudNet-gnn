import torch
import random
import numpy as np


class Config:

    MEMORY_DIM = 64
    EMBEDDING_DIM = 64
    MESSAGE_DIM = 64
    TIME_DIM = 16
    EVENT_DIM = 10

    LEARNING_RATE = 1e-4   # fixed

    EPOCHS = 5
    BATCH_SIZE = 20

    DATA_PATH = "../.vscode/event_stream.csv"

    FRAUD_THRESHOLD = 0.7

    SEED = 42


    @staticmethod
    def set_seed():
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)