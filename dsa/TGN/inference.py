import torch
import pandas as pd

from config import Config
from utils.data_loader import DataLoader

from modules.memory import Memory
from modules.message import messagE
from modules.embedding import embedding
from modules.predictor import predictFraud
from modules.time_encoding import timEncoding
from model.tgn_core import TGN


def run_inference():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize modules
    memory = Memory(Config.MEMORY_DIM).to(device)

    message_fn = messagE(
        Config.MEMORY_DIM,
        Config.EVENT_DIM,
        Config.TIME_DIM,
        Config.MESSAGE_DIM
    ).to(device)

    embedding_fn = embedding(
        Config.MEMORY_DIM,
        Config.MESSAGE_DIM,
        Config.EMBEDDING_DIM
    ).to(device)

    predictor = predictFraud(
        Config.EMBEDDING_DIM,
        Config.EVENT_DIM
    ).to(device)

    time_encoder = timEncoding(Config.TIME_DIM).to(device)

    # build model
    tgn = TGN(
        memory,
        message_fn,
        embedding_fn,
        predictor,
        time_encoder,
        device
    )

    # load trained weights
    checkpoint = torch.load("model.pth", map_location=device)

    memory.load_state_dict(checkpoint["memory"])
    message_fn.load_state_dict(checkpoint["message"])
    embedding_fn.load_state_dict(checkpoint["embedding"])
    predictor.load_state_dict(checkpoint["predictor"])
    time_encoder.load_state_dict(checkpoint["time_encoder"])

    loader = DataLoader(Config.DATA_PATH)

    results = []

    with torch.no_grad():

        for sender, receiver, event_features, timestamp, _ in loader.get_split("test"):

            result = tgn.process_event(
                sender,
                receiver,
                event_features,
                timestamp
            )

            prob = torch.sigmoid(
                result["fraud_probability"]
            ).item()

            results.append({
                "sender": sender,
                "receiver": receiver,
                "timestamp": timestamp,
                "fraud_probability": prob,
                "is_fraud": prob > Config.FRAUD_THRESHOLD
            })

    # save predictions
    df = pd.DataFrame(results)
    df.to_csv("../predicted_events.csv", index=False)

    print("✅ Predictions saved to ../predicted_events.csv")

    return df


if __name__ == "__main__":
    run_inference()