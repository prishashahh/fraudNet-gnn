import torch
import pandas as pd
from config import Config

from modules.memory import Memory
from modules.message import messagE
from modules.embedding import embedding
from modules.predictor import predictFraud
from modules.time_encoding import timEncoding

from model.tgn_core import TGN


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(Config.DATA_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # init modules
    memory = Memory(Config.MEMORY_DIM).to(device)
    message_fn = messagE(Config.MEMORY_DIM, 10, Config.TIME_DIM, Config.MESSAGE_DIM).to(device)
    embedding_fn = embedding(Config.MEMORY_DIM, Config.MESSAGE_DIM, Config.EMBEDDING_DIM).to(device)
    predictor = predictFraud(Config.EMBEDDING_DIM, 10).to(device)
    time_encoder = timEncoding(Config.TIME_DIM).to(device)

    # load trained weights
    checkpoint = torch.load("model.pth", map_location=device)

    memory.load_state_dict(checkpoint["memory"])
    message_fn.load_state_dict(checkpoint["message"])
    embedding_fn.load_state_dict(checkpoint["embedding"])
    predictor.load_state_dict(checkpoint["predictor"])
    time_encoder.load_state_dict(checkpoint["time_encoder"])

    # TGN model
    tgn = TGN(memory, message_fn, embedding_fn, predictor, time_encoder, device)

    predictions = []

    with torch.no_grad():

        for _, row in df.iterrows():

            sender = row["sender"]
            receiver = row["receiver"]
            timestamp = int(row["timestamp"])

            event_features = [
                float(row["amount"]),
                float(row["count"]),
                float(row["total"]),
                float(row["avg"]),
                float(row["velocity"]),
                float(row["unique_devices"]),
                float(row["degree_s"]),
                float(row["degree_r"]),
                float(row["risk_s"]),
                float(row["risk_r"])
            ]

            result = tgn.process_event(sender, receiver, event_features, timestamp)

            prob = torch.sigmoid(result["fraud_probability"]).item()

            predictions.append(prob)

    df["fraud_score"] = predictions

    df.to_csv("dataset/data/predicted_events.csv", index=False)

    print("DONE: predictions saved to predicted_events.csv")


if __name__ == "__main__":
    main()