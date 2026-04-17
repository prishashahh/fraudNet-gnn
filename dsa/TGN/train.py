import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
Config.set_seed()

from utils.data_loader import DataLoader
from utils.metrics import accuracy, precision, recall

from modules.memory import Memory
from modules.message import messagE
from modules.embedding import embedding
from modules.predictor import predictFraud
from modules.time_encoding import timEncoding
from model.tgn_core import TGN


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- INITIALIZE MODULES ----------------

    memory = Memory(Config.MEMORY_DIM).to(device)

    messageFunc = messagE(
        Config.MEMORY_DIM,
        Config.EVENT_DIM,
        Config.TIME_DIM,
        Config.MESSAGE_DIM
    ).to(device)

    embeddingFunc = embedding(
        Config.MEMORY_DIM,
        Config.MESSAGE_DIM,
        Config.EMBEDDING_DIM
    ).to(device)

    predictor = predictFraud(
        Config.EMBEDDING_DIM,
        Config.EVENT_DIM
    ).to(device)

    time_encoder = timEncoding(Config.TIME_DIM).to(device)

    temporalGNN = TGN(
        memory,
        messageFunc,
        embeddingFunc,
        predictor,
        time_encoder,
        device
    )

    # ---------------- OPTIMIZER ----------------

    params = (
        list(memory.parameters())
        + list(messageFunc.parameters())
        + list(embeddingFunc.parameters())
        + list(predictor.parameters())
        + list(time_encoder.parameters())
    )

    optimizer = optim.Adam(params, lr=Config.LEARNING_RATE)

    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(Config.DATA_PATH)

    # ---------------- NODE SET FOR NEGATIVE SAMPLING ----------------

    all_nodes = set()

    for sender, receiver, _, _, _ in loader.get_split("train"):
        all_nodes.add(sender)
        all_nodes.add(receiver)

    all_nodes = list(all_nodes)

    # ================= TRAINING LOOP =================

    for epoch in range(Config.EPOCHS):

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        preds = []
        labels = []

        total_loss = 0
        batch_loss = 0
        batch_count = 0

        loop = tqdm(loader.get_split("train"), desc=f"Epoch {epoch+1}")

        for sender, receiver, event_features, timestamp, label in loop:

            if label is None:
                continue

            # -------- POSITIVE SAMPLE --------

            result = temporalGNN.process_event(
                sender,
                receiver,
                event_features,
                timestamp,
                update_memory=True
            )

            pred = result["fraud_probability"].to(device)
            pred = torch.clamp(pred, -10, 10)

            target = torch.tensor([label], dtype=torch.float32, device=device)

            loss = criterion(pred.view(1), target)

            # -------- NEGATIVE SAMPLE --------

            neg_receiver = random.choice(all_nodes)

            while neg_receiver == receiver:
                neg_receiver = random.choice(all_nodes)

            neg_result = temporalGNN.process_event(
                sender,
                neg_receiver,
                event_features,
                timestamp,
                update_memory=False
            )

            neg_pred = neg_result["fraud_probability"].to(device)
            neg_pred = torch.clamp(neg_pred, -10, 10)

            neg_target = torch.tensor([0.0], dtype=torch.float32, device=device)

            loss += criterion(neg_pred.view(1), neg_target)

            # -------- BATCH ACCUMULATION --------

            batch_loss += loss / Config.BATCH_SIZE
            batch_count += 1

            preds.append(pred.item())
            labels.append(label)

            if batch_count == Config.BATCH_SIZE:

                optimizer.zero_grad()
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()

                total_loss += batch_loss.item()

                batch_loss = 0
                batch_count = 0

            loop.set_postfix(loss=loss.item())

        # -------- HANDLE FINAL PARTIAL BATCH --------

        if batch_count > 0:

            optimizer.zero_grad()
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()

            total_loss += batch_loss.item()

        # -------- EPOCH METRICS --------

        print(f"\nEpoch {epoch+1} Loss: {total_loss:.4f}")
        print("Accuracy:", accuracy(preds, labels))
        print("Precision:", precision(preds, labels))
        print("Recall:", recall(preds, labels))

    # ================= SAVE MODEL =================

    torch.save({
        "memory": memory.state_dict(),
        "message": messageFunc.state_dict(),
        "embedding": embeddingFunc.state_dict(),
        "predictor": predictor.state_dict(),
        "time_encoder": time_encoder.state_dict()
    }, "model.pth")

    print("\nModel saved to model.pth")

    # ================= TESTING =================

    print("\n--- TESTING ---")

    test_preds = []
    test_labels = []

    for sender, receiver, event_features, timestamp, label in loader.get_split("test"):

        if label is None:
            continue

        result = temporalGNN.process_event(
            sender,
            receiver,
            event_features,
            timestamp,
            update_memory=True
        )

        pred = result["fraud_probability"].item()

        test_preds.append(pred)
        test_labels.append(label)

    print("Test Accuracy:", accuracy(test_preds, test_labels))
    print("Test Precision:", precision(test_preds, test_labels))
    print("Test Recall:", recall(test_preds, test_labels))


if __name__ == "__main__":
    train()