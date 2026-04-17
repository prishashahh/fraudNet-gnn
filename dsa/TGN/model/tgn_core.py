import numpy as np
from config import Config
import torch


class TGN:

    def __init__(self, memory, messageFunc, embeddingFunc,
                 predictor, time_encoder, device):

        self.memory = memory
        self.messageFunc = messageFunc
        self.embeddingFunc = embeddingFunc
        self.predictor = predictor
        self.time_encoder = time_encoder

        self.last_update = {}
        self.message_store = {}

        self.device = device


    # --------------------------------------------------
    # MESSAGE STORAGE
    # --------------------------------------------------

    def store_message(self, node, message):

        if node not in self.message_store:
            self.message_store[node] = []

        self.message_store[node].append(message)


    # --------------------------------------------------
    # MESSAGE AGGREGATION (Scaled Dot-Product Attention)
    # --------------------------------------------------

    def aggregate_messages(self, node):

        if node not in self.message_store or len(self.message_store[node]) == 0:
            return None

        msgs = self.message_store[node]

        msgs_tensor = torch.stack(msgs)   # shape: [N, dim]

        query = msgs_tensor[-1]

        scores = torch.matmul(
            msgs_tensor,
            query
        ) / np.sqrt(msgs_tensor.shape[1])

        weights = torch.softmax(scores, dim=0)

        agg = torch.sum(
            weights.unsqueeze(1) * msgs_tensor,
            dim=0
        )

        self.message_store[node] = []

        return agg


    # --------------------------------------------------
    # MAIN EVENT PROCESSING PIPELINE
    # --------------------------------------------------

    def process_event(self,
                      sender,
                      receiver,
                      event_features,
                      timestamp,
                      update_memory=True):

        # ensure tensor
        event_features = torch.as_tensor(
            event_features,
            dtype=torch.float32,
            device=self.device
        )

        # retrieve memory
        mem_s = self.memory.get_memory(sender).to(self.device)
        mem_r = self.memory.get_memory(receiver).to(self.device)

        # last update timestamps
        last_s = self.last_update.get(sender, 0)
        last_r = self.last_update.get(receiver, 0)

        # safe delta computation
        delta_s = max(0, timestamp - last_s)
        delta_r = max(0, timestamp - last_r)

        # time encoding
        timenc_s = self.time_encoder.encode(delta_s).to(self.device)
        timenc_r = self.time_encoder.encode(delta_r).to(self.device)

        # compute messages
        msg_s = self.messageFunc.compute_message(
            mem_s,
            mem_r,
            event_features,
            timenc_s
        )

        msg_r = self.messageFunc.compute_message(
            mem_r,
            mem_s,
            event_features,
            timenc_r
        )

        # store messages
        self.store_message(sender, msg_s)
        self.store_message(receiver, msg_r)

        # aggregate messages
        agg_s = self.aggregate_messages(sender)
        agg_r = self.aggregate_messages(receiver)

        # update memory ONLY if allowed
        if update_memory and agg_s is not None:
            self.memory.update_memory(sender, agg_s, timestamp)

        if update_memory and agg_r is not None:
            self.memory.update_memory(receiver, agg_r, timestamp)

        # update timestamps
        if update_memory:
            self.last_update[sender] = timestamp
            self.last_update[receiver] = timestamp

        # retrieve updated memory
        new_mem_s = self.memory.get_memory(sender).to(self.device)
        new_mem_r = self.memory.get_memory(receiver).to(self.device)

        # compute embeddings
        embed_s = self.embeddingFunc.nodeEmbed(
            new_mem_s,
            agg_s if agg_s is not None else msg_s
        )

        embed_r = self.embeddingFunc.nodeEmbed(
            new_mem_r,
            agg_r if agg_r is not None else msg_r
        )

        # predictor output (LOGITS)
        logits = self.predictor.predict(
            embed_s,
            embed_r,
            event_features
        )

        # safety guard against NaNs
        logits = torch.nan_to_num(logits, nan=0.0)

        return {
            "fraud_probability": logits,  # logits for BCEWithLogitsLoss
            "is_fraud": (
                torch.sigmoid(logits) > Config.FRAUD_THRESHOLD
            ),
            "sender": sender,
            "receiver": receiver,
            "timestamp": timestamp,
            "event_features": event_features.tolist()
        }