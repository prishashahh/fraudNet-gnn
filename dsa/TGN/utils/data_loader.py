import pandas as pd


class DataLoader:

    def __init__(self, path):

        self.data = pd.read_csv(path)

        # sort by timestamp (VERY IMPORTANT)
        self.data = self.data.sort_values(by="timestamp")


    def get_split(self, split="train"):

        n = len(self.data)

        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if split == "train":
            df = self.data.iloc[:train_end]

        elif split == "val":
            df = self.data.iloc[train_end:val_end]

        else:
            df = self.data.iloc[val_end:]


        for _, row in df.iterrows():

            sender = row["sender"]
            receiver = row["receiver"]
            timestamp = int(row["timestamp"])

            features = [
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

            # replace NaNs safely
            features = [0.0 if pd.isna(f) else f for f in features]

            scale = [100, 50, 100, 100, 10, 50, 100, 100, 10, 10]

            features = [
                f / s for f, s in zip(features, scale)
            ]

            label = (
                int(row["label"])
                if "label" in self.data.columns
                else None
            )

            yield sender, receiver, features, timestamp, label