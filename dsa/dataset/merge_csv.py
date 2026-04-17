import csv

# -----------------------------
# LOAD NODE FEATURES
# -----------------------------
node_features = {}

with open("data/node_features.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        node = row["node"]
        node_features[node] = row


# -----------------------------
# LOAD EDGE FEATURES
# -----------------------------
edge_features = {}

with open("data/edge_features.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["sender"], row["receiver"])
        edge_features[key] = row


# -----------------------------
# MERGE DATA
# -----------------------------
merged_data = []

with open("data/transactions.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        sender = row["sender"]
        receiver = row["receiver"]

        # Node features
        s_feat = node_features.get(sender, {})
        r_feat = node_features.get(receiver, {})

        # Edge features
        e_feat = edge_features.get((sender, receiver), {})

        # -----------------------------
        # BUILD FINAL ROW
        # -----------------------------
        new_row = {
            "timestamp": int(row["timestamp"]),
            "sender": sender,
            "receiver": receiver,

            "amount": float(row["amount"]),
            "device_id": row["device_id"],
            "location": row["location"],
            "tx_type": row["tx_type"],

            # -------- NODE FEATURES --------
            "degree_s": float(s_feat.get("degree", 0)),
            "degree_r": float(r_feat.get("degree", 0)),

            "total_amount_s": float(s_feat.get("total_amount", 0)),
            "total_amount_r": float(r_feat.get("total_amount", 0)),

            "latest_tx_s": float(s_feat.get("latest_tx", 0)),
            "latest_tx_r": float(r_feat.get("latest_tx", 0)),

            "avgNbrDeg_s": float(s_feat.get("avgNbrDeg", 0)),
            "avgNbrDeg_r": float(r_feat.get("avgNbrDeg", 0)),

            "clustering_s": float(s_feat.get("clustering", 0)),
            "clustering_r": float(r_feat.get("clustering", 0)),

            "pagerank_s": float(s_feat.get("pagerank", 0)),
            "pagerank_r": float(r_feat.get("pagerank", 0)),

            "risk_s": float(s_feat.get("risk", 0)),
            "risk_r": float(r_feat.get("risk", 0)),

            # -------- EDGE FEATURES --------
            "count": float(e_feat.get("count", 0)),
            "total": float(e_feat.get("total", 0)),
            "avg": float(e_feat.get("avg", 0)),
            "velocity": float(e_feat.get("velocity", 0)),
            "unique_devices": float(e_feat.get("unique_devices", 0)),
            "last_tx": float(e_feat.get("last_tx", 0)),

            # -------- LABEL --------
            "label": int(row["label"])
        }

        merged_data.append(new_row)


# -----------------------------
# SORT BY TIMESTAMP
# -----------------------------
merged_data.sort(key=lambda x: x["timestamp"])


# -----------------------------
# SAVE FINAL FILE
# -----------------------------
output_file = "data/event_stream.csv"

keys = merged_data[0].keys()

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(merged_data)

print("event_stream.csv generated successfully!")