import random
import csv
import os

random.seed(42)


# CONFIG
NUM_ACCOUNTS = 250
NUM_TRANSACTIONS = 5000
FRAUD_RATIO = 0.15

LOCATIONS = ["Delhi", "Mumbai", "Bangalore", "Dubai", "Singapore", "London", "Pakistan", "Ladakh"]
TX_TYPES = ["transfer", "payment", "withdrawal"]

# -----------------------------
# USER PROFILES
# -----------------------------
user_profiles = {}

def random_account():
    return f"A{random.randint(1, NUM_ACCOUNTS)}"

def random_device():
    return f"D{random.randint(1, 300)}"

def random_location():
    return random.choice(LOCATIONS)

def random_amount():
    return random.randint(100, 10000)

def init_user(user):
    if user not in user_profiles:
        user_profiles[user] = {
            "avg_amount": random.randint(500, 5000),
            "devices": {random_device()},
            "home_location": random_location(),
            "last_active": 0
        }

# -----------------------------
# NORMAL TRANSACTION
# -----------------------------
def generate_normal(tx_id, timestamp):
    sender = random_account()
    receiver = random_account()

    init_user(sender)
    profile = user_profiles[sender]

    amount = int(random.gauss(profile["avg_amount"], 200))
    amount = max(100, amount)

    device = random.choice(list(profile["devices"]))

    return [{
        "tx_id": tx_id,
        "sender": sender,
        "receiver": receiver,
        "amount": amount,
        "timestamp": timestamp,
        "device_id": device,
        "location": profile["home_location"],
        "tx_type": random.choice(TX_TYPES),
        "status": "success",
        "label": 0
    }]

# -----------------------------
# FRAUD PATTERNS
# -----------------------------

# Ring (cycle)
def generate_fraud_ring(tx_id, timestamp):
    ring_size = random.randint(3, 6)
    nodes = [random_account() for _ in range(ring_size)]
    txs = []

    amt = random.randint(800, 2000)

    for i in range(ring_size):
        txs.append({
            "tx_id": tx_id + i,
            "sender": nodes[i],
            "receiver": nodes[(i + 1) % ring_size],
            "amount": amt,
            "timestamp": timestamp + i,
            "device_id": f"D_SHARED_{random.randint(1,5)}",
            "location": "Dubai",
            "tx_type": "transfer",
            "status": "success",
            "label": 1
        })
    return txs

# Burst
def generate_fraud_burst(tx_id, timestamp):
    user = random_account()
    init_user(user)

    txs = []
    for i in range(random.randint(5, 12)):
        txs.append({
            "tx_id": tx_id + i,
            "sender": user,
            "receiver": random_account(),
            "amount": random.randint(500, 2000),
            "timestamp": timestamp + i,
            "device_id": random_device(),
            "location": "Singapore",
            "tx_type": "transfer",
            "status": "success",
            "label": 1
        })
    return txs

# High amount spike
def fraud_high_amount(tx_id, timestamp):
    user = random_account()
    init_user(user)
    profile = user_profiles[user]

    return [{
        "tx_id": tx_id,
        "sender": user,
        "receiver": random_account(),
        "amount": profile["avg_amount"] * 20,
        "timestamp": timestamp,
        "device_id": random.choice(list(profile["devices"])),
        "location": profile["home_location"],
        "tx_type": "transfer",
        "status": "success",
        "label": 1
    }]

# Geo anomaly
def fraud_geo(tx_id, timestamp):
    user = random_account()
    init_user(user)

    return [{
        "tx_id": tx_id,
        "sender": user,
        "receiver": random_account(),
        "amount": random_amount(),
        "timestamp": timestamp,
        "device_id": random_device(),
        "location": "London",
        "tx_type": "transfer",
        "status": "success",
        "label": 1
    }]

# New device
def fraud_new_device(tx_id, timestamp):
    user = random_account()
    init_user(user)

    return [{
        "tx_id": tx_id,
        "sender": user,
        "receiver": random_account(),
        "amount": random_amount(),
        "timestamp": timestamp,
        "device_id": random_device(),
        "location": random_location(),
        "tx_type": "transfer",
        "status": "success",
        "label": 1
    }]

# Shared device
def fraud_shared_device(tx_id, timestamp):
    device = f"D_SHARED_{random.randint(1,5)}"
    txs = []

    for i in range(5):
        txs.append({
            "tx_id": tx_id + i,
            "sender": random_account(),
            "receiver": random_account(),
            "amount": random_amount(),
            "timestamp": timestamp + i,
            "device_id": device,
            "location": "Dubai",
            "tx_type": "transfer",
            "status": "success",
            "label": 1
        })
    return txs

# Cycle
def fraud_cycle(tx_id, timestamp):
    nodes = [random_account() for _ in range(4)]
    txs = []

    for i in range(4):
        txs.append({
            "tx_id": tx_id + i,
            "sender": nodes[i],
            "receiver": nodes[(i+1)%4],
            "amount": 1000,
            "timestamp": timestamp + i,
            "device_id": "D_SHARED_1",
            "location": "Dubai",
            "tx_type": "transfer",
            "status": "success",
            "label": 1
        })
    return txs

# Fan-in
def fraud_fanin(tx_id, timestamp):
    target = random_account()
    txs = []

    for i in range(5):
        txs.append({
            "tx_id": tx_id + i,
            "sender": random_account(),
            "receiver": target,
            "amount": random_amount(),
            "timestamp": timestamp + i,
            "device_id": random_device(),
            "location": "Dubai",
            "tx_type": "transfer",
            "status": "success",
            "label": 1
        })
    return txs

# Dormant
def fraud_dormant(tx_id, timestamp):
    user = random_account()
    init_user(user)

    return [{
        "tx_id": tx_id,
        "sender": user,
        "receiver": random_account(),
        "amount": 5000,
        "timestamp": timestamp + 10000,
        "device_id": random_device(),
        "location": "Singapore",
        "tx_type": "transfer",
        "status": "success",
        "label": 1
    }]

# Failed attempts
def fraud_failed(tx_id, timestamp):
    user = random_account()
    txs = []

    for i in range(3):
        txs.append({
            "tx_id": tx_id + i,
            "sender": user,
            "receiver": random_account(),
            "amount": random_amount(),
            "timestamp": timestamp + i,
            "device_id": random_device(),
            "location": random_location(),
            "tx_type": "transfer",
            "status": "failed",
            "label": 1
        })

    txs.append({
        "tx_id": tx_id + 3,
        "sender": user,
        "receiver": random_account(),
        "amount": random_amount(),
        "timestamp": timestamp + 3,
        "device_id": random_device(),
        "location": random_location(),
        "tx_type": "transfer",
        "status": "success",
        "label": 1
    })

    return txs

# -----------------------------
# MAIN GENERATOR
# -----------------------------
def generate_dataset():
    data = []
    tx_id = 1
    timestamp = 0

    patterns = [
        generate_fraud_ring,
        generate_fraud_burst,
        fraud_high_amount,
        fraud_geo,
        fraud_new_device,
        fraud_shared_device,
        fraud_cycle,
        fraud_fanin,
        fraud_dormant,
        fraud_failed
    ]

    while tx_id < NUM_TRANSACTIONS:
        if random.random() < FRAUD_RATIO:
            pattern = random.choice(patterns)
            txs = pattern(tx_id, timestamp)
        else:
            txs = generate_normal(tx_id, timestamp)

        data.extend(txs)
        tx_id += len(txs)
        timestamp += random.randint(1, 10)

    #random.shuffle(data)
    return data

# -----------------------------
# SAVE CSV
# -----------------------------
def save_csv(data, filename="data/transactions.csv"):
    os.makedirs("data", exist_ok=True) 
    keys = data[0].keys()

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    dataset = generate_dataset()
    save_csv(dataset)
    print("Dataset generated successfully!")