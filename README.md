# 🕸️ FraudNet-GNN – Graph-Based Fraud Detection System

FraudNet-GNN is a fraud detection system that models financial transactions as a graph and uses Graph Neural Networks combined with classical graph heuristics and unsupervised anomaly detection to flag suspicious activity. Instead of treating each transaction in isolation, it looks at the relationships between accounts, devices, and transfers to surface fraud patterns that are only visible at the network level such as money laundering rings, fan-in/fan-out schemes, and shared-device fraud clusters.

---

## 📌 Table of Contents

- [✨ Key Features](#-key-features)
- [🧩 Tech Stack](#-tech-stack)
- [📊 Data](#-data)
- [🧠 Feature Engineering](#-feature-engineering)
- [📈 Project Architecture](#-project-architecture)
- [📊 Results](#-results)
- [🏭 Applications](#-applications)

---

## ✨ Key Features

- 🕸️ Models transactions as a **graph** (accounts/entities as nodes, transfers as edges) rather than flat tabular rows
- 🧠 **Graph Neural Network** classification (GraphSAGE / GCN) to learn fraud patterns from node and edge context
- 🔁 Classical **graph heuristics** layered alongside the GNN: cycle detection, fan-in/fan-out detection, burst detection, dense subgraph detection, and device-sharing analysis
- 🚨 **Unsupervised anomaly detection** (Isolation Forest, Local Outlier Factor) to catch novel fraud patterns without labeled examples
- 🔗 Rich **multigraph and temporal edge modeling** to capture repeated and time-sensitive transaction behavior between the same accounts

---

## 🧩 Tech Stack

| Component                  | Technology |
|----------------------------|-----------|
| Core Language               | Python & C++|
| Performance-Critical Modules | C++ |
| Graph Learning               | Dynamic Graph Neural Networks (TGN - Temporal Graph Neural Network) |
| Anomaly Detection            | Isolation Forest, Local Outlier Factor (LOF) |
| Graph Analysis               | Cycle / dense subgraph / fan-in-fan-out heuristics |

---

## 📊 Data

- **Training:** Synthetic dynamic transaction graph data, generated to simulate realistic fraud and non-fraud behavior patterns
- **Testing / Benchmarking:** Public Kaggle fraud datasets, including the **Elliptic Bitcoin Dataset**, used to validate detection performance on real-world-style transaction graphs

---

## 🧠 Feature Engineering

FraudNet-GNN builds a multi-layered feature set for each transaction graph:

- **Node Features** — attributes describing individual accounts
- **Graph-Based Features** — structural signals derived from the surrounding graph topology (e.g. centrality, connectivity patterns)
- **Dynamic Features** — behavior that evolves over time for a given node
- **Edge Features** — including:
  - Raw transaction attributes
  - Multigraph features (handling multiple edges between the same node pair)
  - Temporal features (timing and sequence of transactions)
  - Behavioral features (patterns of interaction between connected accounts)

---

## 📈 Project Architecture

<img width="560" height="847" alt="image" src="https://github.com/user-attachments/assets/39e98ff4-8acb-44db-b02a-1dfaf409c5d1" />


The pipeline combines three complementary detection strategies rather than relying on a single model:

1. **Graph Construction** — raw transaction data is converted into a graph structure (nodes = accounts, edges = transactions)
2. **Heuristic Screening** — graph-theoretic checks (cycles, fan-in/fan-out, bursts, dense subgraphs, shared devices) flag structurally suspicious patterns
3. **GNN Classification** — a dynamic TGN model learns in Real-time from node, edge, and neighborhood context to classify fraud vs. legitimate activity
4. **Anomaly Detection Layer** — Isolation Forest and LOF catch outliers that don't match known fraud signatures, adding coverage for novel fraud types

---

## 📊 Results

<img width="1000" height="480" alt="WhatsApp Image 2026-07-19 at 3 11 14 PM" src="https://github.com/user-attachments/assets/2e2feb53-98f5-4cb5-a0d5-c5a3cb2fde91" />


Final results after 5 training epochs. Model checkpoint saved to `model.pth` after training.

---

## 🏭 Applications

- Anti-money-laundering (AML) monitoring for banks and fintech platforms
- Detecting coordinated fraud rings across multiple accounts
- Cryptocurrency transaction monitoring (e.g. Bitcoin transaction graphs)
- Device-sharing and identity-cluster fraud detection
