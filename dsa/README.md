Our project focuses on detecting fraudulent behavior in financial transaction networks by modeling the system as a dynamic graph, where nodes represent accounts and edges represent transactions enriched with multiple features such as amount, timestamp, device ID, and location. On top of this graph, we compute several structural and statistical graph features including degree, average neighbor degree, clustering coefficient, and PageRank to capture node importance and local connectivity patterns. We further incorporate advanced graph algorithms such as cycle detection for identifying fraud rings, fan-in and fan-out analysis for spotting suspicious flow patterns, burst detection for temporal anomalies, dense subgraph detection (via clustering coefficient) for coordinated fraud groups, and device-sharing analysis to uncover linked malicious accounts. These handcrafted heuristics are combined into a risk scoring mechanism that provides interpretable signals for each node. To enhance detection capability, we then feed these node and edge features into a Graph Neural Network (GNN), specifically TGN - Temporal Graph Neural Network, which learns latent representations of nodes by aggregating neighborhood information. Alongside this, we apply unsupervised anomaly detection techniques like Isolation Forest and Local Outlier Factor (LOF) on the learned embeddings to identify previously unseen fraud patterns. Here are the node and edge features taken into account:

-> Basic Node Features :
    - In-degree
    - Out-Degree
    - Total transaction amount
    - Latest transaction timestamp

-> Graph-based features
   - PageRank
   - Clustering Coefficient
   - Average neighbor degree

->Dynamic features
   - Transaction frequency
   - Burst activity flag
   - Geo anomaly flag
   - shared device count

-> Edge Features

   -> Raw Edge Features 
   - amount
   - timestamp
   - device_id
   - location
   - tx_type
   - status

  -> MultiGraph Features
   - Transaction count
   - Total amount
   - Average amount
   - Latest transaction time
 
  -> Temporal features
   - velocity
   - Duration

  ->Behavioral Edge features 
   - unique devices used
   - Location variatio 
   - Failed transaction ratio

For the dataset - we are generating synthetic data specially designed for our project and using it as training data. For testing, we will take multiple datasets from kaggle including the Elliptic Bitcoin Dataset.
