#include <bits/stdc++.h>
using namespace std;

struct EDGE{
    string to;
    double amount;
    int timestamp;
    string device;
    string location;
};

unordered_map<string, vector<EDGE>> GRAPH;
unordered_map<string, unordered_map<string, vector<EDGE>>> multiGraph;
unordered_map<string, int> indegree;
unordered_map<string, unordered_set<string>> deviceUsers;

// ---------------- FEATURES ----------------

double avgNeighborDegree(string node) {
    double sum = 0;
    int count = 0;

    for (auto &e : GRAPH[node]) {
        sum += GRAPH[e.to].size();
        count++;
    }

    if (count == 0) return 0;
    return sum / count;
}

double clusteringCoeff(string node) {

    vector<string> neighbors;

    for (auto &e : GRAPH[node])
        neighbors.push_back(e.to);

    int links = 0;

    for (int i = 0; i < neighbors.size(); i++) {
        for (int j = i + 1; j < neighbors.size(); j++) {

            string u = neighbors[i];
            string v = neighbors[j];

            for (auto &e : GRAPH[u]) {
                if (e.to == v) {
                    links++;
                    break;
                }
            }
        }
    }

    int k = neighbors.size();

    if (k < 2) return 0;

    return (2.0 * links) / (k * (k - 1));
}

int fanIn(string node){ return indegree[node]; }
int fanOut(string node){ return GRAPH[node].size(); }


// -------- FRAUD RINGS / CYCLES --------

double dfsCycleScore(
    string current,
    string start,
    unordered_map<string,bool> &visited,
    int depth,
    int maxDepth = 6
){

    if (depth > maxDepth) return 0;

    visited[current] = true;

    double score = 0;

    for (auto &e : GRAPH[current]) {

        string neighbor = e.to;

        if (neighbor == start && depth >= 2)
            score += 1.0 / depth;

        if (!visited[neighbor])
            score += dfsCycleScore(neighbor,start,visited,depth+1);
    }

    visited[current] = false;

    return score;
}

double fraudRingScore(string node){

    unordered_map<string,bool> visited;

    double rawScore = dfsCycleScore(node,node,visited,0);

    return min(10.0, rawScore * 5.0);
}


// -------- PATTERNS --------

bool sharedDevices(string node){

    for (auto &e : GRAPH[node])
        if (deviceUsers[e.device].size() > 5)
            return true;

    return false;
}


bool burstDetection(string node){

    auto &edges = GRAPH[node];

    if (edges.size() < 5) return false;

    vector<int> times;

    for (auto &e : edges)
        times.push_back(e.timestamp);

    sort(times.begin(), times.end());

    return (times.back() - times.front() < 10);
}


bool geoAnomaly(string node){

    unordered_map<string,int> locCount;

    for (auto &e : GRAPH[node])
        locCount[e.location]++;

    return locCount.size() > 3;
}


bool denseSubgraph(string node){

    return clusteringCoeff(node) > 0.6;
}


// -------- FINAL RISK --------

double computeRisk(string node){

    double risk = 0;

    if (fanIn(node) > 10) risk += 2;
    if (fanOut(node) > 15) risk += 2;
    if (denseSubgraph(node)) risk += 3;
    if (avgNeighborDegree(node) > 10) risk += 1;
    if (burstDetection(node)) risk += 2;
    if (geoAnomaly(node)) risk += 2;
    if (sharedDevices(node)) risk += 2;
    if (fraudRingScore(node) > 6) risk += 3;

    return min(risk, 10.0);
}


// ---------------- MAIN ----------------

int main(){

    ifstream file("dataset/data/transactions.csv");

    ofstream out("dataset/data/event_stream.csv");

    string line;

    getline(file,line);

    out << "timestamp,sender,receiver,amount,count,total,avg,velocity,unique_devices,"
        << "degree_s,degree_r,risk_s,risk_r,label\n";

    while(getline(file,line)){

        stringstream ss(line);

        string item;

        vector<string> row;

        while(getline(ss,item,','))
            row.push_back(item);

        if(row.size() < 10) continue;

        int timestamp = max(1, stoi(row[4]));

        string sender = row[1];
        string receiver = row[2];

        double amount = min(stod(row[3]), 1000.0);

        string device = row[5];
        string location = row[6];

        int label_val = stoi(row[9]);

        // -------- UPDATE GRAPH --------

        EDGE e;

        e.to = receiver;
        e.amount = amount;
        e.timestamp = timestamp;
        e.device = device;
        e.location = location;

        GRAPH[sender].push_back(e);

        multiGraph[sender][receiver].push_back(e);

        indegree[receiver]++;

        deviceUsers[device].insert(sender);


        // -------- EDGE FEATURES --------

        auto &edges = multiGraph[sender][receiver];

        int count = edges.size();

        double total_amt = 0;

        unordered_set<string> devices;

        int min_t = edges[0].timestamp;
        int max_t = edges[0].timestamp;

        for (auto &ed : edges){

            total_amt += ed.amount;

            min_t = min(min_t, ed.timestamp);
            max_t = max(max_t, ed.timestamp);

            devices.insert(ed.device);
        }

        double avg = count > 0 ? total_amt / count : 0;

        double velocity = 0;

        if (count > 1){

            int duration = max(1, max_t - min_t);

            velocity = min((double)count / duration, 10.0);
        }


        // -------- NODE FEATURES --------

        int degree_s = GRAPH[sender].size();

        int degree_r = GRAPH[receiver].size();

        double risk_s = computeRisk(sender);

        double risk_r = computeRisk(receiver);


        // -------- WRITE OUTPUT --------

        out << timestamp << ","
            << sender << ","
            << receiver << ","
            << amount << ","
            << count << ","
            << total_amt << ","
            << avg << ","
            << velocity << ","
            << devices.size() << ","
            << degree_s << ","
            << degree_r << ","
            << risk_s << ","
            << risk_r << ","
            << label_val << "\n";
    }

    out.close();

    cout << "event_stream.csv generated successfully\n";

    return 0;
}