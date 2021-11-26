#ifndef NODE_H
#define NODE_H

#include <bits/stdc++.h>
#include "./Point.h"

using namespace std;

class Node
{
private:
    int dimensions, K, max_iterations, distanceType, lastIteration;
    vector<Point> dataset, cluster_centers, clusterSum;
    vector<int> memberships, membership_count;
    double total_time;

public:
    void readDataset();
    void extractClusterCenters();
    int getMaxIterations();
    int getIdNearestCluster(Point p);
    void updateClusterSum(Point &clusterCenter, Point p, int factor);
    bool run();
    void reComputeCentroids();
    void writeClusterMembership(string foldername, string filename);

    void getStatistics();
    vector<double> SSW(); // Variance within cluster (https://math.stackexchange.com/questions/1009297/variances-for-k-means-clustering)
    double SSB();
    double squared_norm(Point p1, Point p2);
    double cosine_similarity(Point p1, Point p2);
    void setLastIteration(int);

    Node();
};

#endif