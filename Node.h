#ifndef NODE_H
#define NODE_H

#include <string>
#include <mpi.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include "Point.h"

using namespace std;

class Node
{
private:
    int rank;
    MPI_Comm comm;
    MPI_Datatype pointType;

    int total_values, num_local_points, K, max_iterations;
    int numPoints; // Total number of points in the whole dataset

    int notChanged;  // It is used to establish if during a run there have been changes in membership of a cluster.
                     // If there have not been and it <max iterations, then the algorithm has reached the optimal configuration
    int *memCounter; // Membership count
    int lastIteration;
    int distance; // Integer which refers to the number of the chosen distance by the user among: 1) Euclidean Distance  2) Cosine Similarity

    double *reduceArr, *reduceResults;

    vector<Point> dataset, localDataset, clusters, localSum;
    int numPointsPerNode;
    vector<int> memberships; // This vector has same length as localDataset: for each point in localDataset is
                             //  associated the id of nearest cluster in the corresponding position in membership
    int *globalMembership;
    double total_time, omp_total_time;

    int getIdNearestCluster(Point p); // private
    void updateLocalSum();            // private

public:
    Node(int rank, MPI_Comm comm = MPI_COMM_WORLD);
    ~Node();
    int getMaxIterations();
    void readDataset();
    void scatterDataset();
    void extractCluster();
    int run(int it);
    void computeGlobalMembership();
    int getNumPoints();
    int *getGlobalMemberships();
    void printClusters();
    void writeClusterMembership(string filename);

    void getStatistics();
    vector<double> SSW(); // Variance within cluster (https://math.stackexchange.com/questions/1009297/variances-for-k-means-clustering)
    double SSB();
    double squared_norm(Point p1, Point p2);
    double cosine_similarity(Point p1, Point p2);
    void setLastIteration(int lastIt);
};

#endif // NODE_H
