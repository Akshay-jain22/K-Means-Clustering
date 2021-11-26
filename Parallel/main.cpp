#include <bits/stdc++.h>
#include "Point.h"
#include "./Node.h"
#include <mpi.h>

#define MAX_DIM 100

using namespace std;

/* We must necessarily create a class or a struct Point to make sure that the point is saved in a
 vector. Cannot create a vector of double arrays */

/* Do all your I/O with cout in the process with rank 0. If you want to output some data from other processes,
    just send MPI message with this data to rank 0.*/

int main(int argc, char *argv[])
{
    int numNodes, rank;
    // const int tag = 13;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);

    // cout << "I'm rank: " << rank << endl;
    // cout << "I'm numNodes: " << numNodes << endl;

    int total_values, total_points, K, max_iterations, lastIteration;
    vector<Point> dataset;

    Node node(rank, MPI_COMM_WORLD);
    node.readDataset();
    node.scatterDataset();
    node.extractCluster();

    auto start = chrono::system_clock::now();
    lastIteration = 0;
    for (int it = 0; it < node.getMaxIterations(); it++)
    {
        if (rank == 0)
        {
            cout << "Iteration: ";
            if (it % 3 == 0)
                cout << "/ " << it << "\r" << flush;
            else if (it % 3 == 1)
                cout << "- " << it << "\r" << flush;
            else
                cout << "\\ " << it << "\r" << flush;
        }

        int notChanged = node.run(it);
        // cout << "Iteration " << it << " ends!" << endl;

        if (rank == 0)
        {
            // cout << "Global not changed = " << notChanged << ". NumNodes = " << numNodes << endl;
        }

        if (notChanged == numNodes)
        {
            // cout << "No more changes, k-means terminates at iteration " << it << endl;
            lastIteration = it;
            break;
        }
        lastIteration = it;
    }

    node.setLastIteration(lastIteration);

    node.computeGlobalMembership();

    auto end = chrono::system_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(end - start);

    if (rank == 0)
    {
        int *gm;
        gm = node.getGlobalMemberships();
        int numPoints = node.getNumPoints();

        // node.printClusters();

        string foldername = "../results/";
        string outFilename = "membershipsFilename";
        cout << "Specify output filename (without .csv): ";
        cin.ignore();
        getline(cin, outFilename);

        node.writeClusterMembership(foldername, outFilename);
    }

    node.getStatistics();

    if (rank == 0)
        cout << "\n - Execution time: " << total_time.count() << " ms" << endl;

    MPI_Finalize();
}