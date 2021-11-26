#include <bits/stdc++.h>
#include "./Node.h"

using namespace std;

int main()
{
    Node node;
    node.readDataset();
    node.extractClusterCenters();

    auto start = chrono::system_clock::now();
    int max_iterations = node.getMaxIterations();
    int last_iteration = 0;
    for (int it = 0; it < max_iterations; it++)
    {
        cout << "Iteration: ";
        if (it % 3 == 0)
            cout << "/ " << it << "\r" << flush;
        else if (it % 3 == 1)
            cout << "- " << it << "\r" << flush;
        else
            cout << "\\ " << it << "\r" << flush;

        last_iteration = it;
        bool notChanged = node.run();

        if (notChanged)
            break;

        node.reComputeCentroids();
    }
    auto end = chrono::system_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(end - start);

    node.setLastIteration(last_iteration);

    string foldername = "../results/";
    cout << "Specify output filename (without .csv): ";
    cin.ignore();
    string outFilename;
    getline(cin, outFilename);

    node.writeClusterMembership(foldername, outFilename);

    node.getStatistics();

    cout << "\n - Execution time: " << total_time.count() << " ms" << endl;

    return 0;
}