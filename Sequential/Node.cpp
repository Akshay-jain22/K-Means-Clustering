#include <iostream>
#include <algorithm>
#include <vector>
#include <stddef.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include "./Node.h"

Node::Node()
{
    total_time = 0;
    lastIteration = 0;
    distanceType = 0;
}

void Node::readDataset()
{
    string filename;
    cout << "Enter File Path for input dataset : ";
    getline(cin, filename);

    cout << "\nReading file ..." << endl;
    ifstream infile(filename);

    // Reading First Line
    string line;
    getline(infile, line, '\n');
    stringstream ss(line);

    getline(ss, line, ',');
    dimensions = stoi(line);
    cout << "Total dimensions is: " << dimensions << endl;

    getline(ss, line, ',');
    K = stoi(line);
    cout << "Number of clusters K is: " << K << endl;

    getline(ss, line, '\n');
    max_iterations = stoi(line);
    cout << "Max iteration is: " << max_iterations << endl;

    // Reading Further Data Points
    int num = 0;
    while (getline(infile, line, '\n'))
    {
        Point point;
        point.id = num;
        point.size = dimensions;

        stringstream ss(line);
        while (getline(ss, line, ','))
            point.values.push_back(stod(line));
        num++;

        memberships.push_back(-1);
        dataset.push_back(point);
    }

    infile.close();
    cout << "Reading ended" << endl;

    distanceType = -1;
    bool onDistance = true;
    while (onDistance)
    {
        cout << "\nWhich distance do you want to use? " << endl;
        cout << "1) Euclidean Distance" << endl;
        cout << "2) Cosine Similarity" << endl;
        cout << "Enter your choice and press return: ";
        cin >> distanceType;

        switch (distanceType)
        {
        case 1:
            onDistance = false;
            break;
        case 2:
            onDistance = false;
            break;
        default:
            cout << "Not a Valid Choice. \n";
            cout << "Choose again.\n";
        }
    }
}

void Node::extractClusterCenters()
{
    if (K >= dataset.size())
    {
        cout << "K : " << K << ", Dataset Size : " << dataset.size() << endl;
        cout << "ERROR: Number of cluster >= number of points " << endl;
        return;
    }

    int choice;
    bool gameOn = true;
    while (gameOn)
    {
        cout << "\nChoose initialization method for clusters" << endl;
        cout << "1) Random \n";
        cout << "2) First k points\n";
        cout << "Enter your choice and press return: ";
        cin >> choice;

        switch (choice)
        {
        case 1:
        {
            vector<int> clusterIndices;
            for (int i = 0; i < K; i++)
                while (true)
                {
                    int randIndex = rand() % dataset.size();

                    if (find(clusterIndices.begin(), clusterIndices.end(), randIndex) == clusterIndices.end())
                    {
                        clusterIndices.push_back(randIndex);
                        cluster_centers.push_back(dataset[clusterIndices[i]]);
                        clusterSum.push_back(dataset[clusterIndices[i]]);
                        break;
                    }
                }

            gameOn = false;
            break;
        }

        case 2:
        {
            for (int i = 0; i < K; i++)
            {
                cluster_centers.push_back(dataset[i]);
                clusterSum.push_back(dataset[i]);
            }
            gameOn = false;
            break;
        }

        default:
        {
            cout << "Not a Valid Choice. \n";
            cout << "Choose again.\n";
            cin >> choice;
            break;
        }
        }
    }

    for (int j = 0; j < K; ++j)
        membership_count.push_back(0);
}

int Node::getMaxIterations()
{
    return max_iterations;
}

void Node::setLastIteration(int lastIt)
{
    lastIteration = lastIt;
}

// This function compute || p1 - p2 ||^2
double Node::squared_norm(Point p1, Point p2)
{
    double sum = 0.0;
    for (int j = 0; j < dimensions; j++)
        sum += pow(p1.values[j] - p2.values[j], 2.0);
    return sum;
}

double Node::cosine_similarity(Point p1, Point p2)
{
    double num = 0.0;
    for (int j = 0; j < dimensions; j++)
        num += p1.values[j] * p2.values[j];

    double sum1 = 0.0;
    for (int i = 0; i < dimensions; i++)
        sum1 += pow(p1.values[i], 2.0);
    double norm_p1 = sqrt(sum1);

    double sum2 = 0.0;
    for (int k = 0; k < dimensions; k++)
        sum2 += pow(p2.values[k], 2.0);
    double norm_p2 = sqrt(sum2);

    return num / (norm_p1 * norm_p2);
}

int Node::getIdNearestCluster(Point p)
{
    int idCluster = 0; // is the position in the vector clusters, not the id of the point that represents the initial centroid

    // Refers to Euclidean Distance
    if (distanceType == 1)
    {
        double min_dist = sqrt(squared_norm(cluster_centers[0], p));

        for (int k = 1; k < K; k++)
        {
            double dist = sqrt(squared_norm(cluster_centers[k], p));
            if (dist < min_dist)
            {
                min_dist = dist;
                idCluster = k;
            }
        }
    }

    // Refers to Cosine Similarity
    else if (distanceType == 2)
    {
        double max_sim = 0.0;

        for (int k = 0; k < K; k++)
        {
            double sim = cosine_similarity(cluster_centers[k], p);
            if (sim > max_sim)
            {
                max_sim = sim;
                idCluster = k;
            }
        }
    }

    return idCluster;
}

void Node::updateClusterSum(Point &clusterCenter, Point p, int factor)
{
    for (int i = 0; i < dimensions; i++)
        clusterCenter.values[i] += factor * p.values[i];
}

bool Node::run()
{
    bool notChanged = true;

    for (int i = 0; i < dataset.size(); i++)
    {
        int idCluster = getIdNearestCluster(dataset[i]);
        if (memberships[i] != idCluster)
        {
            if (memberships[i] != -1)
            {
                membership_count[memberships[i]]--;
                updateClusterSum(clusterSum[memberships[i]], dataset[i], -1);
            }
            membership_count[idCluster]++;
            updateClusterSum(clusterSum[idCluster], dataset[i], +1);

            memberships[i] = idCluster;
            notChanged = false;
        }
    }

    return notChanged;
}

void Node::reComputeCentroids()
{
    // Iterate over dataset to append data to centroids
    for (int i = 0; i < K; i++)
        for (int j = 0; j < dimensions; j++)
            cluster_centers[i].values[j] = clusterSum[i].values[j] / (membership_count[i] != 0 ? membership_count[i] : 1);
}

void Node::writeClusterMembership(string foldername, string filename)
{
    ofstream myfile;
    myfile.open(foldername + filename + ".csv");
    myfile << "Point_id, Cluster_id" << endl;
    for (int p = 0; p < dataset.size(); p++)
        myfile << dataset[p].id << ", " << memberships[p] << endl;
    myfile.close();
}

// Standard Deviation of the sum of squares within each cluster
vector<double> Node::SSW()
{
    vector<double> ssws;
    double ssw = 0.0;

    for (int k = 0; k < K; k++)
    {
        double dist, sum;
        for (int j = 0; j < dataset.size(); j++)
            if (k == memberships[j])
            {
                sum = squared_norm(dataset[j], cluster_centers[k]);
                dist = sqrt(sum);
                ssw += dist;
            }
        ssws.push_back(sqrt(sum));
    }
    return ssws;
}

// Standard Deviation of the sum of squares between clusters and mean point values
double Node::SSB()
{
    // Find the mean point values among all points
    double sum = 0.0, ssb = 0.0;
    Point mean;
    for (int i = 0; i < dimensions; i++)
        mean.values.push_back(0);

    for (int i = 0; i < dataset.size(); i++)
        updateClusterSum(mean, dataset[i], 1);

    for (int j = 0; j < dimensions; j++)
        mean.values[j] /= dataset.size();

    for (int k = 0; k < K; k++)
        sum += squared_norm(cluster_centers[k], mean);

    return ssb = sqrt(sum);
}

void Node::getStatistics()
{
    cout << "---------------------  Statistics  ------------------------- " << endl;

    cout << " - Iteration computed: " << lastIteration << endl;

    cout << "\n - Number of points in each cluster" << endl;
    for (int k = 0; k < K; k++)
        cout << "\tCluster " << k << " : " << membership_count[k] << endl;

    cout << "\n - Variance: " << endl;
    cout << "\t- SSW: " << endl;
    vector<double> ssws = SSW();
    for (int i = 0; i < ssws.size(); i++)
        cout << "\t\tCluster " << i << " : " << ssws[i] << endl;

    cout << "\n\t- SSB:    ";
    double ssb = SSB();
    cout << ssb << endl;
}
