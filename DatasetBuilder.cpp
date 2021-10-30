#include <bits/stdc++.h>
using namespace std;

class DatasetBuilder
{
private:
    int numPoints, pointDimension, numClusters, maxIteration;
    string filename, foldername;

    double fRand(double fMin, double fMax)
    {
        double f = (double)rand() / RAND_MAX;
        return fMin + f * (fMax - fMin);
    }

public:
    DatasetBuilder(int numPoints, int pointDimension, int numClusters, int maxIteration, string foldername, string filename)
    {
        this->numPoints = numPoints;
        this->pointDimension = pointDimension;
        this->numClusters = numClusters;
        this->maxIteration = maxIteration;
        this->foldername = foldername;
        this->filename = filename;
    }

    ~DatasetBuilder() {}

    void createDataset()
    {
        srand(time(NULL));

        ofstream myfile;
        myfile.open(foldername + "/" + filename + ".csv");
        myfile << pointDimension << "," << numClusters << "," << maxIteration << "\n";
        for (int p = 0; p < numPoints; p++)
        {
            for (int j = 0; j < pointDimension - 1; j++)
                myfile << fRand(-150, 150) << ",";
            myfile << fRand(-150, 150) << endl;
        }
        myfile.close();
    }
};

int main()
{
    int numPoints, pointDimension, numClusters, maxIteration;
    string filename, foldername = "data";

    cout << "How many points do you want to create : ";
    cin >> numPoints;

    cout << "Point dimension: ";
    cin >> pointDimension;

    cout << "Number of clusters: ";
    cin >> numClusters;

    cout << "Max iteration: ";
    cin >> maxIteration;

    cout << "Filename: ";
    cin.ignore();
    getline(cin, filename);

    DatasetBuilder builder(numPoints, pointDimension, numClusters, maxIteration, foldername, filename);
    builder.createDataset();

    cout << "\nDataset Successfully Created at " << foldername << "/" << filename << ".csv" << endl;
    return 0;
}