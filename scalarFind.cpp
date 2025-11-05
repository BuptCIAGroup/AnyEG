#include <iostream>
#include <math.h>
#include <algorithm>
// #include <stl_map.h>
#include <map>
#include <iomanip>
#include <utility>
#include <fstream>
#include "stopwatch.h"
#include <vector>
using namespace std;
#define MAX_NUMBER_LEN 9

int main(int argc, char **argv)
{
    ifstream ifs;
    ifs.open("/home/liqm/ttk/ttk-build/bin/output",ios::in);
    if (!ifs.is_open()) {
        cerr << "Failed to open file." << endl;
        return 1;
    }

    int N;
    ifs >> N;
    int EdgeNum;
    double* Values = new double[N + 1];
    int* StartEdges = new int[N + 1];
    int* EndEdges = new int[N + 1];
    int* PointFromJTs = new int[N + 1];
    int* EdgeOffsets = new int[N + 1];

    /**
     *  读取数据
     */
    for (int i = 0; i < N; i++) ifs >> Values[i];
    double min_val = Values[0], max_val = Values[0];
    for (int i = 1; i < N; i++) {
        min_val = min(min_val, Values[i]);
        max_val = max(max_val, Values[i]);
    }
    cout << "scalar value range : [" << min_val << "," << max_val << "]" << endl;

    ifs >> EdgeNum;
    cout << "edge num:" << EdgeNum << endl;
    double* StartEdgeValues = new double[EdgeNum + 1];
    double* EndEdgeValues = new double[EdgeNum + 1];
    double* EndPrefixMaxValues = new double[EdgeNum + 1];
    for (int i = 0; i < EdgeNum; i++) {
        ifs >> StartEdges[i] >> EndEdges[i] >> PointFromJTs[i] >> EdgeOffsets[i];
    }
    int ExtendedEdgeNum;
    ifs >> ExtendedEdgeNum;
    cout << "extended edge num:" << ExtendedEdgeNum << endl;

    for (int i = 0; i < EdgeNum; i++) {
        StartEdgeValues[i] = Values[StartEdges[i]];
        EndEdgeValues[i] = Values[EndEdges[i]];
        if (StartEdgeValues[i] > EndEdgeValues[i]) swap(StartEdgeValues[i], EndEdgeValues[i]);
        EndPrefixMaxValues[i] = i == 0 ? EndEdgeValues[i] : max(EndPrefixMaxValues[i - 1], EndEdgeValues[i]);
    }

    cout << "pleale input the scalar value you want to find:";
    double target_val;
    cin >> target_val;
    if (target_val < min_val || target_val > max_val) {
        cout << "the scalar value you want to find is out of range" << endl;
        return 0;
    }
    int pos_end = upper_bound(StartEdgeValues, StartEdgeValues + EdgeNum, target_val) - StartEdgeValues;
    int pos_start = lower_bound(EndPrefixMaxValues, EndPrefixMaxValues + EdgeNum, target_val) - EndPrefixMaxValues;
    cout << "find range:[" << pos_start << ", " << pos_end << ")" << endl;
    streampos pos = ifs.tellg();
    for (int i = pos_start; i < pos_end; i++) {
        if (EndEdgeValues[i] < target_val) continue;
        cout << "find edge:[" << StartEdges[i] << ", " << EndEdges[i] << "] :";
        vector<int> augCTEdges;
        ifs.seekg(EdgeOffsets[i] * (MAX_NUMBER_LEN + 1), ios::cur);
        int EdgeNum = EdgeOffsets[i + 1] - EdgeOffsets[i];
        for (int j = 0; j < EdgeNum; j++) {
            int augCTEdge;
            ifs >> augCTEdge;
            augCTEdges.push_back(augCTEdge);
        }
        for (int j = 0; j < augCTEdges.size(); j++) {
            if (j != augCTEdges.size() - 1) cout << augCTEdges[j] << " -> ";
            else cout << augCTEdges[j] << endl;
        }
        ifs.seekg(- EdgeOffsets[i + 1] * (MAX_NUMBER_LEN + 1), ios::cur);
    }

    return 0;
}