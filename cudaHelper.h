#ifndef _CUDA_HEADER_H_
#define _CUDA_HEADER_H_
#endif

int* compute_split_tree_cuda1 (int N, float* Values, int* NeighborNums, int** Neighbors);
int* compute_split_tree_cuda2(int N, float* Values, int* NeighborNums, int** Neighbors, int ExtremePointNum, int* ExtremePoints, 
                                    int CandidateSaddlePointNum, int* CandidateSaddlePoints, int* ExtremeIndices, int* PointIndexs);
int* compute_join_tree_cuda1 (int N, float* Values, int* NeighborNums, int** Neighbors);
int* compute_join_tree_cuda2 (int N, float* Values, int* NeighborNums, int** Neighbors, int ExtremePointNum, int* ExtremePoints, 
                                    int CandidateSaddlePointNum, int* CandidateSaddlePoints, int* ExtremeIndices, int* PointIndexs);
void compute_contour_tree_cuda1(int N, float* Values, int* NeighborNums, int** Neighbors, int* split_tree, int* join_tree, int splitExtremeNum, int* ExtremePoints,
                                     int joinExtremeNum, int* JTExtremePoints);