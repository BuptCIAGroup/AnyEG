#include <bits/stdc++.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include "helper_cuda.h"
#include "helper_string.h"
#include "stopwatch.h"


using namespace std;


// 6 个面相连的邻居：左右、前后、上下
int h_dx[14] = {
    // dz = -1 层 (4 个)
      0,  1,  0,  1,
      // dz =  0 层 (6 个)
        0,  1,  1, -1, -1,  0,
        // dz = +1 层 (4 个)
         -1,  0,  0, -1
};
int h_dy[14] = {
    // dz=-1
   -1, -1,  0,  0,
   // dz= 0
  -1, -1,  0,  0,  1,  1,
  // dz=+1
  0,  0,  1,  1
};
int h_dz[14] = {
    // dz = -1
     -1, -1, -1, -1,
     // dz =  0
       0,  0,  0,  0,  0,  0,
       // dz = +1
         1,  1,  1,  1
};
int* d_dx, * d_dy, * d_dz;
int dims[3];
__host__ __device__
void getNodeId(int& nodeid, int x, int y, int z,
    int Nx, int Ny) {
    // Nz 不直接用在算式里
    nodeid = x + y * Nx + z * (Nx * Ny);
}

__host__ __device__
void getCoords(int nodeid, int& x, int& y, int& z,
    int Nx, int Ny) {
    int xy = Nx * Ny;
    z = nodeid / xy;
    int rem = nodeid % xy;
    y = rem / Nx;
    x = rem % Nx;
}


void mycheck() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

class GPUTimer {
public:
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void startTimer() { cudaEventRecord(start, 0); }
    void stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
    }
    float getElapsedTime() {
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        return elapsedTime;
    }

private:
    cudaEvent_t start, stop;
};

class CPUTimer {
public:
    CPUTimer() : start_time(), end_time() {}
    void startTimer() { start_time = std::chrono::high_resolution_clock::now(); }
    void stopTimer() { end_time = std::chrono::high_resolution_clock::now(); }
    float getElapsedTime() const {
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};

template <typename KernelFunc, typename... Args>
float measureKernelTime(KernelFunc kernel, dim3 gridSize, dim3 blockSize, Args... args) {
    GPUTimer timer;
    timer.startTimer();
    kernel << <gridSize, blockSize >> > (args...);
    timer.stopTimer();
    return timer.getElapsedTime();
}
template <typename Func, typename... Args>
float measureFunctionTime(Func func, Args... args) {
    CPUTimer timer;
    timer.startTimer();
    func(args...);
    timer.stopTimer();
    return timer.getElapsedTime();
}

template <typename T>
__device__ void cuswap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}
/**
 *  变量
 */

 // 分区数量
int32_t MaxNeighborNum = 14;
// 总点数
int32_t N;
// 极值点数量
int32_t ExtremePointNum = 0;
// 候选鞍点数量
int32_t CandidateSaddlePointNum = 0;
// 保留的候选鞍点数量
int32_t RealCandidateSaddlePointNum = 0;
// 合并树包含边数
int32_t SplitTreeNum = 0;
/**
 *  常量
 */


/**
 *  数组
 */

 // 顶点标量值
float* Values;
// 顶点邻居集合
int32_t* NeighborNums;
int32_t* Neighbors;
int32_t* Neighbors1;
int32_t* Neighbors2;
int32_t* CandidateSaddleNeighbors;
// 极值点集合
int32_t* ExtremePoints;
int32_t* SplitExtremePoints;
int32_t* JoinExtremePoints;

int32_t* FatherLinks_cpy;
int* cpuCandidateSaddlePoints;
int32_t* FatherLinks;
// 顶点所属极值点，一个
int32_t* ExtremeIndices;
// 候选鞍点集合
int32_t* CandidateSaddlePoints;
// 候选鞍点能够连接到的极值点集合
int32_t* CandidateSaddleConnectionNum;
int32_t** CandidateSaddleConnections;
int32_t* CandidateSaddleConnections_New;
// Hash 表

unsigned long long int* HashTable_Key;
int32_t* HashValue;






// 极值点判定
__global__ void judgeExtremePoints_New(int MaxNeighborNum,
    int p,
    int size,
    int N,
    float* Values,
    int32_t* NeighborNums,
    int32_t* Neighbors,
    int32_t* ExtremePoints,
    int32_t* FatherLinks,
    int IsJoinTree) {

    int64_t vertexId = p * size + blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexId >= N) return;

    int numNeighbors = NeighborNums[vertexId];
    bool isExtremePoint = true;

    int32_t validNeighbor = -1;
    float maxGradient = -1.0f;  // 记录最大梯度差

    float vVal = Values[vertexId];
    int32_t baseIdx = vertexId * MaxNeighborNum;

    for (int i = 0; i < numNeighbors; i++) {
        int32_t nei = Neighbors[baseIdx + i];
        float nVal = Values[nei];

        if (IsJoinTree == 0) {
            // 对于 join-tree，极值是局部最小
            if (vVal > nVal) {
                isExtremePoint = false;
                float gradient = vVal - nVal;  
                if (gradient > maxGradient) {
                    maxGradient = gradient;
                    validNeighbor = nei;
                }
            }
        }
        else {
            // 对于 split-tree，极值是局部最大
            if (vVal < nVal) {
                isExtremePoint = false;
                float gradient = nVal - vVal;
                if (gradient > maxGradient) {
                    maxGradient = gradient;
                    validNeighbor = nei;
                }
            }
        }
    }

    //if (isExtremePoint && numNeighbors == MaxNeighborNum) {
    if (isExtremePoint) {
        ExtremePoints[vertexId] = vertexId;
        FatherLinks[vertexId] = vertexId;
    } else {
        ExtremePoints[vertexId] = -1;
        // 如果没有任何“下降”或“上升”方向（validNeighbor 仍为 -1），就指向自身
        FatherLinks[vertexId] = (validNeighbor == -1) ? vertexId : validNeighbor;
    }
}

__global__ void judgeExtremePoints_Coords(
    // 网格划分参数
    int p,
    int size,
    int N,
    // 设备端也能访问的网格维度
    int Nx,
    int Ny,
    int Nz,
    // 标量值数组
    const __half* __restrict__ Values,
    // 输出：极值点标记 & 父链接
    bool* __restrict__ isExtremePoint,
    int32_t* __restrict__ FatherLinks,
    // 判断合并树还是分裂树
    int IsJoinTree,
    // 14 个偏移数组
    const int* __restrict__ dx,
    const int* __restrict__ dy,
    const int* __restrict__ dz
) {
    // 全局顶点 id
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int vertexId = p * size + lane;
    if (vertexId >= N) return;

    // 恢复三维坐标
    int x, y, z;
    getCoords(vertexId, x, y, z, Nx, Ny);

    bool isExtreme = true;
    int  validNeighbor = -1;

    // 遍历 14 个偏移
    for (int k = 0; k < 14; ++k) {
        int xn = x + dx[k];
        int yn = y + dy[k];
        int zn = z + dz[k];
        // 边界检查
        if (xn < 0 || xn >= Nx ||
            yn < 0 || yn >= Ny ||
            zn < 0 || zn >= Nz)
            continue;

        // 计算邻居 nodeId
        int neiId;
        getNodeId(neiId, xn, yn, zn, Nx, Ny);

        float v0 = Values[vertexId];
        float v1 = Values[neiId];

        if (IsJoinTree == 0) {
            // split tree：找局部最小值
            if (v0 > v1) {
                isExtreme = false;
                // 更新 validNeighbor 为最小的比 v0 更小的邻居 id
                validNeighbor = (validNeighbor < 0)
                    ? neiId
                    : min(validNeighbor, neiId);
            }
        }
        else {
            // join tree：找局部最大值
            if (v0 < v1) {
                isExtreme = false;
                // 更新 validNeighbor 为最大的比 v0 更大的邻居 id
                validNeighbor = (validNeighbor < 0)
                    ? neiId
                    : max(validNeighbor, neiId);
            }
        }
    }

    if (isExtreme) {
        isExtremePoint[vertexId] = true;
        FatherLinks[vertexId] = vertexId;
    }
    else {
        isExtremePoint[vertexId] = false;
        FatherLinks[vertexId] = (validNeighbor < 0
            ? vertexId
            : validNeighbor);
    }
}



// 指针加倍
__global__ void updateFa(int32_t* FatherLinks, int32_t N) {
    int vertexId = blockIdx.x * blockDim.x + threadIdx.x;  // 计算顶点的 id
    if (vertexId < N) {
        int fatherLink = FatherLinks[vertexId];
        FatherLinks[vertexId] = FatherLinks[fatherLink];
    }
}

// 更新顶点所属极值点
__global__ void updateExtremeIndices_New(int32_t* FatherLinks, int32_t* ExtremeIndices, int32_t N) {
    int vertexId = blockIdx.x * blockDim.x + threadIdx.x;  // 计算顶点的 id
    if (vertexId < N) {
        ExtremeIndices[vertexId] = FatherLinks[vertexId];
    }

}
// 判定候选鞍点
__global__ void judgeCandidateSaddlePoints_New(int MaxNeighborNum,
    int p,
    int size,
    int N,
    float* Values,
    int32_t* NeighborNums,
    int32_t* Neighbors,
    int32_t* ExtremeIndices,
    int32_t* CandidateSaddlePoints,
    int IsJoinTree) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vertexId = p * size + tid;  // 计算顶点的 id
    if (vertexId < N) {
        int numNeighbors = NeighborNums[vertexId];  // 获取该顶点的邻居数量

        // 定义共享变量，存储每个线程的判断结果
        bool isCandidateSaddlePoint = false;

        // 判断是否是候选鞍点
        for (int i = 0; i < numNeighbors && !isCandidateSaddlePoint; i++) {
            int neighborId = Neighbors[tid * MaxNeighborNum + i];
            if (IsJoinTree == 0) {
                if (Values[vertexId] > Values[neighborId] && ExtremeIndices[vertexId] != ExtremeIndices[neighborId]) {
                    isCandidateSaddlePoint = true;  // 是候选鞍点
                }
            }
            else {
                if (Values[vertexId] < Values[neighborId] && ExtremeIndices[vertexId] != ExtremeIndices[neighborId]) {
                    isCandidateSaddlePoint = true;  // 是候选鞍点
                }
            }
        }

        // 写入候选鞍点的标记
        if (isCandidateSaddlePoint == true) {
            CandidateSaddlePoints[vertexId] = vertexId;  // 将候选鞍点写入数组中
        }
        else {
            CandidateSaddlePoints[vertexId] = -1;  // 不是候选鞍点
        }
    }
}
// judgeCandidateSaddlePoints_Coords：自行计算 14 邻居
__global__ void judgeCandidateSaddlePoints_Coords(
    int p,
    int size,
    int N,
    int Nx,
    int Ny,
    int Nz,
    const __half* __restrict__ Values,
    const int32_t* __restrict__ ExtremeIndices,
    bool* __restrict__ isCandidateSaddlePoint,
    int IsJoinTree,
    const int* __restrict__ dx,
    const int* __restrict__ dy,
    const int* __restrict__ dz
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vertexId = p * size + tid;
    if (vertexId >= N) return;

    // 恢复三维坐标
    int x, y, z;
    getCoords(vertexId, x, y, z, Nx, Ny);

    bool isCandidate = false;
    float v0 = Values[vertexId];
    int   e0 = ExtremeIndices[vertexId];

    // 遍历 14 个偏移
    for (int k = 0; k < 14; ++k) {
        int xn = x + dx[k];
        int yn = y + dy[k];
        int zn = z + dz[k];
        // 边界检查
        if (xn < 0 || xn >= Nx ||
            yn < 0 || yn >= Ny ||
            zn < 0 || zn >= Nz)
            continue;

        int neiId;
        getNodeId(neiId, xn, yn, zn, Nx, Ny);
        float v1 = Values[neiId];
        int   e1 = ExtremeIndices[neiId];

        // 如果 value 和 extreme-region 都满足条件，标记为候选
        if (IsJoinTree == 0) {
            // split tree (判断鞍点需要下溯)：
            if (v0 > v1 && e0 != e1) {
                isCandidate = true;
                break;
            }
        }
        else {
            // join tree (判断鞍点需要上溯)：
            if (v0 < v1 && e0 != e1) {
                isCandidate = true;
                break;
            }
        }
    }

    isCandidateSaddlePoint[vertexId] = isCandidate;
}



// 并查集查找（带路径压缩）
__device__ int uf_find(int x, int parent[]) {
    return (parent[x] == x) ? (x) : (parent[x] = uf_find(parent[x], parent));
}


// ---------------------------------------------
// 1. Device kernel：纯邻居表版 judgeCandidateSaddlePoints_ByMorse
// ---------------------------------------------
__global__ void judgeCandidateSaddlePoints_ByMorse(
    int MaxNei,
    int startIdx,
    int N,
    const float* __restrict__ Values,
    const int* __restrict__ NeighborNums,
    const int* __restrict__ Neighbors,
    int* __restrict__ CandidateSaddlePoints) {
    int tidInGrid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = startIdx + tidInGrid;
    if (a >= N) return;
    // 如果已经被标记为非候选，直接跳过
    if (CandidateSaddlePoints[a] == -1) return;

    // 共享内存布局： per-thread 3×MaxNei ints
    extern __shared__ int shared[];
    int perThreadInts = 3 * MaxNei;
    int base = threadIdx.x * perThreadInts;
    int* gNei = shared + base;            // [0..m-1]
    int* parent = shared + base + MaxNei;   // [0..m-1]
    int* tmp = shared + base + 2 * MaxNei; // 临时保存某邻居的邻居列表

    // 1. 收集所有比 a 大的邻居 v
    int deg = NeighborNums[a];
    int m = 0;
    for (int i = 0; i < deg; ++i) {
        int v = Neighbors[a * MaxNei + i];
        if (Values[v] > Values[a]) {
            gNei[m++] = v;
        }
    }
    if (m <= 1) {
        CandidateSaddlePoints[a] = -1;
        return;
    }

    // 2. 初始化并查集 parent[i] = i
    for (int i = 0; i < m; ++i) {
        parent[i] = i;
    }

    // 3. 对每个高值邻居 u = gNei[i]，
    //    扫描它的所有邻居 tmp[0..t-1]，
    //    然后对 gNei[j] 做线性查找，若匹配就 union(i,j)
    for (int i = 0; i < m; ++i) {
        int u = gNei[i];
        int du = NeighborNums[u];
        int t = 0;
        // 把 u 的所有邻居拷到 tmp[]
        for (int k = 0; k < du; ++k) {
            tmp[t++] = Neighbors[u * MaxNei + k];
        }
        // 在 tmp 中找 gNei[j]
        for (int j = i + 1; j < m; ++j) {
            int w = gNei[j];
            for (int k = 0; k < t; ++k) {
                if (tmp[k] == w) {
                    // 找到连通，合并并查集
                    int ri = uf_find(i, parent);
                    int rj = uf_find(j, parent);
                    if (ri != rj) parent[ri] = rj;
                    break;
                }
            }
        }
    }

    // 4. 统计并查集根的数量 = 连通分量数
    int comps = 0;
    for (int i = 0; i < m; ++i) {
        if (uf_find(i, parent) == i) ++comps;
    }

    // 5. 如果分量数 <= 1，则不是鞍点候选；否则保留 a
    CandidateSaddlePoints[a] = (comps <= 1 ? -1 : a);
}

// ---------------------------------------------
// 2. Host 启动：与之前一样，只是删掉 dims 参数
// ---------------------------------------------
void launchJudgeSaddle(
    int MaxNei,
    int N,
    const float* d_Values,
    const int* d_NeighborNums,
    const int* d_Neighbors,
    int* d_CandidateSaddlePoints,
    double& costTime) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 每线程需要的共享内存大小：3×MaxNei ints
    size_t perThShm = 3 * MaxNei * sizeof(int);

    // 用 Occupancy API 计算推荐的 block size
    int minGridSize = 0, occBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &occBlockSize,
        judgeCandidateSaddlePoints_ByMorse,
        0,      // dynamic shared memory per block
        N       // 估算的线程总数
    );
    // 再受限于设备最大共享内存
    int threadsPerBlock = std::min(
        occBlockSize,
        int(prop.sharedMemPerBlock / perThShm)
    );
    threadsPerBlock = std::max(threadsPerBlock, 1);
    size_t shmBytes = threadsPerBlock * perThShm;

    int fullBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int maxBlocks = prop.maxGridSize[0];

    // 创建计时事件
    cudaEvent_t evtStart, evtStop;
    cudaEventCreate(&evtStart);
    cudaEventCreate(&evtStop);

    float totalMs = 0.0f;
    int processed = 0;
    while (processed < N) {
        int remain = N - processed;
        int thisBlocks = std::min(fullBlocks, maxBlocks);
        int thisCount = std::min(remain, thisBlocks * threadsPerBlock);

        cudaEventRecord(evtStart);
        judgeCandidateSaddlePoints_ByMorse
            << < thisBlocks, threadsPerBlock, shmBytes >> > (
                MaxNei,
                /*startIdx=*/             processed,
                /*N=*/                    N,
                /*Values=*/               d_Values,
                /*NeighborNums=*/         d_NeighborNums,
                /*Neighbors=*/            d_Neighbors,
                /*CandidateSaddlePoints=*/d_CandidateSaddlePoints
                );
        cudaEventRecord(evtStop);

        cudaEventSynchronize(evtStop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evtStart, evtStop);
        totalMs += ms;

        processed += thisCount;
        std::cout << "Processed: "
            << processed << " / " << N << std::endl;
    }

    cudaEventDestroy(evtStart);
    cudaEventDestroy(evtStop);

    std::cout << "第四模块执行时间: "
        << (totalMs / 1000.0) << " s\n";
    costTime += totalMs / 1000.0;
}

// ---------------------------------------------
// 1. Device kernel：基于坐标自行计算邻居的鞍点候选判定
// ---------------------------------------------
__global__ void judgeCandidateSaddlePoints_ByMorse_Coords(
    int startIdx,                       // 本轮处理起始顶点下标
    int N,                              // 顶点总数
    const __half* __restrict__ Values,   // 标量值数组
    bool* __restrict__ isCandidateSaddlePoint, // 输出：候选鞍点标记
    int Nx, int Ny, int Nz,             // 网格尺寸
    const int* __restrict__ dx,         // 14 个偏移
    const int* __restrict__ dy,
    const int* __restrict__ dz
) {
    // 全局线程与顶点映射
    int tidInGrid = blockIdx.x * blockDim.x + threadIdx.x;
    int a = startIdx + tidInGrid;
    if (a >= N) return;
    if (isCandidateSaddlePoint[a] == false)return;

    // 共享内存：每线程两个长度 14 的数组
    extern __shared__ int shared[];
    const int MAXNB = 14;
    int perThread = 2 * MAXNB;
    int base = threadIdx.x * perThread;
    int* gNei = shared + base;              // 存放高值邻居的 nodeId
    int* parent = shared + base + MAXNB;      // 并查集父指针

    // 把 a 转为三维坐标
    int xa, ya, za;
    getCoords(a, xa, ya, za, Nx, Ny);
    __half va = Values[a];

    // 1. 收集值比 va 更大的邻居
    int m = 0;
    for (int k = 0; k < MAXNB; ++k) {
        int xn = xa + dx[k];
        int yn = ya + dy[k];
        int zn = za + dz[k];
        // 边界检查
        if (xn < 0 || xn >= Nx ||
            yn < 0 || yn >= Ny ||
            zn < 0 || zn >= Nz) continue;
        int v;
        getNodeId(v, xn, yn, zn, Nx, Ny);
        if (Values[v] > va) {
            gNei[m++] = v;
        }
    }
    if (m <= 1) {
        isCandidateSaddlePoint[a] = false;
        return;
    }

    // 2. 初始化并查集
    for (int i = 0; i < m; ++i) {
        parent[i] = i;
    }

    // 3. 对所有高值邻居两两做连通性合并（面相连或体对角）
    for (int i = 0; i < m; ++i) {
        int ui = gNei[i];
        int xi, yi, zi;
        getCoords(ui, xi, yi, zi, Nx, Ny);
        for (int j = i + 1; j < m; ++j) {
            int wj = gNei[j];
            int xj, yj, zj;
            getCoords(wj, xj, yj, zj, Nx, Ny);

            int dx0 = (xi - xj);
            int dy0 = (yi - yj);
            int dz0 = (zi - zj);
            if (abs(dx0) > 1 || abs(dy0) > 1 || abs(dz0) > 1)continue;

            bool z0 = (dx0 != -1 && dy0 != 1 && dz0 == -1);
            bool z1 = (dx0 != dy0 && dz0 == 0);
            bool z2 = (dx0 != 1 && dy0 != -1 && dz0 == 1);
            if (z0 || z1 || z2) {
                int ri = uf_find(i, parent);
                int rj = uf_find(j, parent);
                if (ri != rj) parent[ri] = rj;
            }
        }
    }

    // 4. 统计连通分量数
    int comps = 0;
    for (int i = 0; i < m; ++i) {
        if (uf_find(i, parent) == i) ++comps;
    }

    // 5. 输出：连通分量数 > 1 即为鞍点候选
    isCandidateSaddlePoint[a] = true;
}
void launchJudgeSaddle_Coords(
    int N,
    const __half* d_Values,
    bool* isCandidateSaddlePoint,
    int          Nx, int Ny, int Nz,
    double& costTime
) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 每线程需要的共享内存：2×14 ints
    const int MAXNB = 14;
    size_t perThreadShm = 2 * MAXNB * sizeof(int);

    // 计算 occupancy 推荐的 blockSize
    int minGridSize = 0, occBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &occBlockSize,
        judgeCandidateSaddlePoints_ByMorse_Coords,
        0,     // dynamic shared mem per block
        N      // 线程总数估计
    );
    // 再受限于硬件共享内存
    int threadsPerBlock = std::min(
        occBlockSize,
        int(prop.sharedMemPerBlock / perThreadShm)
    );
    threadsPerBlock = std::max(threadsPerBlock, 1);
    size_t shmBytes = threadsPerBlock * perThreadShm;

    int fullBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int maxBlocks = prop.maxGridSize[0];

    // CUDA 事件计时
    cudaEvent_t evtStart, evtStop;
    cudaEventCreate(&evtStart);
    cudaEventCreate(&evtStop);

    float totalMs = 0.0f;
    int processed = 0;
    {
        Stopwatch sw(true);
        while (processed < N) {
            int remain = N - processed;
            int thisBlocks = std::min(fullBlocks, maxBlocks);
            int thisCount = std::min(remain, thisBlocks * threadsPerBlock);

            cudaEventRecord(evtStart);
            judgeCandidateSaddlePoints_ByMorse_Coords
                << < thisBlocks, threadsPerBlock, shmBytes >> > (
                    /*startIdx=*/              processed,
                    /*N=*/                     N,
                    /*Values=*/                d_Values,
                    /*CandidateSaddlePoints=*/ isCandidateSaddlePoint,
                    /*Nx=*/                     Nx,
                    /*Ny=*/                     Ny,
                    /*Nz=*/                     Nz,
                    /*dx=*/                     d_dx,
                    /*dy=*/                     d_dy,
                    /*dz=*/                     d_dz
                    );
            cudaEventRecord(evtStop);

            cudaEventSynchronize(evtStop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evtStart, evtStop);
            totalMs += ms;

            processed += thisCount;
            printf("Processed: %d / %d\n", processed, N);
        }
        sw.stop();
        cout << "第四模块执行时间: " << totalMs / 1000.0 << endl;
    }

    cudaEventDestroy(evtStart);
    cudaEventDestroy(evtStop);

    costTime += totalMs / 1000.0;
}

// 并查集——带路径压缩的 find
int findSet(int x, std::vector<int>& parent) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}
// 并查集合并
void unionSet(int x, int y, std::vector<int>& parent) {
    int rx = findSet(x, parent);
    int ry = findSet(y, parent);
    if (rx != ry) parent[ry] = rx;
}
inline int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
void checkNeighborsClamped(
    int Nx, int Ny, int Nz,
    int MaxNei,
    int N,
    const int* NeighborNums,
    const int* Neighbors   // layout: [ nodeId * MaxNei + i ]
) {
    int bad = 0;
    for (int id = 0; id < N; ++id) {
        // 恢复三维坐标
        int x, y, z;
        getCoords(id, x, y, z, Nx, Ny);

        // 1) Clamp‐of‐bound 重建邻居
        std::vector<int> rebuild;
        rebuild.reserve(14);
        for (int k = 0; k < 14; ++k) {
            int xn = clamp(x + h_dx[k], 0, Nx - 1);
            int yn = clamp(y + h_dy[k], 0, Ny - 1);
            int zn = clamp(z + h_dz[k], 0, Nz - 1);

            int nid;
            getNodeId(nid, xn, yn, zn, Nx, Ny);
            if (nid == id) continue;         // 删掉 self
            rebuild.push_back(nid);
        }
        std::sort(rebuild.begin(), rebuild.end());
        rebuild.erase(std::unique(rebuild.begin(), rebuild.end()),
            rebuild.end());

        // 2) 读出原始 Neighbor 列表
        int deg = NeighborNums[id];
        std::vector<int> listed;
        listed.reserve(deg);
        for (int i = 0; i < deg; ++i) {
            listed.push_back(Neighbors[id * MaxNei + i]);
        }
        std::sort(listed.begin(), listed.end());

        // 3) 差集对比
        std::vector<int> onlyInRebuild, onlyInListed;
        std::set_difference(
            rebuild.begin(), rebuild.end(),
            listed.begin(), listed.end(),
            std::back_inserter(onlyInRebuild)
        );
        std::set_difference(
            listed.begin(), listed.end(),
            rebuild.begin(), rebuild.end(),
            std::back_inserter(onlyInListed)
        );

        if (!onlyInRebuild.empty() || !onlyInListed.empty()) {
            ++bad;
            int xa, ya, za;
            getCoords(id, xa, ya, za, Nx, Ny);
            std::cout << "Node " << id
                << " (" << xa << "," << ya << "," << za << ") mismatch:\n";

            if (!onlyInListed.empty()) {
                std::cout << "  In original list but not in rebuild:";
                for (int v : onlyInListed) {
                    int vx, vy, vz;
                    getCoords(v, vx, vy, vz, Nx, Ny);
                    std::cout << " (" << vx << "," << vy << "," << vz << ")";
                }
                std::cout << "\n";
            }
            if (!onlyInRebuild.empty()) {
                std::cout << "  In rebuild but not in original list:";
                for (int v : onlyInRebuild) {
                    int vx, vy, vz;
                    getCoords(v, vx, vy, vz, Nx, Ny);
                    std::cout << " (" << vx << "," << vy << "," << vz << ")";
                }
                std::cout << "\n";
            }
        }
    }

    std::cout << "Total mismatched nodes: " << bad
        << " / " << N << std::endl;
}

// 打印一个点的 coordinate 和 它的 neighbor 列表（coord 形式）
void dumpPointNeighbors(int id,
    int Nx, int Ny, int Nz,
    int MaxNei,
    const int* NeighborNums,
    const int* Neighbors) {
    int x, y, z;
    getCoords(id, x, y, z, Nx, Ny);
    std::cout << "Point " << id
        << " coord=(" << x << "," << y << "," << z << ")\n";
    int deg = NeighborNums[id];
    std::cout << "  degree = " << deg << "\n";
    std::cout << "  neighbors:\n";
    for (int i = 0;i < deg;++i) {
        int nid = Neighbors[id * MaxNei + i];
        int xn, yn, zn;
        getCoords(nid, xn, yn, zn, Nx, Ny);
        std::cout << "    [" << i << "] id=" << nid
            << " coord=(" << xn << "," << yn << "," << zn << ")\n";
    }
    std::cout << std::endl;
}

void printSpecialNeighbors(
    int Nx, int Ny, int Nz,
    int MaxNei, int N,
    const int* NeighborNums,
    const int* Neighbors
) {
    std::vector<std::tuple<int, int, int>> corners;
    // 顶角 8 个
    corners.push_back({ 0,0,0 });
    corners.push_back({ Nx - 1,0,0 });
    corners.push_back({ 0,Ny - 1,0 });
    corners.push_back({ Nx - 1,Ny - 1,0 });
    corners.push_back({ 0,0,Nz - 1 });
    corners.push_back({ Nx - 1,0,Nz - 1 });
    corners.push_back({ 0,Ny - 1,Nz - 1 });
    corners.push_back({ Nx - 1,Ny - 1,Nz - 1 });


    // 每条棱的中点 (12 条棱)
    std::vector<std::tuple<int, int, int>> edgeMids;
    edgeMids.reserve(12);
    // 1–4: 底面 z = 0
    edgeMids.push_back({ Nx / 2,     0,      0 });  // 底面前边中点
    edgeMids.push_back({ 0, Ny / 2,      0 });  // 底面左边中点
    edgeMids.push_back({ Nx / 2, Ny - 1,      0 });  // 底面后边中点
    edgeMids.push_back({ Nx - 1, Ny / 2,      0 });  // 底面右边中点

    // 5–8: 顶面 z = Nz-1
    edgeMids.push_back({ Nx / 2,     0, Nz - 1 });  // 顶面前边中点
    edgeMids.push_back({ 0, Ny / 2, Nz - 1 });  // 顶面左边中点
    edgeMids.push_back({ Nx / 2, Ny - 1, Nz - 1 });  // 顶面后边中点
    edgeMids.push_back({ Nx - 1, Ny / 2, Nz - 1 });  // 顶面右边中点

    // 9–12: 四条垂直棱的中点
    edgeMids.push_back({ 0,     0, Nz / 2 });  // 前左竖棱中点
    edgeMids.push_back({ Nx - 1,     0, Nz / 2 });  // 前右竖棱中点
    edgeMids.push_back({ 0, Ny - 1, Nz / 2 });  // 后左竖棱中点
    edgeMids.push_back({ Nx - 1, Ny - 1, Nz / 2 });  // 后右竖棱中点


    // 内部点示例
    std::vector<std::tuple<int, int, int>> interiors;
    interiors.push_back({ Nx / 2, Ny / 2, Nz / 2 });           // 中心
    interiors.push_back({ 1,   1,   1 });              // 靠近原点
    interiors.push_back({ Nx / 2, Ny / 3, Nz / 4 });           // 随机位置
    interiors.push_back({ Nx / 3, Ny / 2, Nz * 3 / 4 });         // 随机位置

    std::cout << "=== Corners ===\n";
    for (size_t i = 0; i < corners.size(); ++i) {
        int x = std::get<0>(corners[i]);
        int y = std::get<1>(corners[i]);
        int z = std::get<2>(corners[i]);
        int id;
        getNodeId(id, x, y, z, Nx, Ny);
        dumpPointNeighbors(id, Nx, Ny, Nz, MaxNei,
            NeighborNums, Neighbors);
    }


    std::cout << "=== Edge mids ===\n";
    for (size_t i = 0; i < edgeMids.size(); ++i) {
        int x = std::get<0>(edgeMids[i]);
        int y = std::get<1>(edgeMids[i]);
        int z = std::get<2>(edgeMids[i]);
        int id;
        getNodeId(id, x, y, z, Nx, Ny);
        dumpPointNeighbors(id, Nx, Ny, Nz, MaxNei,
            NeighborNums, Neighbors);
    }

    std::cout << "=== Interior ===\n";
    for (size_t i = 0; i < interiors.size(); ++i) {
        int x = std::get<0>(interiors[i]);
        int y = std::get<1>(interiors[i]);
        int z = std::get<2>(interiors[i]);
        int id;
        getNodeId(id, x, y, z, Nx, Ny);
        dumpPointNeighbors(id, Nx, Ny, Nz, MaxNei,
            NeighborNums, Neighbors);
    }

}
// ---------------------------------------------
// 1. “半立方体”14方向偏移（dx, dy, dz）
// ---------------------------------------------
static const int dx14[14] = {
    // dz = -1 层 (4 个)
      0,  1,  0,  1,
      // dz =  0 层 (6 个)
        0,  1,  1, -1, -1,  0,
        // dz = +1 层 (4 个)
         -1,  0,  0, -1
};
static const int dy14[14] = {
    // dz=-1
   -1, -1,  0,  0,
   // dz= 0
  -1, -1,  0,  0,  1,  1,
  // dz=+1
  0,  0,  1,  1
};
static const int dz14[14] = {
    // dz = -1
     -1, -1, -1, -1,
     // dz =  0
       0,  0,  0,  0,  0,  0,
       // dz = +1
         1,  1,  1,  1
};

// ---------------------------------------------
// 2. 通用验证函数：根据给定偏移重新生成邻居并与原列表对比
// ---------------------------------------------
void checkNeighborsWithOffsets(
    int Nx, int Ny, int Nz,
    int MaxNei, int N,
    const int* NeighborNums,    // [N]
    const int* Neighbors        // [N * MaxNei]
) {
    auto clamp = [&](int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
        };

    int totalBad = 0;
    for (int id = 0; id < N; ++id) {
        // (1) 恢复三维坐标
        int x, y, z;
        getCoords(id, x, y, z, Nx, Ny);

        // (2) 用“半立方体”偏移重建 list
        std::vector<int> rebuild;
        rebuild.reserve(14);
        for (int k = 0; k < 14; ++k) {
            // int xn = clamp(x + dx14[k], 0, Nx - 1);
            // int yn = clamp(y + dy14[k], 0, Ny - 1);
            // int zn = clamp(z + dz14[k], 0, Nz - 1);
            int xn = (x + dx14[k]);
            int yn = (y + dy14[k]);
            int zn = (z + dz14[k]);
            if (xn < 0 || xn >= Nx ||
                yn < 0 || yn >= Ny ||
                zn < 0 || zn >= Nz) continue;
            int nid;
            getNodeId(nid, xn, yn, zn, Nx, Ny);
            if (nid != id) rebuild.push_back(nid);
        }
        std::sort(rebuild.begin(), rebuild.end());
        rebuild.erase(std::unique(rebuild.begin(), rebuild.end()), rebuild.end());

        // (3) 读原列表
        int deg = NeighborNums[id];
        std::vector<int> listed(deg);
        for (int i = 0; i < deg; ++i)
            listed[i] = Neighbors[id * MaxNei + i];
        std::sort(listed.begin(), listed.end());

        // (4) 差集对比
        std::vector<int> onlyInListed, onlyInRebuild;
        std::set_difference(
            listed.begin(), listed.end(),
            rebuild.begin(), rebuild.end(),
            std::back_inserter(onlyInListed)
        );
        std::set_difference(
            rebuild.begin(), rebuild.end(),
            listed.begin(), listed.end(),
            std::back_inserter(onlyInRebuild)
        );

        if (!onlyInListed.empty() || !onlyInRebuild.empty()) {
            ++totalBad;
            int xa, ya, za; getCoords(id, xa, ya, za, Nx, Ny);
            std::cout << "Mismatch at id=" << id
                << " coord=(" << xa << "," << ya << "," << za << ")\n";
            if (!onlyInListed.empty()) {
                std::cout << "  In original but not rebuilt:";
                for (int v : onlyInListed) {
                    int vx, vy, vz; getCoords(v, vx, vy, vz, Nx, Ny);
                    std::cout << "(" << vx << "," << vy << "," << vz << ") ";
                }
                std::cout << "\n";
            }
            if (!onlyInRebuild.empty()) {
                std::cout << "  In rebuilt but not original:";
                for (int v : onlyInRebuild) {
                    int vx, vy, vz; getCoords(v, vx, vy, vz, Nx, Ny);
                    std::cout << "(" << vx << "," << vy << "," << vz << ") ";
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << "Total mismatches: " << totalBad << " / " << N << "\n";
}
__half* Values_fp16;



int countLocalMaxima(
    int32_t N,
    int32_t MaxNeighborNum,
    const float* Values,
    const int32_t* NeighborNums,
    const int32_t* Neighbors)
{
    int32_t count = 0;

    for (int32_t vid = 0; vid < N; ++vid) {
        float vVal = Values[vid];
        int32_t numNei = NeighborNums[vid];
        int32_t baseIdx = vid * MaxNeighborNum;

        bool isMax = true;
        // 遍历所有邻居，若发现有邻居值严格大于自己，则不是局部最大
        for (int32_t i = 0; i < numNei; ++i) {
            int32_t nei = Neighbors[baseIdx + i];
            if (uint8_t(Values[nei]) > uint8_t(vVal)) {
                isMax = false;
                break;
            }
        }

        if (isMax) {
            ++count;
        }
    }

    return count;
}

int* compute(int _N, float* _Values, int* _NeighborNums, int** _Neighbors, int* _dims) {
    dims[0] = _dims[0];
    dims[1] = _dims[1];
    dims[2] = _dims[2];
    cudaMalloc(&d_dx, 14 * sizeof(int));
    cudaMalloc(&d_dy, 14 * sizeof(int));
    cudaMalloc(&d_dz, 14 * sizeof(int));
    cudaMemcpy(d_dx, h_dx, 14 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy, 14 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz, h_dz, 14 * sizeof(int), cudaMemcpyHostToDevice);

    double costTime = 0;
    cout << "Compute Extreme Graph" << endl;

    /**
     * 读取输入数据
     */

     // 读取总点数
    N = _N;
    cout << "N: " << N << endl;


    /*
    // 1) 构建邻接表
    std::vector<std::vector<int>> adj(N);
    for (int i = 0; i < N; i++) {
        adj[i].reserve(_NeighborNums[i]);
        for (int j = 0; j < _NeighborNums[i]; j++) {
            adj[i].push_back(_Neighbors[i][j]);
        }
    }

    // 2) 标记局部最大点
    std::vector<bool> isMax(N, false);
    for (int i = 0; i < N; i++) {
        bool allLower = true;
        for (int u : adj[i]) {
            if (_Values[u] >= _Values[i]) {
                allLower = false;
                break;
            }
        }
        if (allLower) isMax[i] = true;
    }

    // 3) 按标量值降序处理顶点（高 → 低）
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return _Values[a] > _Values[b]; });

    // 并查集初始化，每个点自成一组
    std::vector<int> parent(N);
    std::iota(parent.begin(), parent.end(), 0);

    // 记录哪些点已经“激活”到超水平集中
    std::vector<bool> active(N, false);

    // 输出数组，初始都非鞍点
    cpuCandidateSaddlePoints = new int[N];
    std::fill(cpuCandidateSaddlePoints, cpuCandidateSaddlePoints + N, -1);

    int saddleCount = 0;
    // 遍历所有顶点
    for (int v : order) {
        active[v] = true;

        // 收集所有已激活邻居的连通分量根
        std::set<int> comps;
        for (int u : adj[v]) {
            if (active[u]) {
                comps.insert(findSet(u, parent));
            }
        }

        // 如果有 ≥2 个分量且 v 不是局部最大 → v 是一个鞍点
        if (comps.size() >= 2 && !isMax[v]) {
            cpuCandidateSaddlePoints[v] = v;
            saddleCount++;
        }

        // 把 v 与它所有已激活的邻居合并到同一集合
        for (int u : adj[v]) {
            if (active[u]) {
                unionSet(v, u, parent);
            }
        }
    }

    // 4) 打印鞍点总数
    std::cout << "Saddle point count (excluding maxima): "
        << saddleCount << std::endl;

    */


    //先按标量值排序 相同大小按索引值排序 最后用1、2、3...赋值标量值
    // 读取每个顶点标量值
     checkCudaErrors(cudaMallocManaged((void**)&Values, sizeof(float) * (N + 1)));
    for (int i = 0; i < N; i++) {
        Values[i] = _Values[i];
    }

    // //标量值唯一化
    // // 分配 + 初始化 + 排序
    // Node* Nodes;
    // checkCudaErrors(cudaMallocManaged((void**)&Nodes, sizeof(Node) * (N + 1)));
    // for (int i = 0; i < N; i++) {
    //     Nodes[i].idx = i;
    //     Nodes[i].value = _Values[i];
    // }

    // // 使用 thrust 排序（Unified Memory 支持）
    // thrust::sort(thrust::device, Nodes, Nodes + N, CompareNode());
    // cudaDeviceSynchronize(); // 确保排序完成

    // for (int i = 0; i < N; i++) {
    //     int id = Nodes[i].idx;
    //     Values[id] = i;
    // }

    // 读取顶点邻居集合，并统一存储在连续数组中
    checkCudaErrors(cudaMallocManaged((void**)&NeighborNums, sizeof(int32_t) * (N + 1)));
    int maxNeighborNum = 0;
    for (int i = 0; i < N; i++) {
        NeighborNums[i] = _NeighborNums[i];
        maxNeighborNum = max(maxNeighborNum, NeighborNums[i]);
    }
    MaxNeighborNum = maxNeighborNum;
    cout << "MaxNeighborNum: " << MaxNeighborNum << endl;

    // 分配并填充邻居列表（按顶点展开）
    checkCudaErrors(cudaMallocManaged((void**)&Neighbors, sizeof(int32_t) * N * MaxNeighborNum));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < NeighborNums[i]; j++) {
            Neighbors[i * MaxNeighborNum + j] = _Neighbors[i][j];
        }
        // 对于缺失的邻居，填充 -1
        for (int j = NeighborNums[i]; j < MaxNeighborNum; j++) {
            Neighbors[i * MaxNeighborNum + j] = -1;
        }
    }
    // printSpecialNeighbors(
    //     /*Nx=*/ dims[0], /*Ny=*/ dims[1], /*Nz=*/ dims[2],
    //     /*MaxNei=*/ MaxNeighborNum,
    //     /*N=*/ N,
    //     NeighborNums,
    //     Neighbors
    // );
    // checkNeighborsWithOffsets(
    //     /*Nx=*/ dims[0], /*Ny=*/ dims[1], /*Nz=*/ dims[2],
    //     /*MaxNei=*/ MaxNeighborNum,
    //     /*N=*/ N,
    //     NeighborNums,
    //     Neighbors
    // );

    // exit(0);

    // 假设所有设备数据都已经拷贝回 host，或者直接在 host 上调用：
    // checkNeighborsClamped(
    //     /*Nx=*/ dims[0], /*Ny=*/ dims[1], /*Nz=*/ dims[2],
    //     /*MaxNei=*/ MaxNeighborNum,
    //     /*N=*/ N,
    //     NeighborNums,    // host 上的 NeighborNums 数组
    //     Neighbors       // host 上的 Neighbors 数组
    // );


    // 分配其他辅助数组
    checkCudaErrors(cudaMallocManaged((void**)&ExtremePoints, sizeof(int32_t) * (N + 1)));
    checkCudaErrors(cudaMallocManaged((void**)&FatherLinks, sizeof(int32_t) * (N + 1)));
    checkCudaErrors(cudaMallocManaged((void**)&FatherLinks_cpy, sizeof(int32_t) * (N + 1)));
    checkCudaErrors(cudaMallocManaged((void**)&ExtremeIndices, sizeof(int32_t) * (N + 1)));
    checkCudaErrors(cudaMallocManaged((void**)&CandidateSaddlePoints, sizeof(int32_t) * (N + 1)));

    // 将邻居数据从托管内存拷贝到设备
    // int32_t* NeighborsDev;
    // checkCudaErrors(cudaMalloc((void**)&NeighborsDev, sizeof(int32_t) * N * MaxNeighborNum));
    // checkCudaErrors(cudaMemcpy(NeighborsDev, Neighbors, sizeof(int32_t) * N * MaxNeighborNum, cudaMemcpyHostToDevice));

    double time4 = 0;

    // 第一模块：极值点判定（全局执行）
    for (int i = 0; i < N; i++) {
        ExtremePoints[ExtremePointNum++] = -1;
    }

    // cout<<countLocalMaxima(N,MaxNeighborNum,Values,NeighborNums,Neighbors)<<"<====\n";
    
    {
        Stopwatch module1(true);
        int32_t minGridSize = 0, blockSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, judgeExtremePoints_New, 0, N);
        int32_t gridSize = (N + blockSize - 1) / blockSize;
        judgeExtremePoints_New << <gridSize, blockSize >> > (MaxNeighborNum, 0, N, N, Values, NeighborNums, Neighbors, ExtremePoints, FatherLinks, 1);
        cudaDeviceSynchronize();
        module1.stop();
        // 统计极值点数量并重新排列
        ExtremePointNum = 0;
        for (int i = 0; i < N; i++) {
            if (ExtremePoints[i] != -1) {
                ExtremePoints[ExtremePointNum++] = ExtremePoints[i];
            }
        }
        checkCudaErrors(cudaMemcpy(FatherLinks_cpy, FatherLinks, sizeof(int32_t) * N, cudaMemcpyDeviceToHost));
        cout << "ExtremePointNum: " << ExtremePointNum << endl;
        cout << "第一模块执行时间: " << module1.ms() / 1000.0 << endl;
        costTime += module1.ms() / 1000.0;
    }


    // 第三模块：指针加倍（全局执行）

    {
        Stopwatch module2(true);
        printf("Running pointer doubling...\n");
        for (int stride = 1; stride < N; stride *= 2) {
            int32_t minGridSize = 0, blockSize = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateFa, 0, N);
            int32_t gridSize = (N + blockSize - 1) / blockSize;
            updateFa << <gridSize, blockSize >> > (FatherLinks, N);
            cudaDeviceSynchronize();
        }
        module2.stop();
        cout << "第二模块执行时间: " << module2.ms() / 1000.0 << endl;
        costTime += module2.ms() / 1000.0;
        // 更新每个顶点所属极值点索引
        int32_t minGridSize = 0, blockSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateExtremeIndices_New, 0, N);
        int32_t gridSize = (N + blockSize - 1) / blockSize;
        updateExtremeIndices_New << <gridSize, blockSize >> > (FatherLinks, ExtremeIndices, N);
        cudaDeviceSynchronize();
        //FatherLinks是最终的每个点的所属极值点
    }



    // 第三模块：候选鞍点判定（全局执行）

    {
        Stopwatch module3(true);
        int32_t minGridSize = 0, blockSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, judgeCandidateSaddlePoints_New, 0, N);
        int32_t gridSize = (N + blockSize - 1) / blockSize;
        judgeCandidateSaddlePoints_New << <gridSize, blockSize >> > (MaxNeighborNum, 0, N, N, Values, NeighborNums, Neighbors, ExtremeIndices, CandidateSaddlePoints, 1);
        cudaDeviceSynchronize();
        module3.stop();
        cout << "第三模块执行时间: " << module3.ms() / 1000.0 << endl;
        costTime += module3.ms() / 1000.0;
    }

    CandidateSaddlePointNum = 0;
    for (int i = 0; i < N; i++) {
        if (CandidateSaddlePoints[i] != -1) {
            CandidateSaddlePointNum++;
        }
    }

    // CandidateSaddlePointNum = 0;
    // for (int i = 0; i < N; i++) {
    //     if (FatherLinks[i] != i) {
    //         CandidateSaddlePoints[i]=i;
    //         CandidateSaddlePointNum++;
    //     }
    //     else {
    //         CandidateSaddlePoints[i]=-1;
    //     }
    // }

    cout << "CandidateSaddlePointNum: " << CandidateSaddlePointNum << endl;

    
    // 第四模块：根据Morse理论继续筛选鞍点
    launchJudgeSaddle(MaxNeighborNum, N, Values, NeighborNums, Neighbors, CandidateSaddlePoints, costTime);


    // 统计候选鞍点数量并重新排列
    CandidateSaddlePointNum = 0;
    for (int i = 0; i < N; i++) {
        if (CandidateSaddlePoints[i] != -1) {
            CandidateSaddlePoints[CandidateSaddlePointNum++] = CandidateSaddlePoints[i];  // 更新数组
        }
    }
    cout << "CandidateSaddlePointNum: " << CandidateSaddlePointNum << endl;
   

    cout << "总执行时间：" << costTime << endl;


    // exit(0);
    // 结果导出
    int32_t* result = new int32_t[CandidateSaddlePointNum + ExtremePointNum + N + N + N + 2];
    int32_t result_index = 0;
    result[result_index++] = CandidateSaddlePointNum;
    for (int i = 0; i < CandidateSaddlePointNum; i++) {
        result[result_index++] = CandidateSaddlePoints[i];
    }
    result[result_index++] = ExtremePointNum;
    for (int i = 0; i < ExtremePointNum; i++) {
        result[result_index++] = ExtremePoints[i];
    }
    for (int i = 0; i < N; i++) {
        result[result_index++] = FatherLinks_cpy[i];
    }
    for (int i = 0; i < N; i++) {
        result[result_index++] = FatherLinks[i];
    }
    for (int i = 0; i < N; i++) {
        result[result_index++] = Values[i];
    }
    // for(int i = 0; i < CandidateSaddlePointNum; i++){
    //     int temp= CandidateSaddlePoints[i];
    //     while(FatherLinks[temp]!=temp){
    //         std::cout<<temp<<"->";
    //         temp = FatherLinks_cpy[temp];
    //     }
    //     cout<<temp<<endl;
    // }
    

    return result;
}



int* compute_coords(int _N, float* _Values, int* _dims) {
    dims[0] = _dims[0];
    dims[1] = _dims[1];
    dims[2] = _dims[2];
    cudaMalloc(&d_dx, 14 * sizeof(int));
    cudaMalloc(&d_dy, 14 * sizeof(int));
    cudaMalloc(&d_dz, 14 * sizeof(int));
    cudaMemcpy(d_dx, h_dx, 14 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy, 14 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz, h_dz, 14 * sizeof(int), cudaMemcpyHostToDevice);

    double costTime = 0;
    cout << "Compute Extreme Graph" << endl;

    // 读取总点数
    N = _N;
    cout << (sizeof(int32_t) * N) << "\n";

    // cout << "N: " << N << endl;


    // 读取每个顶点标量值
    // checkCudaErrors(cudaMallocManaged((void**)&Values, sizeof(float) * (N + 1)));
    checkCudaErrors(cudaMallocManaged((void**)&Values_fp16, sizeof(__half) * (N + 1)));
    double costmem = sizeof(__half) * (N + 1)/1024.0/1024.0/1024.0;
    printf("mem cost %.3lfGB\n",costmem);
    for (int i = 0; i < N; i++) {
        // Values[i] = _Values[i];
        Values_fp16[i] = __float2half_rn(_Values[i]);
    }
    cout <<"fp16 done"<<endl;

    // 分配其他辅助数组
    bool* isExtremePoint;
    checkCudaErrors(cudaMallocManaged((void**)&isExtremePoint, sizeof(bool) * (N + 1)));
    costmem +=sizeof(bool) * (N + 1)/1024.0/1024.0/1024.0;
    printf("mem cost %.3lfGB\n",costmem);
    // checkCudaErrors(cudaMallocManaged((void**)&ExtremePoints, sizeof(int32_t) * (N + 1)));
    ExtremePoints = new  int32_t[N + 1];
    checkCudaErrors(cudaMallocManaged((void**)&FatherLinks, sizeof(int32_t) * (N + 1)));
    costmem +=sizeof(int32_t) * (N + 1)/1024.0/1024.0/1024.0;
    printf("mem cost %.3lfGB\n",costmem);
    // checkCudaErrors(cudaMallocManaged((void**)&FatherLinks_cpy, sizeof(int32_t) * (N + 1)));
    FatherLinks_cpy = new int32_t[N + 1];
    // checkCudaErrors(cudaMallocManaged((void**)&ExtremeIndices, sizeof(int32_t) * (N + 1)));
    // checkCudaErrors(cudaMallocManaged((void**)&CandidateSaddlePoints, sizeof(int32_t) * (N + 1)));
    CandidateSaddlePoints = new int32_t[N + 1];
    cout<<"alloc done"<<endl;

    // 第一模块：极值点判定（全局执行）
    {
        Stopwatch module1(true);
        // 计算最优 Block 大小
        int32_t minGridSize = 0, blockSize = 0;
        cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize,
            judgeExtremePoints_Coords,
            1,
            N
        );
        // 计算 Grid 大小
        int32_t gridSize = (N + blockSize - 1) / blockSize;
        // 启动 kernel
        judgeExtremePoints_Coords << <gridSize, blockSize >> > (
            /*p=*/            0,
            /*size=*/         N,
            /*N=*/            N,
            /*Nx=*/           dims[0],
            /*Ny=*/           dims[1],
            /*Nz=*/           dims[2],
            /*Values=*/       Values_fp16,
            /*ExtremePoints=*/isExtremePoint,
            /*FatherLinks=*/  FatherLinks,
            /*IsJoinTree=*/   1,
            /*dx=*/           d_dx,
            /*dy=*/           d_dy,
            /*dz=*/           d_dz
            );
        // 同步与计时
        cudaDeviceSynchronize();
        module1.stop();
        // 统计极值点数量并重新排列
        ExtremePointNum = 0;
        for (int i = 0; i < N; i++) {
            if (isExtremePoint[i]) {
                ExtremePoints[ExtremePointNum++] = i;
            }
        }
        checkCudaErrors(cudaMemcpy(FatherLinks_cpy, FatherLinks, sizeof(int32_t) * N, cudaMemcpyDeviceToHost));
        cout << "ExtremePointNum: " << ExtremePointNum << endl;
        cout << "第一模块执行时间: " << module1.ms() / 1000.0 << endl;
        costTime += module1.ms() / 1000.0;;
    }
    checkCudaErrors(cudaFree(isExtremePoint));
    isExtremePoint = nullptr;
    /**
     * 第二模块：指针加倍（全局执行）
     */
    {
        Stopwatch module2(true);
        printf("Running pointer doubling...\n");
        for (int64_t stride = 1; stride < N; stride *= 2) {
            // cout << "stride: " << stride << endl;
            int32_t minGridSize = 0, blockSize = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateFa, 1, N);
            int32_t gridSize = (N + blockSize - 1) / blockSize;
            updateFa << <gridSize, blockSize >> > (FatherLinks, N);
            cudaDeviceSynchronize();
        }
        module2.stop();
        cout << "第二模块执行时间: " << module2.ms() / 1000.0 << endl;
        costTime += module2.ms() / 1000.0;

        // 更新每个顶点所属极值点索引
        // int32_t minGridSize = 0, blockSize = 0;
        // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, updateExtremeIndices_New, 0, N);
        // int32_t gridSize = (N + blockSize - 1) / blockSize;
        // updateExtremeIndices_New << <gridSize, blockSize >> > (FatherLinks, ExtremeIndices, N);
        // cudaDeviceSynchronize();
    }

    bool* isCandidateSaddlePoint;
    checkCudaErrors(cudaMallocManaged((void**)&isCandidateSaddlePoint, sizeof(bool) * (N + 1)));

    // 第三模块：边界鞍点判定（全局执行）
    {
        Stopwatch module3(true);
        // 1. 用 Occupancy API 选 block 大小
        int32_t minGridSize = 0, blockSize = 0;
        cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize,
            judgeCandidateSaddlePoints_Coords,
            0,
            N
        );
        // 2. 计算 gridSize
        int32_t gridSize = (N + blockSize - 1) / blockSize;
        // 3. Launch kernel
        judgeCandidateSaddlePoints_Coords << <gridSize, blockSize >> > (
            /*p=*/                 0,
            /*size=*/              N,
            /*N=*/                 N,
            /*Nx=*/                dims[0],
            /*Ny=*/                dims[1],
            /*Nz=*/                dims[2],
            /*Values=*/            Values_fp16,
            /*ExtremeIndices=*/    FatherLinks,
            /*CandidateSaddlePoints=*/ isCandidateSaddlePoint,
            /*IsJoinTree=*/        1,
            /*dx=*/                d_dx,
            /*dy=*/                d_dy,
            /*dz=*/                d_dz
            );

        // 4. 同步并计时
        cudaDeviceSynchronize();
        module3.stop();
        cout << "第三模块执行时间: " << module3.ms() / 1000.0 << endl;
        costTime += module3.ms() / 1000.0;
    }


    CandidateSaddlePointNum = 0;
    for (int i = 0; i < N; i++) {
        if (isCandidateSaddlePoint[i]) {
            CandidateSaddlePointNum++;
        }
    }
    cout << "CandidateSaddlePointNum: " << CandidateSaddlePointNum << endl;


    // 第四模块：根据Morse理论继续筛选鞍点
    launchJudgeSaddle_Coords(N, Values_fp16, isCandidateSaddlePoint, dims[0], dims[1], dims[2], costTime);


    // 统计候选鞍点数量并重新排列
    CandidateSaddlePointNum = 0;
    for (int i = 0; i < N; i++) {
        if (isCandidateSaddlePoint[i]) {
            CandidateSaddlePoints[CandidateSaddlePointNum++] = i;  // 更新数组
        }
    }

    cout << "CandidateSaddlePointNum: " << CandidateSaddlePointNum << endl;



    cout << "总执行时间：" << costTime << endl;

    

    // exit(0);
    // 结果导出
    // int32_t* result = new int32_t[CandidateSaddlePointNum + ExtremePointNum + N + 2];
    // int32_t result_index = 0;
    // result[result_index++] = CandidateSaddlePointNum;
    // for (int i = 0; i < CandidateSaddlePointNum; i++) {
    //     result[result_index++] = CandidateSaddlePoints[i];
    // }
    // result[result_index++] = ExtremePointNum;
    // for (int i = 0; i < ExtremePointNum; i++) {
    //     result[result_index++] = ExtremePoints[i];
    // }
    // for (int i = 0; i < N; i++) {
    //     result[result_index++] = FatherLinks_cpy[i];
    // }
    
    printf("Result exporting...\n");
    int32_t* result = new int32_t[0ll+CandidateSaddlePointNum + ExtremePointNum + N + N + N + 2];
    printf("result初始化完成\n");
    int64_t result_index = 0;
    result[result_index++] = CandidateSaddlePointNum;
    for (int i = 0; i < CandidateSaddlePointNum; i++) {
        result[result_index++] = CandidateSaddlePoints[i];
    }

    result[result_index++] = ExtremePointNum;
    for (int i = 0; i < ExtremePointNum; i++) {
        result[result_index++] = ExtremePoints[i];
    }

    for (int i = 0; i < N; i++) {
        result[result_index++] = FatherLinks_cpy[i];
    }

    for (int i = 0; i < N; i++) {
        result[result_index++] = FatherLinks[i];
    }

    // for (int i = 0; i < N; i++) {
    //     result[result_index++] = Values[i];
    // }

    printf("Result exported\n");

    checkCudaErrors(cudaFree(isCandidateSaddlePoint));
    isCandidateSaddlePoint = nullptr;
    checkCudaErrors(cudaFree(Values_fp16));
    Values_fp16 = nullptr;
    checkCudaErrors(cudaFree(FatherLinks));
    FatherLinks = nullptr;

    return result;
}

