#include </home/lqm/ttk/core/vtk/ttkExtremeGraphDev/ttkExtremeGraphDev.h>

#include <vtkInformation.h>

#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkSortDataArray.h>
#include <vtkConnectivityFilter.h>
#include <vtkCellData.h>     // 对应 GetCellData()
#include <vtkIntArray.h>     // 对应 vtkIntArray::SafeDownCast

#include <ttkMacros.h>
#include <ttkUtils.h>
#include <math.h>
#include <algorithm>
// #include <stl_map.h>
#include <map>
#include <iomanip>
#include <utility>
#include <fstream>
#include <iomanip>

#include <vtkImageData.h>          // for vtkImageData
#include <vtkStructuredGrid.h>     // for vtkStructuredGrid
#include <vtkUnstructuredGrid.h>   // for vtkUnstructuredGrid
#include <vtkPolyData.h>           // for vtkPolyData
#include <vtkLine.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkDoubleArray.h>

#include </home/lqm/ttk/core/vtk/ttkExtremeGraphDev/cudaHelper.h>
#include "stopwatch.h"
using namespace std;

int* compute(int N, float* Values, int* NeighborNums, int** Neighbors, int* dims);
int* compute_coords(int _N, float* _Values, int* _dims);
#define MAX_NUMBER_LEN 9
struct CTEdge {
  int32_t start, end;
  double val_start, val_end;
};
struct CompareCTEdge {
  bool operator()(const CTEdge& x, const CTEdge& y) const {
    if (x.val_start == y.val_start) return x.start < y.start;
    else return x.val_start < y.val_start;
  }
};

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ (h2 << 1); // 或其他混合方式
  }
};

// A VTK macro that enables the instantiation of this class via ::New()
// You do not have to modify this
vtkStandardNewMacro(ttkExtremeGraphDev);

/**
 * TODO 7: Implement the filter constructor and destructor in the cpp file.
 *
 * The constructor has to specify the number of input and output ports
 * with the functions SetNumberOfInputPorts and SetNumberOfOutputPorts,
 * respectively. It should also set default values for all filter
 * parameters.
 *
 * The destructor is usually empty unless you want to manage memory
 * explicitly, by for example allocating memory on the heap that needs
 * to be freed when the filter is destroyed.
 */
ttkExtremeGraphDev::ttkExtremeGraphDev() {
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

/**
 * TODO 8: Specify the required input data type of each input port
 *
 * This method specifies the required input object data types of the
 * filter by adding the vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE() key to
 * the port information.
 */
int ttkExtremeGraphDev::FillInputPortInformation(int port, vtkInformation* info) {
  if (port == 0) {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    return 1;
  }
  return 0;
}

/**
 * TODO 9: Specify the data object type of each output port
 *
 * This method specifies in the port information object the data type of the
 * corresponding output objects. It is possible to either explicitly
 * specify a type by adding a vtkDataObject::DATA_TYPE_NAME() key:
 *
 *      info->Set( vtkDataObject::DATA_TYPE_NAME(), "vtkUnstructuredGrid" );
 *
 * or to pass a type of an input port to an output port by adding the
 * ttkAlgorithm::SAME_DATA_TYPE_AS_INPUT_PORT() key (see below).
 *
 * Note: prior to the execution of the RequestData method the pipeline will
 * initialize empty output data objects based on this information.
 */
int ttkExtremeGraphDev::FillOutputPortInformation(int port, vtkInformation* info) {
  if (port == 0) {
    info->Set(ttkAlgorithm::SAME_DATA_TYPE_AS_INPUT_PORT(), 0);
    return 1;
  }
  return 0;
}
/**
 * TODO 10: Pass VTK data to the base code and convert base code output to VTK
 *
 * This method is called during the pipeline execution to update the
 * already initialized output data objects based on the given input
 * data objects and filter parameters.
 *
 * Note:
 *     1) The passed input data objects are validated based on the information
 *        provided by the FillInputPortInformation method.
 *     2) The output objects are already initialized based on the information
 *        provided by the FillOutputPortInformation method.
 */
/*int ttkExtremeGraphDev::RequestData(vtkInformation* ttkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector) {

  // Get input object from input vector
  // Note: has to be a vtkDataSet as required by FillInputPortInformation
  vtkDataSet* inputDataSet = vtkDataSet::GetData(inputVector[0]);
  if (!inputDataSet)
    return 0;

  // Get input array that will be processed
  //
  // Note: VTK provides abstract functionality to handle array selections, but
  //       this essential functionality is unfortunately not well documented.
  //       Before you read further, please keep in mind the the TTK developer
  //       team is not responsible for the existing VTK Api ;-)
  //
  //       In a nutshell, prior to the RequestData execution one has to call
  //
  //           SetInputArrayToProcess (
  //               int idx,
  //               int port,
  //               int connection,
  //               int fieldAssociation,
  //               const char *name
  //            )
  //
  //       The parameter 'idx' is often misunderstood: lets say the filter
  //       requires n arrays, then idx enumerates them from 0 to n-1.
  //
  //       The 'port' is the input port index at which the object is connected
  //       from which we want to get the array.
  //
  //       The 'connection' is the connection index at that port (we have to
  //       specify this because VTK allows multiple connections at the same
  //       input port).
  //
  //       The 'fieldAssociation' integer specifies if the array should be taken
  //       from 0: point data, 1: cell data, or 2: field data.
  //
  //       The final parameter is the 'name' of the array.
  //
  //       Example: SetInputArrayToProcess(3,1,0,1,"EdgeLength") will store that
  //                for the 3rd array the filter needs the cell data array named
  //                "EdgeLength" that it will retrieve from the vtkDataObject
  //                at input port 1 (first connection). During the RequestData
  //                method one can then actually retrieve the 3rd array it
  //                requires for its computation by calling
  //                GetInputArrayToProcess(3, inputVector)
  //
  //       If this filter is run within ParaView, then the UI will automatically
  //       call SetInputArrayToProcess (see ContourTreeDev.xml file).
  //
  //       During the RequestData execution one can then retrieve an actual
  //       array with the method "GetInputArrayToProcess".
  vtkDataArray* inputArray = this->GetInputArrayToProcess(0, inputVector);









  // 1. 数组本身的尺寸信息
  vtkIdType numTuples = inputArray->GetNumberOfTuples();     // 元组个数
  int       numComponents = inputArray->GetNumberOfComponents(); // 每个元组的分量数
  std::cout
    << "Array Tuples: " << numTuples
    << ", Components: " << numComponents
    << std::endl;
  int32_t grid_type = 0;
  // 2. 数据集（网格）级别的信息
  vtkDataSet* ds = vtkDataSet::GetData(inputVector[0]);
  // 如果是 vtkImageData（规则规则的像素／体素网格）
  int dims[3];
  if (auto img = vtkImageData::SafeDownCast(ds)) {
    img->GetDimensions(dims);  // x, y, z 三个方向上的点数
    std::cout
      << "ImageData Dimensions: "
      << dims[0] << " × "
      << dims[1] << " × "
      << dims[2]
      << std::endl;
    grid_type = 0;
  }
  // 如果是 vtkStructuredGrid（任意形状但结构化的网格）
  else if (auto sg = vtkStructuredGrid::SafeDownCast(ds)) {
    sg->GetDimensions(dims);
    std::cout
      << "StructuredGrid Dimensions: "
      << dims[0] << " × "
      << dims[1] << " × "
      << dims[2]
      << std::endl;
    grid_type = 1;
  }
  // 其它类型（如 vtkUnstructuredGrid / vtkPolyData 等）就没有“维度”，
  else {
    vtkIdType nPts = ds->GetNumberOfPoints();
    vtkIdType nCell = ds->GetNumberOfCells();
    std::cout
      << "Unstructured / Poly Data — Points: " << nPts
      << ", Cells: " << nCell
      << std::endl;
    grid_type = 2;
  }




  if (inputArray->GetSize() != 1ll * dims[0] * dims[1] * dims[2]&& grid_type == 0) {
    std::cout << "err?";
    exit(0);
  }

  if (!inputArray) {
    this->printErr("Unable to retrieve input array.");
    return 0;
  }

  // To make sure that the selected array can be processed by this filter,
  // one should also check that the array association and format is correct.
  if (this->GetInputArrayAssociation(0, inputVector) != 0) {
    this->printErr("Input array needs to be a point data array.");
    return 0;
  }
  if (inputArray->GetNumberOfComponents() != 1) {
    this->printErr("Input array needs to be a scalar array.");
    return 0;
  }

  // If all checks pass then log which array is going to be processed.
  this->printMsg("Starting computation...");
  this->printMsg("  Scalar Array: " + std::string(inputArray->GetName()));

  // Create an output array that has the same data type as the input array
  // Note: vtkSmartPointers are well documented
  //       (https://vtk.org/Wiki/VTK/Tutorials/SmartPointers)

  // Get ttk::triangulation of the input vtkDataSet (will create one if one does
  // not exist already).
  ttk::Triangulation* triangulation
    = ttkAlgorithm::GetTriangulation(inputDataSet);
  if (!triangulation)
    return 0;

  // Precondition the triangulation (e.g., enable fetching of vertex neighbors)
  this->preconditionTriangulation(triangulation); // implemented in base class

  // Templatize over the different input array data types and call the base code
  // int status = 0; // this integer checks if the base code returns an error
  // ttkVtkTemplateMacro(inputArray->GetDataType(), triangulation->getType(),
  //                     (status = this->computeAverages<VTK_TT, TTK_TT>(
  //                        (VTK_TT *)ttkUtils::GetVoidPointer(outputArray),
  //                        (VTK_TT *)ttkUtils::GetVoidPointer(inputArray),
  //                        (TTK_TT *)triangulation->getData())));

  // // On error cancel filter execution
  // if(status != 1)
  //   return 0;

  // 将数据转成数组作为参数传递
  int N = inputArray->GetSize();
  if (N != 1ll * dims[0] * dims[1] * dims[2]) {
    std::cout << "err?";
  }

  cout << "N" << N << "\n" << "?";
  // int N = 36;
  float* Values = new float[N];
  for (int i = 0; i < N; i++) Values[i] = inputArray->GetVariantValue(i).ToFloat();



  int* NeighborNums = new int[N];
  int** Neighbors = new int* [N];

  size_t maxneighbor = 0;
  for (int i = 0; i < N; i++) {
    size_t const nNeighbors = triangulation->getVertexNeighborNumber(i);
    NeighborNums[i] = nNeighbors;
    Neighbors[i] = new int[nNeighbors];
    ttk::SimplexId neighborId { -1 };
    maxneighbor = max(maxneighbor, nNeighbors);
    for (size_t j = 0; j < nNeighbors; j++) {
      triangulation->getVertexNeighbor(i, j, neighborId);
      Neighbors[i][j] = neighborId;
    }
  }

  cout << "maxneighbor" << maxneighbor << "\n";


  int* result = compute(N, Values, NeighborNums, Neighbors, dims);

  int* result2 = compute_coords(N, Values, dims);
  // Get output vtkDataSet (which was already instantiated based on the
  // information provided by FillOutputPortInformation)
  vtkDataSet* outputDataSet = vtkDataSet::GetData(outputVector, 0);

  // make a SHALLOW copy of the input
  outputDataSet->ShallowCopy(inputDataSet);

  // add to the output point data the computed output array
  vtkSmartPointer<vtkDataArray> const outputArray
    = vtkSmartPointer<vtkDataArray>::Take(inputArray->NewInstance());
  outputArray->SetName(this->OutputArrayName.data()); // set array name
  outputArray->SetNumberOfComponents(1); // only one component per tuple
  // outputArray->SetNumberOfTuples(inputArray->GetNumberOfTuples());
  int K = N + result2[0] + result2[1] + 2;
  outputArray->SetNumberOfTuples(N + result2[0] + result2[1] + 2);
  for (vtkIdType i = 0; i < K; ++i) {
    // component=0，因为是一维标量
    outputArray->SetComponent(i, 0, static_cast<double>(result2[i]));
  }

  outputDataSet->GetPointData()->AddArray(outputArray);

  // std::ostream &os = std::cout;
  // inputArray->Print(os);
  // outputDataSet->GetPointData()->Print(os);
  // vtkDataArray* output = outputDataSet->GetPointData()->GetArray(0);
  // vtkSmartPointer<vtkDataArray> const indexArray
  //   = vtkSmartPointer<vtkDataArray>::Take(inputArray->NewInstance());
  // indexArray->SetName("IndexArray"); // set array name
  // indexArray->SetNumberOfComponents(1); // only one component per tuple
  // indexArray->SetNumberOfTuples(output->GetNumberOfTuples());
  // for (int i = 0; i < indexArray->GetSize(); i++) indexArray->SetVariantValue(i, i);
  // std::cout << output->GetSize() << std::endl;
  // std::cout << output->GetNumberOfTuples() << std::endl;
  // vtkSortDataArray* sortedArray = vtkSortDataArray::New();
  // sortedArray->Sort(output, indexArray);
  // // std::cout << output->GetNumberOfComponents() << std::endl;
  // // output->SetVariantValue(0, 1);
  // for (int i = 0; i < 10; i++) {
  //   std::cout << indexArray->GetVariantValue(i) << "," << output->GetVariantValue(i) << std::endl;
  // }

  // return success
  return 1;
}
*/

int ttkExtremeGraphDev::RequestData(vtkInformation* ttkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector) {

  // Get input object from input vector
  vtkDataSet* inputDataSet = vtkDataSet::GetData(inputVector[0]);
  if (!inputDataSet)
    return 0;

  // Get input array that will be processed
  vtkDataArray* inputArray = this->GetInputArrayToProcess(0, inputVector);

  if (!inputArray) {
    this->printErr("Unable to retrieve input array.");
    return 0;
  }

  if (this->GetInputArrayAssociation(0, inputVector) != 0) { // 0 for point data
    this->printErr("Input array needs to be a point data array.");
    return 0;
  }
  if (inputArray->GetNumberOfComponents() != 1) {
    this->printErr("Input array needs to be a scalar array.");
    return 0;
  }

  this->printMsg("Starting computation...");
  this->printMsg("  Scalar Array: " + std::string(inputArray->GetName()));

  // 1. 数组本身的尺寸信息
  vtkIdType numTuples = inputArray->GetNumberOfTuples();     // 元组个数 (should be N_points for point data)
  int       numComponents = inputArray->GetNumberOfComponents(); // 每个元组的分量数 (should be 1)
  std::cout
    << "Array Tuples: " << numTuples
    << ", Components: " << numComponents
    << std::endl;

  vtkIdType N_points = inputDataSet->GetNumberOfPoints(); // Number of points in the dataset
  if (numTuples != N_points) {
      this->printErr("Number of tuples in scalar array does not match number of points in dataset.");
      return 0;
  }


  int32_t grid_type = 0; // 0: ImageData, 1: StructuredGrid, 2: Other (Unstructured, PolyData, etc.)
  int dims[3] = {0, 0, 0}; // Initialize dims

  // 2. 数据集（网格）级别的信息
  // vtkDataSet* ds = vtkDataSet::GetData(inputVector[0]); // Already got inputDataSet
  if (auto img = vtkImageData::SafeDownCast(inputDataSet)) {
    img->GetDimensions(dims);
    std::cout
      << "ImageData Dimensions: "
      << dims[0] << " × "
      << dims[1] << " × "
      << dims[2]
      << std::endl;
    grid_type = 0;
    // For ImageData, N_points should be dims[0]*dims[1]*dims[2]
    if (N_points != 1LL * dims[0] * dims[1] * dims[2]) {
        this->printErr("ImageData point count mismatch.");
        // It's possible for an ImageData to have fewer points than its dimensions suggest
        // if it represents a sub-extent, but typically N_points == product of dims.
        // Depending on how 'compute' uses 'dims', this might or might not be an error.
        // For now, we'll proceed but this could be a point of failure for 'compute'.
    }
  }
  else if (auto sg = vtkStructuredGrid::SafeDownCast(inputDataSet)) {
    sg->GetDimensions(dims);
    std::cout
      << "StructuredGrid Dimensions: "
      << dims[0] << " × "
      << dims[1] << " × "
      << dims[2]
      << std::endl;
    grid_type = 1;
    if (N_points != 1LL * dims[0] * dims[1] * dims[2]) {
        this->printErr("StructuredGrid point count mismatch.");
        // Similar to ImageData, this could indicate an issue or a specific configuration.
    }
  }
  else { // vtkUnstructuredGrid, vtkPolyData, etc.
    vtkIdType nCell = inputDataSet->GetNumberOfCells();
    std::cout
      << "Unstructured / Poly Data — Points: " << N_points
      << ", Cells: " << nCell
      << std::endl;
    grid_type = 2;
    // For unstructured grids, dims is less meaningful in the same way.
    // We can set dims[0] = N_points, dims[1]=1, dims[2]=1 if 'compute' expects something.
    // Or, ensure 'compute' handles grid_type=2 differently regarding 'dims'.
    // For now, dims remains {0,0,0} or its last set value, which might be fine if 'compute' ignores it for grid_type 2.
    // Or, more robustly:
    dims[0] = N_points; dims[1] = 1; dims[2] = 1; // Or some other convention for unstructured.
  }

  // The old check was:
  // if (inputArray->GetSize() != 1ll * dims[0] * dims[1] * dims[2]&& grid_type == 0) {
  // This is now covered by N_points check against product of dims for grid_type 0 and 1.
  // inputArray->GetSize() == numTuples * numComponents. Since numComponents is 1, GetSize() == numTuples.

  // 将数据转成数组作为参数传递
  // N_points is the number of points and also number of tuples in our scalar point array
  float* Values = new float[N_points];
  for (vtkIdType i = 0; i < N_points; i++) {
      Values[i] = inputArray->GetTuple1(i); // More direct for single component array
  }

  int* NeighborNums = new int[N_points];
  int** Neighbors = new int*[N_points];
  for(vtkIdType i=0; i<N_points; ++i) Neighbors[i] = nullptr; // Initialize

  size_t maxneighbor = 0;

  if (grid_type == 0 || grid_type == 1) { // Structured grids (ImageData, StructuredGrid)
    this->printMsg("Using TTK triangulation for neighbor finding (structured grid).");
    ttk::Triangulation* triangulation = ttkAlgorithm::GetTriangulation(inputDataSet);
    if (!triangulation) {
      delete[] Values;
      delete[] NeighborNums;
      delete[] Neighbors; // Individual arrays not allocated yet in this path
      return 0;
    }
    this->preconditionTriangulation(triangulation);

    for (vtkIdType i = 0; i < N_points; i++) {
      size_t const nNeighbors = triangulation->getVertexNeighborNumber(i);
      NeighborNums[i] = static_cast<int>(nNeighbors);
      if (nNeighbors > 0) {
        Neighbors[i] = new int[nNeighbors];
        ttk::SimplexId neighborId{-1};
        for (size_t j = 0; j < nNeighbors; j++) {
          triangulation->getVertexNeighbor(i, j, neighborId);
          Neighbors[i][j] = static_cast<int>(neighborId);
        }
      } else {
        Neighbors[i] = nullptr; // Or new int[0] if compute function prefers
      }
      maxneighbor = std::max(maxneighbor, nNeighbors);
    }
  } else { // Unstructured grids (grid_type == 2)
    this->printMsg("Using vtkDataSet topology for neighbor finding (unstructured grid).");
    vtkNew<vtkIdList> cellIds;          // List of cells connected to a point
    vtkNew<vtkIdList> pointIdsInCell;   // List of points in a cell

    for (vtkIdType ptId = 0; ptId < N_points; ++ptId) {
      std::set<vtkIdType> uniqueNeighbors; // Use std::set to handle duplicate neighbors
      inputDataSet->GetPointCells(ptId, cellIds.GetPointer());

      for (vtkIdType i = 0; i < cellIds->GetNumberOfIds(); ++i) {
        vtkIdType currentCellId = cellIds->GetId(i);
        inputDataSet->GetCellPoints(currentCellId, pointIdsInCell.GetPointer());
        // Alternatively:
        // vtkCell* cell = inputDataSet->GetCell(currentCellId);
        // vtkIdList* ptsInCell = cell->GetPointIds();

        for (vtkIdType j = 0; j < pointIdsInCell->GetNumberOfIds(); ++j) {
          vtkIdType neighborPtId = pointIdsInCell->GetId(j);
          if (neighborPtId != ptId) { // Don't add the point itself
            uniqueNeighbors.insert(neighborPtId);
          }
        }
      }

      NeighborNums[ptId] = static_cast<int>(uniqueNeighbors.size());
      if (NeighborNums[ptId] > 0) {
        Neighbors[ptId] = new int[NeighborNums[ptId]];
        int k = 0;
        for (vtkIdType neighbor : uniqueNeighbors) {
          Neighbors[ptId][k++] = static_cast<int>(neighbor);
        }
      } else {
        Neighbors[ptId] = nullptr;
      }
      maxneighbor = std::max(maxneighbor, static_cast<size_t>(NeighborNums[ptId]));
    }
  }

  std::cout << "Max number of neighbors: " << maxneighbor << std::endl;
  int* result=NULL;
  // if (grid_type==2){
  //     printf("with neighbor\n");
  //     result = compute(N_points, Values, NeighborNums, Neighbors, dims);
  // }
  // else {
  //     printf("without neighbor\n");
  //     result = compute_coords(N_points, Values, dims); // Assuming this returns an array
  // }
  result = compute(N_points, Values, NeighborNums, Neighbors, dims);
  //可视化过程
  long long idx = 0;
  std::vector<int> CandidateSaddlePoints,ExtremePoints,FatherLinks_cpy,FinalLinks;
  int CandidateSaddlePointNum,ExtremePointNum,N;
  CandidateSaddlePointNum = result[idx++];
  for(int i = 0; i < CandidateSaddlePointNum; i++){
    int pos = result[idx++];
    CandidateSaddlePoints.push_back(pos);
  }
  ExtremePointNum = result[idx++];
  for(int i = 0; i < ExtremePointNum; i++){
    int pos = result[idx++];
    ExtremePoints.push_back(pos);
  }
  N = static_cast<int>(N_points);
  for(int i = 0; i < N; i++){
    int pos = result[idx++];
    FatherLinks_cpy.push_back(pos);
  }
  for(int i = 0; i < N; i++){
    int pos = result[idx++];
    FinalLinks.push_back(pos);
  }
  //获取标量值
  std::vector<double> scalars;
  for(int i = 0; i < N; i++){
    double pos = Values[i];
    scalars.push_back(pos);
  }
  // std::cout << std::fixed << std::setprecision(8); // 强制显示8位小数
  // for(double x : scalars) std::cout << x << " ";

  // //提取标量值数组  scalars
  // vtkDataArray* dataArray = inputDataSet->GetPointData()->GetArray("ImageFile");
  // if (!dataArray) {
  //     std::cerr << "Scalar array " << "ImageFile" << " not found!\n";
  //     return {};
  // }

  // int numTuples_Lable = dataArray->GetNumberOfTuples();
  // std::vector<double> scalars(numTuples_Lable);
  // for (int i = 0; i < numTuples_Lable; ++i) {
  //     scalars[i] = dataArray->GetComponent(i, 0);
  // }

  //标记鞍点-极值点对拓扑持久度最小的5%和最大的5%的鞍点
  const int numSaddles = CandidateSaddlePoints.size();
  std::vector<SaddlePersistence> persistences;

  // 1. 遍历所有鞍点，计算其与对应极值点之间的标量差
  for (int i = 0; i < numSaddles; ++i) {
      int saddleId = CandidateSaddlePoints[i];
      int extremumId = FinalLinks[saddleId];
      double saddleVal = scalars[saddleId];
      double extremumVal = scalars[extremumId];
      double persistence = std::abs(saddleVal - extremumVal);

      persistences.push_back({saddleId, persistence});
  }

  // 2. 排序持久性值
  std::sort(persistences.begin(), persistences.end(),
            [](const SaddlePersistence& a, const SaddlePersistence& b) {
                return a.persistence < b.persistence;
            });

  // 3. 找出中间90%的范围
  int lower = static_cast<int>(numSaddles * 0.00);
  int upper = static_cast<int>(numSaddles * 1.00);
  double threshold_lower = persistences[lower].persistence;
  double threshold_upper = persistences[upper].persistence;
  threshold_lower = 0.0;
  threshold_upper = 20000.0;
  std::cout << "Threshold: " << threshold_lower << " - " << threshold_upper << std::endl;

  // std::cout << "Lower: " << lower << ", Upper: " << upper << std::endl;
  // std::cout << "numSaddles: " << numSaddles << std::endl;

  std::vector<int> CandidateSaddlePoints_Label(N, 1);
  for (int i = lower; i < upper; ++i) {
      int idx = persistences[i].saddleIdx;
      //cout << "idx: " << idx << " ";
      CandidateSaddlePoints_Label[idx] = 1;
  }


  // 1. 读取原始网格
  std::cout << "Reading input data..." << std::endl;
  vtkPoints* inputPoints = inputDataSet->GetPoints();
  vtkIdType numPoints = inputDataSet->GetNumberOfPoints();

  // 2. 建立极值点集合，便于快速判断
  std::cout << "Building extrema..." << std::endl;
  std::unordered_set<int> extremaSet(ExtremePoints.begin(), ExtremePoints.end());

  //弧束化
  std::unordered_map<std::pair<int, int>, std::vector<int>, pair_hash> extremumPairToSaddles;
  std::unordered_set<int> survivingSaddles;
  int use_edge_bundling = 0;
  //收集 extremum-pair → saddle 映射
  for (int saddle : CandidateSaddlePoints) {
    if (CandidateSaddlePoints_Label[saddle] != 1)
      continue;

    std::unordered_set<int> linkedExtrema;
    for (int ni = 0; ni < NeighborNums[saddle]; ++ni) {
      int neighbor = Neighbors[saddle][ni];
      int finalExt = FinalLinks[neighbor];
      linkedExtrema.insert(finalExt);
    }

    // 根据 extremum 个数决定如何处理
    if (linkedExtrema.size() == 1) {
      continue; // 跳过，只连接一个 extremum，不保留
    }
    else if (linkedExtrema.size() == 2) {
      //  候选弧束化
      auto it = linkedExtrema.begin();
      int e1 = *it;
      int e2 = *(++it);
      if (e1 > e2) std::swap(e1, e2);
      extremumPairToSaddles[{e1, e2}].push_back(saddle);
    }
    else {
      // 连接多个 extremum，直接保留
      survivingSaddles.insert(saddle);
    }
  }
  //找出每个 extremum pair 的代表鞍点
  for (auto& kv : extremumPairToSaddles) {
    const auto& saddles = kv.second;

    int bestSaddle = saddles[0];
    double bestVal = scalars[bestSaddle];

    for (int i = 1; i < saddles.size(); ++i) {
      int s = saddles[i];
      if (scalars[s] > bestVal) {
        bestVal = scalars[s];
        bestSaddle = s;
      }
    }

    survivingSaddles.insert(bestSaddle);
  }
  
  vtkSmartPointer<vtkPoints> outputPoints = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> outputLines = vtkSmartPointer<vtkCellArray>::New();

  std::unordered_map<int, vtkIdType> origToOutputId;
  vtkIdType nextId = 0;
  // int used_ExtremePointNum = 0;
  // std::set<int> used_EPS;
  // int x = 0;

  int count_abs = 0;
  for (int saddle : CandidateSaddlePoints) {
    // if (CandidateSaddlePoints_Label[saddle] != 1 || FinalLinks[saddle] == FatherLinks_cpy[saddle])
    // if (CandidateSaddlePoints_Label[saddle] != 1)
    // continue;
    
    if (use_edge_bundling && survivingSaddles.count(saddle) == 0)
    continue; // 非代表鞍点，跳过

    // Step 1: 将邻居按 FinalLinks 分组，选择每组中 scalar 值最大的那个
    std::unordered_map<int, int> finalGroupToBestNeighbor;
    for (int ni = 0; ni < NeighborNums[saddle]; ++ni) {
      int neighbor = Neighbors[saddle][ni];
      // if (extremaSet.count(neighbor)) // 跳过极值点邻居
      //   continue;

      int finalExtremum = FinalLinks[neighbor];
      if (finalGroupToBestNeighbor.find(finalExtremum) == finalGroupToBestNeighbor.end() ||
          scalars[neighbor] > scalars[finalGroupToBestNeighbor[finalExtremum]]) {
        finalGroupToBestNeighbor[finalExtremum] = neighbor;
      }
    }

    // Step 2: 处理每组中被选中的代表邻居
    for (const auto& pair : finalGroupToBestNeighbor) {
      int neighbor = pair.second;

      
      if(abs(scalars[saddle] - scalars[FinalLinks[neighbor]]) < threshold_lower || abs(scalars[saddle] - scalars[FinalLinks[neighbor]]) >= threshold_upper){
        count_abs++;
      }
      if(abs(scalars[saddle] - scalars[FinalLinks[neighbor]]) >= threshold_lower && abs(scalars[saddle] - scalars[FinalLinks[neighbor]]) < threshold_upper){
        // 添加 saddle 和 neighbor 到点集
        for (int id : {saddle, neighbor}) {
          if (origToOutputId.count(id) == 0) {
            double coord[3]; inputPoints->GetPoint(id, coord);
            origToOutputId[id] = nextId++;
            outputPoints->InsertNextPoint(coord);
          }
        }

        // 添加线段 saddle -> neighbor
        auto line0 = vtkSmartPointer<vtkLine>::New();
        line0->GetPointIds()->SetId(0, origToOutputId[saddle]);
        line0->GetPointIds()->SetId(1, origToOutputId[neighbor]);
        outputLines->InsertNextCell(line0);

        // 沿 FatherLinks_cpy 向上直到极值点
        int curr = neighbor;
        while (true) {
          int parent = FatherLinks_cpy[curr];
          if (parent < 0 || parent >= numPoints || curr == parent)
            break;

          if (origToOutputId.count(parent) == 0) {
            double coord[3]; inputPoints->GetPoint(parent, coord);
            origToOutputId[parent] = nextId++;
            outputPoints->InsertNextPoint(coord);
          }

          auto line = vtkSmartPointer<vtkLine>::New();
          line->GetPointIds()->SetId(0, origToOutputId[curr]);
          line->GetPointIds()->SetId(1, origToOutputId[parent]);
          outputLines->InsertNextCell(line);

          if (extremaSet.count(parent))
            break;

          curr = parent;
        }
      }
    }
      
  }
  std::cout << "过滤数量：" << count_abs << std::endl;
  
  auto polyData = vtkSmartPointer<vtkPolyData>::New();
  polyData->SetPoints(outputPoints);
  polyData->SetLines(outputLines);

  // 5. 添加点数据：NodeType（0 普通，1 鞍点，2 极值点）
  std::cout << "Adding point data..." << std::endl;
  auto nodeType = vtkSmartPointer<vtkIntArray>::New();
  nodeType->SetName("NodeType");
  nodeType->SetNumberOfComponents(1);
  nodeType->SetNumberOfTuples(outputPoints->GetNumberOfPoints());

  // 反向映射：outputId -> 原始点索引
  std::vector<int> reverseMap(outputPoints->GetNumberOfPoints(), -1);
  for (const auto& pair : origToOutputId)
      reverseMap[pair.second] = pair.first;

  std::unordered_set<int> saddleSet(CandidateSaddlePoints.begin(), CandidateSaddlePoints.end());

  for (vtkIdType i = 0; i < outputPoints->GetNumberOfPoints(); ++i)
  {
      int origId = reverseMap[i];
      if (extremaSet.count(origId))
          nodeType->SetValue(i, 2);
      else if (saddleSet.count(origId))
          nodeType->SetValue(i, 1);
      else
          nodeType->SetValue(i, 0);
  }

  polyData->GetPointData()->AddArray(nodeType);

  vtkDataArray* imageFileArray = inputDataSet->GetPointData()->GetArray("ImageFile");
  if (imageFileArray) {
    vtkSmartPointer<vtkDataArray> imageFileCopy = vtkSmartPointer<vtkDataArray>::NewInstance(imageFileArray);
    imageFileCopy->SetName("ImageFile");
    imageFileCopy->SetNumberOfComponents(imageFileArray->GetNumberOfComponents());
    imageFileCopy->SetNumberOfTuples(outputPoints->GetNumberOfPoints());

    for (vtkIdType i = 0; i < outputPoints->GetNumberOfPoints(); ++i)
    {
        int origId = reverseMap[i];
        imageFileCopy->SetTuple(i, origId, imageFileArray);
    }

    polyData->GetPointData()->AddArray(imageFileCopy);
  }
/*
  // 1. 构建 polyData 并检测连通分量
  auto polyData1 = vtkSmartPointer<vtkPolyData>::New();
  polyData1->SetPoints(outputPoints);
  polyData1->SetLines(outputLines);

  auto connectivity = vtkSmartPointer<vtkConnectivityFilter>::New();
  connectivity->SetInputData(polyData1);
  connectivity->SetExtractionModeToAllRegions();
  connectivity->ColorRegionsOn();
  connectivity->Update();

  auto connectedOutput = connectivity->GetOutput();
  auto regionIds = vtkIntArray::SafeDownCast(connectedOutput->GetCellData()->GetArray("RegionId"));

  std::unordered_map<int, int> regionSize;
  vtkIdType numCells = connectedOutput->GetNumberOfCells();
  for (vtkIdType i = 0; i < numCells; ++i) {
      int regionId = regionIds->GetValue(i);
      regionSize[regionId]++;
  }

  // 找到最大的连通分量
  int mainRegion = -1, maxSize = 0;
  for (const auto& kv : regionSize) {
      if (kv.second > maxSize) {
          mainRegion = kv.first;
          maxSize = kv.second;
      }
  }

  vtkPoints* pts = connectedOutput->GetPoints();
  std::unordered_map<int, std::vector<vtkIdType>> regionToCells;
  for (vtkIdType i = 0; i < numCells; ++i) {
      int regionId = regionIds->GetValue(i);
      regionToCells[regionId].push_back(i);
  }

  // 构建点的反向映射（cell → point id）
  std::unordered_map<vtkIdType, std::pair<vtkIdType, vtkIdType>> cellToLineEndpoints;
  for (vtkIdType i = 0; i < numCells; ++i) {
      auto line = vtkLine::SafeDownCast(connectedOutput->GetCell(i));
      if (!line) continue;
      cellToLineEndpoints[i] = { line->GetPointId(0), line->GetPointId(1) };
  }

  // 修复每个非主干分量
  for (const auto& kv : regionToCells) {
      int regionId = kv.first;
      if (regionId == mainRegion) continue;

      vtkIdType connectPid = -1;
      double connectPt[3] = {0, 0, 0};

      vtkIdType seedCell = -1;

      // 找到这个分量中第一个合法的线段
      for (vtkIdType cid : kv.second) {
          auto it = cellToLineEndpoints.find(cid);
          if (it != cellToLineEndpoints.end()) {
              seedCell = cid;
              connectPid = it->second.first;
              break;
          }
      }

      if (connectPid < 0 || connectPid >= pts->GetNumberOfPoints()) {
          std::cerr << "Warning: cannot find a valid connection point in region " << regionId << std::endl;
          continue;  // 无法修复该子图
      }

      pts->GetPoint(connectPid, connectPt);

      // 在主连通分量中找最近点
      double minDist2 = VTK_DOUBLE_MAX;
      vtkIdType nearestId = -1;
      for (vtkIdType i = 0; i < pts->GetNumberOfPoints(); ++i) {
          double p[3];
          pts->GetPoint(i, p);
          // 注意：不是点，而是 regionId 要从 cell → 点映射中拿出来再判断，或直接跳过
          double dist2 = vtkMath::Distance2BetweenPoints(connectPt, p);
          if (dist2 < minDist2) {
              minDist2 = dist2;
              nearestId = i;
          }
      }

      if (nearestId >= 0 && connectPid >= 0 && connectPid < pts->GetNumberOfPoints()) {
          auto line = vtkSmartPointer<vtkLine>::New();
          line->GetPointIds()->SetId(0, connectPid);
          line->GetPointIds()->SetId(1, nearestId);
          outputLines->InsertNextCell(line);
      } else {
          std::cerr << "Warning: unable to connect region " << regionId << std::endl;
      }
  }

  polyData1->SetPoints(outputPoints);
  polyData1->SetLines(outputLines);*/



  std::cout << "Writing output..." << std::endl;
  auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  std::string out = "results/EG_skull_" + std::to_string(threshold_lower) + "-" + std::to_string(threshold_upper) + ".vtp";
  writer->SetFileName(out.c_str());
  writer->SetInputData(polyData);
  writer->Write();
  //此处结束可视化过程


// Call your computation functions
// Ensure `compute` and `compute_coords` are correctly defined/declared
// int* result_compute = compute(N_points, Values, NeighborNums, Neighbors, dims);
// int* result_coords = compute_coords(N_points, Values, dims); // Assuming this returns an array


  // Clean up memory for Neighbors and Values
  delete[] Values;
  for (vtkIdType i = 0; i < N_points; i++) {
    delete[] Neighbors[i];
  }
  delete[] Neighbors;
  delete[] NeighborNums;


  // Get output vtkDataSet
  // vtkDataSet* outputDataSet = vtkDataSet::GetData(outputVector, 0);
  // outputDataSet->ShallowCopy(inputDataSet); // Make a shallow copy

  // // Create output array
  // vtkSmartPointer<vtkDataArray> outputArray =
  //   vtkSmartPointer<vtkDataArray>::Take(inputArray->NewInstance()); // Same type as input
  // outputArray->SetName(this->OutputArrayName.data());
  // outputArray->SetNumberOfComponents(1);

  // Size of output array depends on result_coords
  // result_coords[0] = count for group A, result_coords[1] = count for group B
  // Total tuples = N_points (original) + result_coords[0] + result_coords[1] + 2 (for the counts themselves)
  // This interpretation is based on your original code:
  // int K = N + result2[0] + result2[1] + 2;
  // outputArray->SetNumberOfTuples(N + result2[0] + result2[1] + 2);
  // for (vtkIdType i = 0; i < K; ++i) {
  //    outputArray->SetComponent(i, 0, static_cast<double>(result2[i]));
  // }
  // So, result_coords should already contain all the data needed including original points if required.
  // Let's assume result_coords is structured as [countA, countB, data_pt0, data_pt1, ..., data_ptN-1, data_extraA..., data_extraB...]
  // Or, more likely based on your K: result_coords contains N_points values, and result_coords[0] and result_coords[1] are just counts of additional things.
  // Let's stick to your K calculation for sizing:
  // int numExtraTuplesFromCoords0 = result_coords[0];
  // int numExtraTuplesFromCoords1 = result_coords[1];
  // vtkIdType totalOutputTuples = N_points + numExtraTuplesFromCoords0 + numExtraTuplesFromCoords1 + 2;

  // // Important: The `result_coords` array must be large enough to hold `totalOutputTuples` values
  // // if it's directly used to populate `outputArray`.
  // // The previous loop `for (vtkIdType i = 0; i < K; ++i)` suggests `result_coords` *is* this combined array.
  // outputArray->SetNumberOfTuples(totalOutputTuples);
  // for (vtkIdType i = 0; i < totalOutputTuples; ++i) {
  //   outputArray->SetTuple1(i, static_cast<double>(result_coords[i])); // Assuming result_coords has all data
  // }

  // outputDataSet->GetPointData()->AddArray(outputArray);

  // // Clean up result_coords (if it was dynamically allocated by compute_coords)
  // delete[] result_coords;
  // delete[] result_compute; // If you use it and it's dynamically allocated

  return 1;
}

