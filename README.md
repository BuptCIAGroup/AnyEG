# AnyEG: Fast Extremum Graph Computation for Large-scale Arbitrary Grids
## Introduction

**AnyEG** (*Any Extremum Graph*) is a high-performance algorithm for extremum graph construction.  
It automatically identifies topological extrema (maxima, minima, and saddle points) on arbitrary mesh structures  
and generates their corresponding connectivity graph.

This project is built upon **TTK** (Topology ToolKit) and integrates **VTK 8.2.0**  
for data I/O and visualization support.  
It enables efficient GPU-based topological analysis and features the following advantages:

- **High Performance**: GPU-parallelized design achieving 3–6× speedup over traditional CPU implementations  
- **High Generality**: Supports arbitrary-dimensional and unstructured mesh data  
- **Topological Correctness**: Strictly preserves the topology between extrema and saddle points  

---

## Dataset Preparation

Please download the required test dataset from Kaggle:

[AnyEG Dataset (Kaggle)](https://www.kaggle.com/datasets/wh1stle/anyeg-data/data)

After downloading, extract the data into the `data/` folder under the project directory:
```
AnyEG/
 ├── data/
 │    ├── sample1.vti
 │    ├── sample2.vti
 │    └── ...
 └── CMakeLists.txt
```
---

## Dependencies

Dependencies (minimum required versions):

- CUDA ≥ 10.2  
- C++14  
- CMake ≥ 3.12  
- VTK ≥ 8.2.0  
- TTK ≥ 2.2.0  

---

## Build and Run

### 1. Clone the Repository

```bash
git clone https://github.com/BuptCIAGroup/AnyEG.git
cd AnyEG
```

### 2. Create a Build Directory and Compile

```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

After compilation, the executable file will be located in:

```
build/bin/ttkExtremeGraphDevCmd
```

### 3.Run Example

Run the program and specify the input data:

```
./build/bin/ttkExtremeGraphDevCmd -i ./AnyEG/data/sample1.vti
```

The program will automatically perform the following steps:

1. Load the input VTK data  
2. Perform GPU-accelerated critical point detection and gradient path identification  
3. Construct the connectivity graph between extrema  
4. Output the results to the `results/` directory  

You can visualize the generated `.vtp` / `.vtu` files using **ParaView** or **VTK Viewer**.

### Note: Persistence-Based Extrema–Saddle Filtering

To adjust the persistence threshold and control the retained range of extrema–saddle pairs,  edit lines **637–638** in `ttkExtremeGraphDev.cpp`.  To change the output filename, modify **line 947**, then recompile the project.

Below are recommended persistence threshold ranges for several example datasets:

- `silicium_98x34x34_uint8.vti`: 0–200  
- `foot_256x256x256_uint8.vti`: 110–200  
- `lobster_301x324x56_uint8.vti`: 50–100  
- `hydrogen_atom_128x128x128_uint8.vti`: 79–500  
- `tectonicPuzzle.vtp`: 0–1 (Scalar Field: `T`)  

You can freely adjust these thresholds based on visualization results to achieve the most suitable level of noise filtering.

## Topological Correctness Verification Example

### 1. Create Custom Small-Scale Test Data

```bash
cd data
```

Modify lines **7–17** to customize the grid dimensions, extrema points, and gradient paths,
 and update the dataset name on **line 136**.

```bash
python testdata_create_withsaddlecheck.py
```

### 2.Run the Verification Data

Run the program and specify the generated input data:

```
./build/bin/ttkExtremeGraphDevCmd -i ./AnyEG ../data/test_data.vti
```

The program will automatically perform the following steps:

1. Load the input VTK data
2. Perform GPU-accelerated critical point detection and gradient path identification
3. Construct the connectivity graph between extrema
4. Output the results to the `results/` directory

You can visualize the generated `.vtp` / `.vtu` files using **ParaView** or **VTK Viewer**.
