import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from collections import defaultdict

# 参数设置
dims = (10, 10, 10)
spacing = (1.0, 1.0, 1.0)

# === 1. 预设极值图 ===
maxima_points = [(1, 1, 1), (5, 5, 2), (7, 7, 8)]            # 极值点（最大值）
paths = [                                        # 梯度路径：鞍点 -> 极值点
    [(2, 2, 2), (2, 1, 2), (2, 1, 1), (1, 1, 1)],
    [(2, 2, 2), (3, 3, 2), (4, 4, 2), (5, 5, 2)],
    [(7, 6, 6), (6, 6, 6), (5, 5, 6), (5, 5, 5), (5, 5, 4), (5, 5, 4), (5, 5, 3), (5, 5, 2)], 
    [(7, 6, 6), (7, 6, 7), (7, 6, 8), (7, 7, 8)], 
]

# 转 index 用
def to_index(x, y, z, dims):
    return z * dims[0] * dims[1] + y * dims[0] + x

# === 2. 提取有效鞍点（连接至少两个不同极值点） ===
saddle_to_maxima = defaultdict(set)
for path in paths:
    if len(path) < 2:
        continue
    saddle = path[0]
    maximum = path[-1]
    saddle_to_maxima[tuple(saddle)].add(tuple(maximum))

valid_saddles = [s for s, maxset in saddle_to_maxima.items() if len(maxset) >= 2]
valid_saddles_set = set(valid_saddles)

# 过滤路径，仅保留合法鞍点路径
filtered_paths = []
for path in paths:
    if tuple(path[0]) in valid_saddles_set:
        filtered_paths.append(path)

saddle_points = valid_saddles
print(f"合法鞍点数: {len(saddle_points)}")
print(f"保留路径数: {len(filtered_paths)}")

# === 3. 初始化标量场 ===
scalar_field = np.zeros(dims)

# 设定极值点标量值高
for pt in maxima_points:
    scalar_field[pt] = 100

# 设定合法鞍点标量值为中间值
for pt in saddle_points:
    scalar_field[pt] = 50

# 沿每条路径线性插值
for path in filtered_paths:
    start_val = scalar_field[tuple(path[-1])]
    end_val = scalar_field[tuple(path[0])]
    n = len(path)
    for i, pt in enumerate(reversed(path)):
        scalar_field[pt] = start_val + (end_val - start_val) * (i / (n - 1))

# === 4. 唯一化标量值（防止重复） ===
def uniquify_scalar_field(field, epsilon=1e-5):
    flat = field.ravel()
    sorted_idx = np.argsort(flat, kind="stable")
    perturb = flat.copy()
    for rank, idx in enumerate(sorted_idx):
        perturb[idx] += epsilon * rank
    return perturb.reshape(field.shape)

scalar_field = uniquify_scalar_field(scalar_field)

# === 5. 保存为 VTI ===
image = vtk.vtkImageData()
image.SetDimensions(*dims)
image.SetSpacing(*spacing)

flat_scalars = scalar_field.ravel(order='C')
vtk_array = numpy_to_vtk(flat_scalars, deep=True)
vtk_array.SetName("ScalarField")
image.GetPointData().SetScalars(vtk_array)

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("scalar_field_v1.vti")
writer.SetInputData(image)
writer.Write()

# === 6. 保存极值图为 VTP ===
points = vtk.vtkPoints()
point_types = vtk.vtkIntArray()
point_types.SetName("NodeType")  # 0: regular, 1: saddle, 2: max
point_id_map = {}
next_id = 0

def add_point(pt, label):
    global next_id
    key = tuple(pt)
    if key in point_id_map:
        return point_id_map[key]
    pid = next_id
    next_id += 1
    point_id_map[key] = pid
    points.InsertNextPoint(*pt)
    point_types.InsertNextValue(label)
    return pid

# 添加极值点
for pt in maxima_points:
    add_point(pt, 2)
# 添加鞍点
for pt in saddle_points:
    add_point(pt, 1)
# 添加路径中的点
for path in filtered_paths:
    for pt in path:
        add_point(pt, 0)

# 添加连线
lines = vtk.vtkCellArray()
for path in filtered_paths:
    ids = [add_point(pt, 0) for pt in path]
    for i in range(len(ids) - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, ids[i])
        line.GetPointIds().SetId(1, ids[i + 1])
        lines.InsertNextCell(line)

polyData = vtk.vtkPolyData()
polyData.SetPoints(points)
polyData.SetLines(lines)
polyData.GetPointData().AddArray(point_types)

vtp_writer = vtk.vtkXMLPolyDataWriter()
vtp_writer.SetFileName("extremum_graph_v2.vtp")
vtp_writer.SetInputData(polyData)
vtp_writer.Write()
