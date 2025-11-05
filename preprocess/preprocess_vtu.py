import vtk
import numpy as np

def uniquify_scalar_values(values, epsilon=1e-8):
    """
    对浮点标量值做唯一化扰动，返回新的 numpy 数组。
    保证相对大小顺序不变，且所有值不同。
    """
    values = np.asarray(values, dtype=np.float64)
    sorted_indices = np.argsort(values, kind='stable')
    perturbed_values = values.copy()
    for rank, idx in enumerate(sorted_indices):
        perturbed_values[idx] += epsilon * rank
    return perturbed_values

def read_vtu_scalar_array(filename, array_name):
    """
    从 VTU 文件中读取指定的标量数组。
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    mesh = reader.GetOutput()

    arr = mesh.GetPointData().GetArray(array_name)
    if not arr:
        raise RuntimeError(f"字段 '{array_name}' 不存在。")

    data = np.array([arr.GetValue(i) for i in range(arr.GetNumberOfTuples())])
    return mesh, data

def write_vtu_with_array(mesh, array_data, array_name, output_filename):
    """
    将新的数组写回 VTU 并保存。
    """
    new_array = vtk.vtkFloatArray()
    new_array.SetName(array_name)
    new_array.SetNumberOfTuples(len(array_data))
    for i, v in enumerate(array_data):
        new_array.SetValue(i, v)

    mesh.GetPointData().RemoveArray(array_name)
    mesh.GetPointData().AddArray(new_array)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(mesh)
    writer.Write()

def main():
    input_filename = "tangaroa-V1.30.vtu"
    output_filename = "tangaroa-V1.30_SVU.vtu"
    array_name = "V"
    
    mesh, scalar_data = read_vtu_scalar_array(input_filename, array_name)
    unique_data = uniquify_scalar_values(scalar_data)
    write_vtu_with_array(mesh, unique_data, array_name, output_filename)
    print(f"唯一化完成，写入文件：{output_filename}")

if __name__ == "__main__":
    main()
