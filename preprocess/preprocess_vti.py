import vtk
import numpy as np
from collections import defaultdict

def uniquify_scalar_values(values, epsilon=1e-8):
    
    #对浮点标量值做唯一化扰动，返回新的 numpy 数组。
    #保证相对大小顺序不变，且所有值不同。
    values = np.asarray(values, dtype=np.float64)
    sorted_indices = np.argsort(values, kind='stable')  # 保证顺序稳定
    perturbed_values = values.copy()

    for rank, idx in enumerate(sorted_indices):
        perturbed_values[idx] += epsilon * rank  # 逐步加扰动

    return perturbed_values
"""
def uniquify_scalar_values(values, epsilon=1e-8, verbose=True):
    #对浮点标量值做唯一化扰动，返回新的 numpy 数组。
    #保证相对大小顺序不变，且所有值不同。
    values = np.asarray(values, dtype=np.float64)
    sorted_indices = np.argsort(values, kind='stable')  # 保证顺序稳定
    perturbed_values = values.copy()

    for rank, idx in enumerate(sorted_indices):
        old_val = values[idx]
        delta = epsilon * rank
        new_val = old_val + delta
        perturbed_values[idx] = new_val

        if verbose and delta > 0:
            print(f"值 {old_val:.8f} 被扰动为 {new_val:.8f}，原索引: {idx}，扰动等级: {rank}")

    return perturbed_values
"""
def read_vti_scalar_array(filename, array_name):
    """
    从 VTI 文件中读取指定的标量数组。
    """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    image = reader.GetOutput()

    arr = image.GetPointData().GetArray(array_name)
    if not arr:
        raise RuntimeError(f"字段 '{array_name}' 不存在。")

    data = np.array([arr.GetValue(i) for i in range(arr.GetNumberOfTuples())])
    return image, data

def write_vti_with_array(image, array_data, array_name, output_filename):
    """
    将新的数组写回 VTI 并保存。
    """
    new_array = vtk.vtkFloatArray()
    new_array.SetName(array_name)
    new_array.SetNumberOfTuples(len(array_data))
    for i, v in enumerate(array_data):
        new_array.SetValue(i, v)

    image.GetPointData().RemoveArray(array_name)
    image.GetPointData().AddArray(new_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(image)
    writer.Write()

def main():
    input_filename = "neghip.vti"
    output_filename = "neghip_SVU.vti"
    array_name = "ImageFile"
    #根据实际情况修改
    image, scalar_data = read_vti_scalar_array(input_filename, array_name)
    unique_data = uniquify_scalar_values(scalar_data)
    write_vti_with_array(image, unique_data, array_name, output_filename)
    print(f"唯一化完成，写入文件：{output_filename}")

if __name__ == "__main__":
    main()