ttk_add_vtk_module()

project(ttkExtremeGraphDev LANGUAGES CXX CUDA)

# set(SOURCES
#   ttkExtremeGraphDev.cpp
#   # 其他源文件可在这里补充
# )

# 先声明 TTK VTK 模块
# ttk_add_vtk_module(
#   NAME ttkExtremeGraphDev
#   SOURCES ${SOURCES}
#   DEPENDS
#     VTK::CommonCore
#     VTK::CommonDataModel
#     VTK::IOCore
#     VTK::IOXML
#     VTK::IOGeometry
# )

# # 包含 CUDA 头文件路径
# include_directories(/home/lqm/ttk/core/vtk/ttkContourTreeDev/cuda-samples-master/Common/)

# # 添加共享库
# add_library(cudaMatMul SHARED
#     /home/lqm/ttk/core/vtk/ttkContourTreeDev/cudaHelper.cu
#     /home/lqm/ttk/core/vtk/ttkContourTreeDev/cudaHelper.h
# )

# # 指定 CUDA 相关设置
# set_property(TARGET cudaMatMul PROPERTY CUDA_ARCHITECTURES 61)
# set_target_properties(cudaMatMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# # 指定链接的 CUDA 库
# target_link_libraries(cudaMatMul PRIVATE cudart)

# 添加链接
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    cudaExtremeGraphDev
    VTK::IOXML
)