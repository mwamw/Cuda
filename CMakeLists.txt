cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(life3d LANGUAGES CXX CUDA)

# 设置 CUDA 编译器标志
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(life3d life3d.cu)
add_executable(life3d_gpu life3d_gpu.cu)
# 设置目标属性
set_target_properties(life3d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 如果需要特定的 CUDA 架构，可以取消下面的注释并修改架构值
# set_target_properties(life3d PROPERTIES CUDA_ARCHITECTURES "60;70;75") 