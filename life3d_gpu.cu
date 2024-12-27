#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <string>

using namespace std;

#define AT(x, y, z) universe[(x) * N * N + (y) * N + z]
#define BLOCK_SIZE 8

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// GPU 核心计算函数
__global__ void life3d_kernel(const char* universe, char* next, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= N || idy >= N || idz >= N) return;
    
    int alive = 0;
    // 计算邻居存活数量
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                
                int nx = (idx + dx + N) % N;
                int ny = (idy + dy + N) % N;
                int nz = (idz + dz + N) % N;
                
                alive += universe[nx * N * N + ny * N + nz];
            }
        }
    }
    
    // 应用生命游戏规则
    int current = universe[idx * N * N + idy * N + idz];
    if (current && (alive < 5 || alive > 7))
        next[idx * N * N + idy * N + idz] = 0;
    else if (!current && alive == 6)
        next[idx * N * N + idy * N + idz] = 1;
    else
        next[idx * N * N + idy * N + idz] = current;
}

// 存活细胞数（在 CPU 上计算）
int population(int N, char *universe)
{
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}

// GPU 版本的生命游戏主函数
void life3d_run(int N, char *universe, int T)
{
    // 分配 GPU 内存
    char *d_universe, *d_next;
    size_t size = N * N * N * sizeof(char);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_universe, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_next, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_universe, universe, size, cudaMemcpyHostToDevice));
    
    // 设置 CUDA 网格和块的大小
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y,
              (N + block.z - 1) / block.z);
    
    // 主循环
    for (int t = 0; t < T; t++) {
        life3d_kernel<<<grid, block>>>(d_universe, d_next, N);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 交换指针
        char* temp = d_universe;
        d_universe = d_next;
        d_next = temp;
    }
    
    // 将结果复制回 CPU
    CHECK_CUDA_ERROR(cudaMemcpy(universe, d_universe, size, cudaMemcpyDeviceToHost));
    
    // 释放 GPU 内存
    CHECK_CUDA_ERROR(cudaFree(d_universe));
    CHECK_CUDA_ERROR(cudaFree(d_next));
}

// 读取输入文件
void read_file(char *input_file, char *buffer)
{
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        cout << "Error: Could not open file " << input_file << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size)) {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();
}

// 写入输出文件
void write_file(char *output_file, char *buffer, int N)
{
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file) {
        cout << "Error: Could not open file " << output_file << std::endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        cout << "usage: ./life3d_gpu N T input output" << endl;
        return 1;
    }
    
    int N = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    char *input_file = argv[3];
    char *output_file = argv[4];
    
    char *universe = (char *)malloc(N * N * N);
    read_file(input_file, universe);
    
    int start_pop = population(N, universe);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    life3d_run(N, universe, T);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int final_pop = population(N, universe);
    
    write_file(output_file, universe, N);
    
    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;
    
    free(universe);
    return 0;
}