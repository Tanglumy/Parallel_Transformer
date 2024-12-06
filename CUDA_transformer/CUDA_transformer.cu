#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <string>
#include <iomanip>
#include <omp.h>       // 引入 OpenMP
#include <mpi.h>       // 引入 MPI
#include <cublas_v2.h> // 引入 cuBLAS
#include <cuda_runtime.h>
#include <random>      // 引入 C++11 随机数生成器

using namespace std;
using namespace std::chrono;

// 错误检查宏
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// 定义一个结构体来存储每个模块的时间
struct Timings {
    double data_loading = 0.0;
    double positional_encoding = 0.0;
    double mha_forward = 0.0;
    double feed_forward = 0.0;
    double layer_norm1 = 0.0;
    double layer_norm2 = 0.0;
    double mpi_comm = 0.0;
    double total = 0.0;
};

// 全局函数：逐元素相加两个矩阵
vector<vector<float>> add_vectors(const vector<vector<float>>& a, const vector<vector<float>>& b) {
    assert(a.size() == b.size());
    assert(a[0].size() == b[0].size());
    size_t rows = a.size();
    size_t cols = a[0].size();
    vector<vector<float>> result(rows, vector<float>(cols, 0.0f));

    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result[i][j] = a[i][j] + b[i][j];
    
    return result;
}

// Helper function: Matrix multiplication with cuBLAS
void matmul_cuda(cublasHandle_t handle,
                const float* d_A, const float* d_B, float* d_C,
                int m, int k, int n,
                float alpha = 1.0f, float beta = 0.0f) {
    // cuBLAS 使用列主序，因此需要注意参数顺序
    // C = alpha * A * B + beta * C
    // A: m x k, B: k x n, C: m x n

    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             d_B, n,
                             d_A, k,
                             &beta,
                             d_C, n));
}

// Helper function: Add bias with OpenMP optimization
void add_bias(vector<vector<float>>& matrix, const vector<float>& bias) {
    assert(matrix[0].size() == bias.size());

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] += bias[j];
        }
    }
}

// Helper function: Apply layer normalization with OpenMP optimization
vector<vector<float>> layer_norm(const vector<vector<float>>& input, const vector<float>& gamma, const vector<float>& beta, float epsilon = 1e-6) {
    size_t seq_len = input.size();
    size_t dim = input[0].size();
    vector<vector<float>> output(seq_len, vector<float>(dim, 0.0f));

    #pragma omp parallel for
    for (size_t i = 0; i < seq_len; ++i) {
        // Compute mean
        float mean = 0.0f;
        for (float val : input[i]) mean += val;
        mean /= dim;

        // Compute variance
        float var = 0.0f;
        for (float val : input[i]) var += (val - mean) * (val - mean);
        var /= dim;

        // Normalize
        for (size_t j = 0; j < dim; ++j) {
            output[i][j] = gamma[j] * ((input[i][j] - mean) / sqrt(var + epsilon)) + beta[j];
        }
    }
    return output;
}

// Positional Encoding
vector<vector<float>> positional_encoding(size_t seq_len, size_t d_model) {
    vector<vector<float>> pos_enc(seq_len, vector<float>(d_model, 0.0f));
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
            if (i % 2 == 0) {
                pos_enc[pos][i] = sin(pos / pow(10000.0f, (float)i / d_model));
            } else {
                pos_enc[pos][i] = cos(pos / pow(10000.0f, (float)(i - 1) / d_model));
            }
        }
    }
    return pos_enc;
}

// Function to generate random matrix with small values using C++11 <random>
vector<vector<float>> random_matrix(size_t rows, size_t cols) {
    vector<vector<float>> mat(rows, vector<float>(cols, 0.0f));
    #pragma omp parallel
    {
        // 每个线程使用独立的随机数生成器
        unsigned int seed = omp_get_thread_num() + 1;
        mt19937 generator(seed);
        uniform_real_distribution<float> distribution(-0.01f, 0.01f);

        #pragma omp for
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                mat[i][j] = distribution(generator);
    }
    return mat;
}

// Function to split heads correctly
vector<vector<float>> split_heads(const vector<vector<float>>& X, size_t num_heads, size_t d_k) {
    size_t seq_len = X.size();
    size_t d_model = X[0].size();
    vector<vector<float>> X_split(seq_len * num_heads, vector<float>(d_k, 0.0f));

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_k; ++j) {
                X_split[h * seq_len + i][j] = X[i][h * d_k + j];
            }
        }
    }
    return X_split;
}

// Function to concatenate heads correctly
vector<vector<float>> concatenate_heads(const vector<vector<float>>& X, size_t num_heads, size_t seq_len, size_t d_k) {
    size_t d_model = num_heads * d_k;
    vector<vector<float>> X_concat(seq_len, vector<float>(d_model, 0.0f));

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_k; ++j) {
                X_concat[i][h * d_k + j] += X[h * seq_len + i][j];
            }
        }
    }
    return X_concat;
}

// Transpose matrix
vector<vector<float>> transpose(const vector<vector<float>>& X) {
    if (X.empty()) return {};
    size_t rows = X.size();
    size_t cols = X[0].size();
    vector<vector<float>> X_T(cols, vector<float>(rows, 0.0f));

    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            X_T[j][i] = X[i][j];
    
    return X_T;
}

// Multi-Head Self-Attention with CUDA and cuBLAS optimization
class MultiHeadAttention {
public:
    size_t d_model;
    size_t num_heads;
    size_t d_k;
    size_t d_v;

    // Weight matrices
    vector<vector<float>> W_Q;
    vector<vector<float>> W_K;
    vector<vector<float>> W_V;
    vector<vector<float>> W_O;

    // GPU device pointers
    float* d_W_Q;
    float* d_W_K;
    float* d_W_V;
    float* d_W_O;

    // 构造函数
    MultiHeadAttention(size_t d_model_, size_t num_heads_, cublasHandle_t handle) 
        : d_model(d_model_), num_heads(num_heads_) {
        assert(d_model % num_heads == 0);
        d_k = d_model / num_heads;
        d_v = d_model / num_heads;

        // 初始化权重
        W_Q = random_matrix(d_model, d_model);
        W_K = random_matrix(d_model, d_model);
        W_V = random_matrix(d_model, d_model);
        W_O = random_matrix(d_model, d_model);

        // 分配设备内存并复制权重
        size_t size = d_model * d_model * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_W_Q, size));
        CUDA_CHECK(cudaMalloc((void**)&d_W_K, size));
        CUDA_CHECK(cudaMalloc((void**)&d_W_V, size));
        CUDA_CHECK(cudaMalloc((void**)&d_W_O, size));

        CUDA_CHECK(cudaMemcpy(d_W_Q, W_Q.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W_K, W_K.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W_V, W_V.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W_O, W_O.data(), size, cudaMemcpyHostToDevice));
    }

    // 析构函数
    ~MultiHeadAttention() {
        CUDA_CHECK(cudaFree(d_W_Q));
        CUDA_CHECK(cudaFree(d_W_K));
        CUDA_CHECK(cudaFree(d_W_V));
        CUDA_CHECK(cudaFree(d_W_O));
    }

    // Forward pass with CUDA and cuBLAS
    vector<vector<float>> forward_cuda(const vector<vector<float>>& input, Timings& timings, cublasHandle_t handle) {
        size_t seq_len = input.size();

        // 将输入复制到设备
        float* d_input;
        size_t input_size = seq_len * d_model * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_input, input_size));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice));

        // 设备端输出指针
        float* d_Q, *d_K, *d_V;
        size_t output_size = seq_len * d_model * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_Q, output_size));
        CUDA_CHECK(cudaMalloc((void**)&d_K, output_size));
        CUDA_CHECK(cudaMalloc((void**)&d_V, output_size));

        // 执行 Q = input * W_Q
        auto proj_start = high_resolution_clock::now();
        matmul_cuda(handle, d_input, d_W_Q, d_Q, seq_len, d_model, d_model);
        // 执行 K = input * W_K
        matmul_cuda(handle, d_input, d_W_K, d_K, seq_len, d_model, d_model);
        // 执行 V = input * W_V
        matmul_cuda(handle, d_input, d_W_V, d_V, seq_len, d_model, d_model);
        auto proj_end = high_resolution_clock::now();
        double proj_time = duration_cast<microseconds>(proj_end - proj_start).count() / 1000.0; // ms
        timings.mha_forward += proj_time;

        // 复制结果回主机
        vector<float> h_Q(seq_len * d_model);
        vector<float> h_K(seq_len * d_model);
        vector<float> h_V(seq_len * d_model);
        CUDA_CHECK(cudaMemcpy(h_Q.data(), d_Q, output_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_K.data(), d_K, output_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, output_size, cudaMemcpyDeviceToHost));

        // 清理设备内存
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));

        // 转换为二维向量
        vector<vector<float>> Q(seq_len, vector<float>(d_model, 0.0f));
        vector<vector<float>> K(seq_len, vector<float>(d_model, 0.0f));
        vector<vector<float>> V(seq_len, vector<float>(d_model, 0.0f));

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_model; ++j) {
                Q[i][j] = h_Q[i * d_model + j];
                K[i][j] = h_K[i * d_model + j];
                V[i][j] = h_V[i * d_model + j];
            }
        }

        // 计算注意力权重并生成输出（简化示例）
        // 实际实现应包括 Scaled Dot-Product Attention、Softmax、加权求和等步骤
        // 为简化示例，此处仅返回 Q
        return Q;
    }
};

// Feed Forward Network
class FeedForward {
public:
    size_t d_model;
    size_t d_ff;

    vector<vector<float>> W1;
    vector<float> b1;
    vector<vector<float>> W2;
    vector<float> b2;

    // 构造函数
    FeedForward(size_t d_model_, size_t d_ff_) : d_model(d_model_), d_ff(d_ff_) {
        // 初始化权重
        W1 = random_matrix(d_model, d_ff);
        b1 = vector<float>(d_ff, 0.0f);
        W2 = random_matrix(d_ff, d_model);
        b2 = vector<float>(d_model, 0.0f);
    }

    // Forward pass with OpenMP optimization
    vector<vector<float>> forward(const vector<vector<float>>& input, Timings& timings) {
        // Linear layer 1
        auto linear1_start = high_resolution_clock::now();
        vector<vector<float>> hidden = matmul(input, W1); // (seq_len, d_ff)
        add_bias(hidden, b1);
        auto linear1_end = high_resolution_clock::now();
        double linear1_time = duration_cast<microseconds>(linear1_end - linear1_start).count() / 1000.0; // ms
        timings.feed_forward += linear1_time;

        // ReLU activation with OpenMP optimization
        auto relu_start = high_resolution_clock::now();
        size_t seq_len = hidden.size();
        size_t d_ff_local = hidden[0].size();

        #pragma omp parallel for
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_ff_local; ++j) {
                hidden[i][j] = max(0.0f, hidden[i][j]);
            }
        }
        auto relu_end = high_resolution_clock::now();
        double relu_time = duration_cast<microseconds>(relu_end - relu_start).count() / 1000.0; // ms
        timings.feed_forward += relu_time;

        // Linear layer 2
        auto linear2_start = high_resolution_clock::now();
        vector<vector<float>> output = matmul(hidden, W2); // (seq_len, d_model)
        add_bias(output, b2);
        auto linear2_end = high_resolution_clock::now();
        double linear2_time = duration_cast<microseconds>(linear2_end - linear2_start).count() / 1000.0; // ms
        timings.feed_forward += linear2_time;

        return output;
    }
};

// Transformer Encoder Layer
class EncoderLayer {
public:
    size_t d_model;
    size_t num_heads;
    size_t d_ff;

    MultiHeadAttention mha;
    FeedForward ff;
    vector<float> gamma;
    vector<float> beta;

    EncoderLayer(size_t d_model_, size_t num_heads_, size_t d_ff_, cublasHandle_t handle) :
        d_model(d_model_), num_heads(num_heads_), d_ff(d_ff_),
        mha(d_model_, num_heads_, handle), ff(d_model_, d_ff_) 
    {
        // Initialize gamma and beta for layer normalization
        gamma = vector<float>(d_model, 1.0f);
        beta = vector<float>(d_model, 0.0f);
    }

    // Forward pass with CUDA and OpenMP optimization
    vector<vector<float>> forward(const vector<vector<float>>& input, Timings& timings, int mpi_rank, int mpi_size, cublasHandle_t handle) {
        // Multi-Head Attention
        auto mha_start = high_resolution_clock::now();
        vector<vector<float>> mha_out = mha.forward_cuda(input, timings, handle);
        auto mha_end = high_resolution_clock::now();
        double mha_time = duration_cast<microseconds>(mha_end - mha_start).count() / 1000.0; // ms
        timings.mha_forward += mha_time;

        // Residual connection and layer normalization
        auto res_norm1_start = high_resolution_clock::now();
        vector<vector<float>> add1 = add_vectors(input, mha_out);
        vector<vector<float>> norm1 = layer_norm(add1, gamma, beta);
        auto res_norm1_end = high_resolution_clock::now();
        double res_norm1_time = duration_cast<microseconds>(res_norm1_end - res_norm1_start).count() / 1000.0; // ms
        timings.layer_norm1 += res_norm1_time;

        // Feed Forward
        auto ff_start = high_resolution_clock::now();
        vector<vector<float>> ff_out = ff.forward(norm1, timings);
        auto ff_end = high_resolution_clock::now();
        double ff_time = duration_cast<microseconds>(ff_end - ff_start).count() / 1000.0; // ms
        timings.feed_forward += ff_time;

        // Residual connection and layer normalization
        auto res_norm2_start = high_resolution_clock::now();
        vector<vector<float>> add2 = add_vectors(norm1, ff_out);
        vector<vector<float>> norm2 = layer_norm(add2, gamma, beta);
        auto res_norm2_end = high_resolution_clock::now();
        double res_norm2_time = duration_cast<microseconds>(res_norm2_end - res_norm2_start).count() / 1000.0; // ms
        timings.layer_norm2 += res_norm2_time;

        return norm2;
    }
};

// Transformer Encoder consisting of multiple EncoderLayers
class TransformerEncoder {
public:
    size_t num_layers;
    vector<EncoderLayer> layers;

    TransformerEncoder(size_t d_model, size_t num_heads, size_t d_ff, size_t num_layers_, cublasHandle_t handle) : num_layers(num_layers_) {
        for (size_t i = 0; i < num_layers_; ++i) {
            layers.emplace_back(EncoderLayer(d_model, num_heads, d_ff, handle));
        }
    }

    vector<vector<float>> forward(const vector<vector<float>>& input, Timings& timings, int mpi_rank, int mpi_size, cublasHandle_t handle) {
        vector<vector<float>> output = input;
        for (size_t i = 0; i < num_layers; ++i) {
            output = layers[i].forward(output, timings, mpi_rank, mpi_size, handle);
        }
        return output;
    }
};

// Function to load dataset from file with OpenMP timing
// Each line in the file should contain (seq_len * embed_dim) float numbers separated by spaces
vector<vector<float>> load_dataset(const string& filename, size_t sequence_length, size_t embedding_dim, Timings& timings, int mpi_rank) {
    auto start = high_resolution_clock::now();

    ifstream file(filename);
    if (!file.is_open()) {
        if (mpi_rank == 0)
            cerr << "Failed to open file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<vector<float>> dataset;
    string line;
    size_t expected_size = sequence_length * embedding_dim;

    while (getline(file, line)) {
        istringstream stream(line);
        vector<float> sample;
        float value;
        while (stream >> value) {
            sample.push_back(value);
        }
        if (sample.size() != expected_size) {
            if (mpi_rank == 0)
                cerr << "Sample size mismatch. Expected " << expected_size
                     << ", got " << sample.size() << ". Skipping sample." << endl;
            continue; // Skip malformed samples
        }
        dataset.push_back(sample);
    }
    file.close();

    auto end = high_resolution_clock::now();
    timings.data_loading += duration_cast<microseconds>(end - start).count() / 1000.0; // ms
    return dataset;
}

int main(int argc, char** argv) {
    // 初始化 MPI
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // 获取可用 GPU 数量
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count < mpi_size) {
        if (mpi_rank == 0)
            cerr << "Not enough GPUs for the number of MPI processes." << endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // 为每个 MPI 进程分配不同的 GPU
    int device_id = mpi_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    // 初始化 cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 参数设置
    string filename = "dataset_vectors.txt";
    size_t sequence_length = 10; // 与 Python 预处理一致
    size_t embedding_dim = 50;
    size_t d_model = 50; // Must match embedding_dim
    size_t num_heads = 5; // d_model should be divisible by num_heads
    size_t d_ff = 128; // Feed forward hidden size
    size_t num_layers = 2; // 多层编码器
    size_t batch_size = 1; // 批量大小

    // 检查头数是否能被进程数整除
    if (num_heads < static_cast<size_t>(mpi_size)) {
        if (mpi_rank == 0)
            cerr << "Number of MPI processes (" << mpi_size << ") exceeds number of heads (" << num_heads << ")." << endl;
        CUBLAS_CHECK(cublasDestroy(handle));
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // 初始化 Timings 结构体
    Timings timings;

    // 总时间起点
    auto total_start = high_resolution_clock::now();

    // 加载数据集
    if (mpi_rank == 0) {
        cout << "Loading dataset..." << endl;
    }
    vector<vector<float>> dataset_flat = load_dataset(filename, sequence_length, embedding_dim, timings, mpi_rank);
    if (dataset_flat.empty()) {
        if (mpi_rank == 0)
            cerr << "No data loaded. Exiting." << endl;
        CUBLAS_CHECK(cublasDestroy(handle));
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (mpi_rank == 0) {
        cout << "Loaded " << dataset_flat.size() << " samples." << endl;
    }

    // 初始化 Transformer Encoder
    TransformerEncoder encoder(d_model, num_heads, d_ff, num_layers, handle);

    // 处理批量样本
    for (size_t b = 0; b < batch_size; ++b) {
        // 选择第一个样本进行处理（所有进程使用相同的样本）
        vector<float> flat_sample = dataset_flat[0];
        // 重塑为 (seq_len, embed_dim)
        vector<vector<float>> sample(sequence_length, vector<float>(embedding_dim, 0.0f));
        for (size_t i = 0; i < sequence_length; ++i)
            for (size_t j = 0; j < embedding_dim; ++j)
                sample[i][j] = flat_sample[i * embedding_dim + j];

        // 添加位置编码
        auto pos_enc_start = high_resolution_clock::now();
        vector<vector<float>> pos_enc = positional_encoding(sequence_length, d_model);
        // 并行化位置编码的添加
        #pragma omp parallel for
        for (size_t i = 0; i < sequence_length; ++i) {
            for (size_t j = 0; j < d_model; ++j) {
                sample[i][j] += pos_enc[i][j];
            }
        }
        auto pos_enc_end = high_resolution_clock::now();
        timings.positional_encoding += duration_cast<microseconds>(pos_enc_end - pos_enc_start).count() / 1000.0; // ms

        // 前向传播
        auto encoder_start_time = high_resolution_clock::now();
        vector<vector<float>> encoder_output = encoder.forward(sample, timings, mpi_rank, mpi_size, handle);
        auto encoder_end_time = high_resolution_clock::now();
        double encoder_time = duration_cast<microseconds>(encoder_end_time - encoder_start_time).count() / 1000.0; // ms
        timings.total += encoder_time;
    }

    // 总时间结束
    auto total_end = high_resolution_clock::now();
    double total_time = duration_cast<microseconds>(total_end - total_start).count() / 1000.0; // ms
    timings.total = total_time;

    // 汇总所有时间测量结果
    double data_loading_total, positional_encoding_total, mha_forward_total, feed_forward_total, layer_norm1_total, layer_norm2_total, mpi_comm_total, global_total;
    MPI_Reduce(&timings.data_loading, &data_loading_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.positional_encoding, &positional_encoding_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.mha_forward, &mha_forward_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.feed_forward, &feed_forward_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.layer_norm1, &layer_norm1_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.layer_norm2, &layer_norm2_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.mpi_comm, &mpi_comm_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.total, &global_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // 主进程打印并保存时间测量结果
    if (mpi_rank == 0) {
        cout << fixed << setprecision(3);
        cout << "Execution Timings (in ms):" << endl;
        cout << "Data Loading        : " << data_loading_total << " ms" << endl;
        cout << "Positional Encoding : " << positional_encoding_total << " ms" << endl;
        cout << "Multi-Head Attention: " << mha_forward_total << " ms" << endl;
        cout << "Feed Forward        : " << feed_forward_total << " ms" << endl;
        cout << "Layer Norm 1        : " << layer_norm1_total << " ms" << endl;
        cout << "Layer Norm 2        : " << layer_norm2_total << " ms" << endl;
        cout << "MPI Communication   : " << mpi_comm_total << " ms" << endl;
        cout << "Total Execution     : " << global_total << " ms" << endl;

        // 保存时间测量结果到 CSV 文件
        ofstream timing_file("timings.csv");
        if (!timing_file.is_open()) {
            cerr << "Failed to open timings.csv for writing." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 写入表头
        timing_file << "Module,Time_ms\n";

        // 写入各模块时间
        timing_file << "Data Loading," << data_loading_total << "\n";
        timing_file << "Positional Encoding," << positional_encoding_total << "\n";
        timing_file << "Multi-Head Attention," << mha_forward_total << "\n";
        timing_file << "Feed Forward," << feed_forward_total << "\n";
        timing_file << "Layer Norm 1," << layer_norm1_total << "\n";
        timing_file << "Layer Norm 2," << layer_norm2_total << "\n";
        timing_file << "MPI Communication," << mpi_comm_total << "\n";
        timing_file << "Total Execution," << global_total << "\n";

        timing_file.close();
        cout << "Timings saved to timings.csv" << endl;
    }

    // 清理 cuBLAS 句柄
    CUBLAS_CHECK(cublasDestroy(handle));

    // 结束 MPI
    MPI_Finalize();
    return 0;
}