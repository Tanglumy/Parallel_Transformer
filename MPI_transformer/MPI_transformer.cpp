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
#include</opt/homebrew/Cellar/libomp/19.1.5/include/omp.h>
#include <mpi.h>   // 引入 MPI
#include <random>  // 引入 C++11 随机数生成器

using namespace std;
using namespace std::chrono;

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

// Helper function: Matrix multiplication with OpenMP optimization
vector<vector<float>> matmul(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t common = B.size();
    vector<vector<float>> result(rows, vector<float>(cols, 0.0f));

    // 并行化外层循环
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < common; ++k) {
            float a_ik = A[i][k];
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] += a_ik * B[k][j];
            }
        }
    }
    return result;
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

// Helper function: Apply softmax to rows of a matrix with OpenMP optimization
void softmax_rows(vector<vector<float>>& matrix) {
    size_t rows = matrix.size();

    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        float max_val = *max_element(matrix[i].begin(), matrix[i].end());
        float sum_exp = 0.0f;
        for (auto& val : matrix[i]) {
            val = exp(val - max_val);
            sum_exp += val;
        }
        for (auto& val : matrix[i]) {
            val /= sum_exp;
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
        unsigned int seed = omp_get_thread_num();
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

// Multi-Head Self-Attention with MPI and OpenMP optimization
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

    MultiHeadAttention(size_t d_model_, size_t num_heads_) : d_model(d_model_), num_heads(num_heads_) {
        assert(d_model % num_heads == 0);
        d_k = d_model / num_heads;
        d_v = d_model / num_heads;

        // Initialize weights with random values for simplicity
        // In practice, weights should be learned parameters
        W_Q = random_matrix(d_model, d_model);
        W_K = random_matrix(d_model, d_model);
        W_V = random_matrix(d_model, d_model);
        W_O = random_matrix(d_model, d_model);
    }

    // Forward pass with MPI and OpenMP timing
    vector<vector<float>> forward(const vector<vector<float>>& input, Timings& timings, int mpi_rank, int mpi_size) {
        size_t seq_len = input.size();

        // Linear projections
        auto proj_start = high_resolution_clock::now();
        vector<vector<float>> Q = matmul(input, W_Q); // (seq_len, d_model)
        vector<vector<float>> K = matmul(input, W_K); // (seq_len, d_model)
        vector<vector<float>> V = matmul(input, W_V); // (seq_len, d_model)
        auto proj_end = high_resolution_clock::now();
        double proj_time = duration_cast<microseconds>(proj_end - proj_start).count() / 1000.0; // ms
        timings.mha_forward += proj_time;

        // Split into heads
        auto split_start = high_resolution_clock::now();
        vector<vector<float>> Q_heads = split_heads(Q, num_heads, d_k); // (num_heads * seq_len, d_k)
        vector<vector<float>> K_heads = split_heads(K, num_heads, d_k); // (num_heads * seq_len, d_k)
        vector<vector<float>> V_heads = split_heads(V, num_heads, d_v); // (num_heads * seq_len, d_v)
        auto split_end = high_resolution_clock::now();
        double split_time = duration_cast<microseconds>(split_end - split_start).count() / 1000.0; // ms
        timings.mha_forward += split_time;

        // Determine heads per MPI process
        size_t heads_per_proc = num_heads / mpi_size;
        size_t remainder = num_heads % mpi_size;
        size_t local_num_heads = (mpi_rank < remainder) ? (heads_per_proc + 1) : heads_per_proc;
        size_t start_head = mpi_rank * heads_per_proc + min(static_cast<size_t>(mpi_rank), remainder);
        size_t end_head = start_head + local_num_heads;

        // Prepare local heads
        vector<vector<float>> Q_local(local_num_heads * seq_len, vector<float>(d_k, 0.0f));
        vector<vector<float>> K_local(local_num_heads * seq_len, vector<float>(d_k, 0.0f));
        vector<vector<float>> V_local(local_num_heads * seq_len, vector<float>(d_v, 0.0f));

        for (size_t h = 0; h < local_num_heads; ++h) {
            size_t global_head = start_head + h;
            for (size_t i = 0; i < seq_len; ++i) {
                Q_local[h * seq_len + i] = Q_heads[global_head * seq_len + i];
                K_local[h * seq_len + i] = K_heads[global_head * seq_len + i];
                V_local[h * seq_len + i] = V_heads[global_head * seq_len + i];
            }
        }

        // Each process computes its local attention
        auto attention_start = high_resolution_clock::now();
        vector<vector<float>> attention_local(local_num_heads * seq_len, vector<float>(d_v, 0.0f));

        #pragma omp parallel for
        for (size_t h = 0; h < local_num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                // Compute scores = Q * K^T
                vector<float> scores(seq_len, 0.0f);
                for (size_t j = 0; j < seq_len; ++j) {
                    for (size_t dk = 0; dk < d_k; ++dk) {
                        scores[j] += Q_local[h * seq_len + i][dk] * K_local[h * seq_len + j][dk];
                    }
                }

                // Scale scores
                float scale = sqrt(static_cast<float>(d_k));
                for (size_t j = 0; j < seq_len; ++j) {
                    scores[j] /= scale;
                }

                // Apply softmax
                float max_val = *max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (auto& val : scores) {
                    val = exp(val - max_val);
                    sum_exp += val;
                }
                for (auto& val : scores) {
                    val /= sum_exp;
                }

                // Weighted sum of V
                for (size_t j = 0; j < seq_len; ++j) {
                    for (size_t dv = 0; dv < d_v; ++dv) {
                        attention_local[h * seq_len + i][dv] += scores[j] * V_local[h * seq_len + j][dv];
                    }
                }
            }
        }
        auto attention_end = high_resolution_clock::now();
        double attention_time = duration_cast<microseconds>(attention_end - attention_start).count() / 1000.0; // ms
        timings.mha_forward += attention_time;

        // Flatten the local attention outputs
        vector<float> attention_flat(local_num_heads * seq_len * d_v, 0.0f);
        for (size_t h = 0; h < local_num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t dv = 0; dv < d_v; ++dv) {
                    attention_flat[h * seq_len * d_v + i * d_v + dv] = attention_local[h * seq_len + i][dv];
                }
            }
        }

        // Root process prepares buffer to receive all attention outputs
        vector<float> all_attention_flat;
        size_t total_heads = num_heads;
        size_t total_attention_size = total_heads * seq_len * d_v;

        if (mpi_rank == 0) {
            all_attention_flat.resize(total_attention_size, 0.0f);
        }

        // Gather all attention_flat from each process to all_attention_flat at root
        vector<int> recvcounts(mpi_size, 0);
        vector<int> displs(mpi_size, 0);

        // Calculate recvcounts and displacements
        if (mpi_rank == 0) {
            for (int p = 0; p < mpi_size; ++p) {
                size_t proc_heads = num_heads / mpi_size + (p < (num_heads % mpi_size) ? 1 : 0);
                recvcounts[p] = proc_heads * seq_len * d_v;
                displs[p] = (p > 0) ? (displs[p-1] + recvcounts[p-1]) : 0;
            }
        }

        // Gather all attention outputs to root
        MPI_Gatherv(attention_flat.data(), local_num_heads * seq_len * d_v, MPI_FLOAT,
                    all_attention_flat.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);

        // Root process reconstructs the attention_heads
        vector<vector<float>> attention_heads;
        if (mpi_rank == 0) {
            attention_heads.reserve(num_heads * seq_len);
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    vector<float> head(d_v, 0.0f);
                    for (size_t j = 0; j < d_v; ++j) {
                        head[j] = all_attention_flat[h * seq_len * d_v + i * d_v + j];
                    }
                    attention_heads.emplace_back(head);
                }
            }
        }

        auto comm_end = high_resolution_clock::now();
        double comm_time = duration_cast<microseconds>(comm_end - attention_end).count() / 1000.0; // ms
        timings.mpi_comm += comm_time;

        // Only root process reconstructs the concatenated matrix
        vector<vector<float>> concat;
        if (mpi_rank == 0) {
            concat = concatenate_heads(attention_heads, num_heads, seq_len, d_v); // (seq_len, d_model)
        }

        return concat; // Only root process will have valid data
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

    FeedForward(size_t d_model_, size_t d_ff_) : d_model(d_model_), d_ff(d_ff_) {
        // Initialize weights
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

    EncoderLayer(size_t d_model_, size_t num_heads_, size_t d_ff_) :
        d_model(d_model_), num_heads(num_heads_), d_ff(d_ff_),
        mha(d_model_, num_heads_), ff(d_model_, d_ff_) 
    {
        // Initialize gamma and beta for layer normalization
        gamma = vector<float>(d_model, 1.0f);
        beta = vector<float>(d_model, 0.0f);
    }

    // Forward pass with OpenMP optimization
    vector<vector<float>> forward(const vector<vector<float>>& input, Timings& timings, int mpi_rank, int mpi_size) {
        // Multi-Head Attention
        vector<vector<float>> mha_out = mha.forward(input, timings, mpi_rank, mpi_size);

        // Residual connection and layer normalization
        auto res_norm1_start = high_resolution_clock::now();
        vector<vector<float>> add1 = add_vectors(input, mha_out);
        vector<vector<float>> norm1 = layer_norm(add1, gamma, beta);
        auto res_norm1_end = high_resolution_clock::now();
        double res_norm1_time = duration_cast<microseconds>(res_norm1_end - res_norm1_start).count() / 1000.0; // ms
        timings.layer_norm1 += res_norm1_time;

        // Feed Forward
        vector<vector<float>> ff_out = ff.forward(norm1, timings);

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

    TransformerEncoder(size_t d_model, size_t num_heads, size_t d_ff, size_t num_layers_) : num_layers(num_layers_) {
        for (size_t i = 0; i < num_layers_; ++i) {
            layers.emplace_back(EncoderLayer(d_model, num_heads, d_ff));
        }
    }

    vector<vector<float>> forward(const vector<vector<float>>& input, Timings& timings, int mpi_rank, int mpi_size) {
        vector<vector<float>> output = input;
        for (size_t i = 0; i < num_layers; ++i) {
            output = layers[i].forward(output, timings, mpi_rank, mpi_size);
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

    // 参数设置
    string filename = "/Users/tanglu/csi596-project/dataset_vectors.txt";
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
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (mpi_rank == 0) {
        cout << "Loaded " << dataset_flat.size() << " samples." << endl;
    }

    // 初始化 Transformer Encoder
    TransformerEncoder encoder(d_model, num_heads, d_ff, num_layers);

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
        vector<vector<float>> encoder_output = encoder.forward(sample, timings, mpi_rank, mpi_size);
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

    // 结束 MPI
    MPI_Finalize();
    return 0;
};