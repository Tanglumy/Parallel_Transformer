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
    double total = 0.0;
};

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

// Function to generate random matrix with small values
vector<vector<float>> random_matrix(size_t rows, size_t cols) {
    vector<vector<float>> mat(rows, vector<float>(cols, 0.0f));
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.02f - 0.01f; // Small random values
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

// Multi-Head Self-Attention
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

    // Forward pass with OpenMP timing
    vector<vector<float>> forward(const vector<vector<float>>& input, Timings& timings) {
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

        // Compute attention for each head
        auto attention_start = high_resolution_clock::now();
        vector<vector<float>> attention_heads; // Will store all heads' output
        attention_heads.reserve(num_heads * seq_len); // Preallocate

        #pragma omp parallel for
        for (size_t h = 0; h < num_heads; ++h) {
            // Extract h-th head
            vector<vector<float>> Q_h, K_h, V_h;
            Q_h.reserve(seq_len);
            K_h.reserve(seq_len);
            V_h.reserve(seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                Q_h.emplace_back(Q_heads[h * seq_len + i]);
                K_h.emplace_back(K_heads[h * seq_len + i]);
                V_h.emplace_back(V_heads[h * seq_len + i]);
            }

            // Compute scores = Q * K^T
            vector<vector<float>> K_h_T = transpose(K_h); // (d_k, seq_len)
            vector<vector<float>> scores = matmul(Q_h, K_h_T); // (seq_len, seq_len)

            // Scale scores
            float scale = sqrt(static_cast<float>(d_k));
            #pragma omp parallel for
            for (size_t i = 0; i < scores.size(); ++i)
                for (size_t j = 0; j < scores[i].size(); ++j)
                    scores[i][j] /= scale;

            // Apply softmax
            softmax_rows(scores); // (seq_len, seq_len)

            // Weighted sum of V
            vector<vector<float>> head = matmul(scores, V_h); // (seq_len, d_v)

            // Protect concurrent writes
            #pragma omp critical
            {
                for (size_t i = 0; i < seq_len; ++i)
                    attention_heads.emplace_back(head[i]);
            }
        }
        auto attention_end = high_resolution_clock::now();
        double attention_time = duration_cast<microseconds>(attention_end - attention_start).count() / 1000.0; // ms
        timings.mha_forward += attention_time;

        // Concatenate all heads
        auto concat_start = high_resolution_clock::now();
        vector<vector<float>> concat = concatenate_heads(attention_heads, num_heads, seq_len, d_v); // (seq_len, d_model)
        auto concat_end = high_resolution_clock::now();
        double concat_time = duration_cast<microseconds>(concat_end - concat_start).count() / 1000.0; // ms
        timings.mha_forward += concat_time;

        // Final linear projection
        auto proj_out_start = high_resolution_clock::now();
        vector<vector<float>> output = matmul(concat, W_O); // (seq_len, d_model)
        auto proj_out_end = high_resolution_clock::now();
        double proj_out_time = duration_cast<microseconds>(proj_out_end - proj_out_start).count() / 1000.0; // ms
        timings.mha_forward += proj_out_time;

        return output;
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

// Function to add two matrices element-wise with OpenMP optimization
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
    vector<vector<float>> forward(const vector<vector<float>>& input, Timings& timings) {
        // Multi-Head Attention
        vector<vector<float>> mha_out = mha.forward(input, timings);

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

// Function to load dataset from file with OpenMP timing
// Each line in the file should contain (seq_len * embed_dim) float numbers separated by spaces
vector<vector<float>> load_dataset(const string& filename, size_t sequence_length, size_t embedding_dim, Timings& timings) {
    auto start = high_resolution_clock::now();

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        exit(EXIT_FAILURE);
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

int main() {
    // 参数设置
    string filename = "/Users/tanglu/csi596-project/dataset_vectors.txt";
    size_t sequence_length = 10; // 与 Python 预处理一致
    size_t embedding_dim = 50;
    size_t d_model = 50; // Must match embedding_dim
    size_t num_heads = 5; // d_model should be divisible by num_heads
    size_t d_ff = 128; // Feed forward hidden size

    // 初始化 Timings 结构体
    Timings timings;

    // 总时间起点
    auto total_start = high_resolution_clock::now();

    // 加载数据集
    cout << "Loading dataset..." << endl;
    vector<vector<float>> dataset_flat = load_dataset(filename, sequence_length, embedding_dim, timings);
    if (dataset_flat.empty()) {
        cerr << "No data loaded. Exiting." << endl;
        return EXIT_FAILURE;
    }
    cout << "Loaded " << dataset_flat.size() << " samples." << endl;

    // 选择第一个样本进行处理
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

    // 初始化 Encoder Layer
    EncoderLayer encoder(d_model, num_heads, d_ff);

    // 前向传播
    auto encoder_start = high_resolution_clock::now();
    vector<vector<float>> encoder_output = encoder.forward(sample, timings);
    auto encoder_end = high_resolution_clock::now();
    double encoder_time = duration_cast<microseconds>(encoder_end - encoder_start).count() / 1000.0; // ms
    timings.total += encoder_time;

    // 总时间结束
    auto total_end = high_resolution_clock::now();
    double total_time = duration_cast<microseconds>(total_end - total_start).count() / 1000.0; // ms
    timings.total = total_time;

    // 打印 Encoder 输出（可选）
    /*
    cout << "Encoder Output:" << endl;
    for (size_t i = 0; i < sequence_length; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            cout << encoder_output[i][j] << " ";
        }
        cout << endl;
    }
    */

    // 打印时间测量结果
    cout << fixed << setprecision(3);
    cout << "Execution Timings (in ms):" << endl;
    cout << "Data Loading        : " << timings.data_loading << " ms" << endl;
    cout << "Positional Encoding : " << timings.positional_encoding << " ms" << endl;
    cout << "Multi-Head Attention: " << timings.mha_forward << " ms" << endl;
    cout << "Feed Forward        : " << timings.feed_forward << " ms" << endl;
    cout << "Layer Norm 1        : " << timings.layer_norm1 << " ms" << endl;
    cout << "Layer Norm 2        : " << timings.layer_norm2 << " ms" << endl;
    cout << "Total Execution     : " << timings.total << " ms" << endl;

    // 保存时间测量结果到 CSV 文件
    ofstream timing_file("timings.csv");
    if (!timing_file.is_open()) {
        cerr << "Failed to open timings.csv for writing." << endl;
        return EXIT_FAILURE;
    }

    // 写入表头
    timing_file << "Module,Time_ms\n";

    // 写入各模块时间
    timing_file << "Data Loading," << timings.data_loading << "\n";
    timing_file << "Positional Encoding," << timings.positional_encoding << "\n";
    timing_file << "Multi-Head Attention," << timings.mha_forward << "\n";
    timing_file << "Feed Forward," << timings.feed_forward << "\n";
    timing_file << "Layer Norm 1," << timings.layer_norm1 << "\n";
    timing_file << "Layer Norm 2," << timings.layer_norm2 << "\n";
    timing_file << "Total Execution," << timings.total << "\n";

    timing_file.close();
    cout << "Timings saved to timings.csv" << endl;

    return 0;
}