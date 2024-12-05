// transformer.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <omp.h> // 引入 OpenMP 头文件

using namespace std;

// Helper function: Matrix multiplication with OpenMP
vector<vector<float>> matmul(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t common = B.size();
    vector<vector<float>> result(rows, vector<float>(cols, 0.0f));

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < common; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

// Helper function: Add bias with OpenMP
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

// Helper function: Apply softmax to rows of a matrix with OpenMP
void softmax_rows(vector<vector<float>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        float max_val = *max_element(matrix[i].begin(), matrix[i].end());
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = exp(matrix[i][j] - max_val);
            sum_exp += matrix[i][j];
        }
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] /= sum_exp;
        }
    }
}

// Layer normalization with OpenMP
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

    // Forward pass
    vector<vector<float>> forward(const vector<vector<float>>& input) {
        // Linear projections
        vector<vector<float>> Q = matmul(input, W_Q);
        vector<vector<float>> K = matmul(input, W_K);
        vector<vector<float>> V = matmul(input, W_V);

        // Split into heads
        vector<vector<float>> Q_heads = split_heads(Q);
        vector<vector<float>> K_heads = split_heads(K);
        vector<vector<float>> V_heads = split_heads(V);

        // Scaled dot-product attention for each head
        vector<vector<float>> attention_heads(num_heads * Q_heads.size() / num_heads, vector<float>(d_v, 0.0f));

        #pragma omp parallel for
        for (size_t h = 0; h < num_heads; ++h) {
            // Extract head-specific Q, K, V
            vector<vector<float>> Q_head = get_head(Q_heads, h);
            vector<vector<float>> K_head = get_head(K_heads, h);
            vector<vector<float>> V_head = get_head(V_heads, h);

            // Compute scores = Q * K^T
            vector<vector<float>> scores = matmul(Q_head, transpose(K_head));

            // Scale scores
            float scale = sqrt((float)d_k);
            for (auto& row : scores) {
                for (auto& val : row) val /= scale;
            }

            // Apply softmax
            softmax_rows(scores);

            // Compute attention = scores * V
            vector<vector<float>> attention = matmul(scores, V_head);

            // Assign to attention_heads
            for (size_t i = 0; i < attention.size(); ++i) {
                attention_heads[h * attention.size() + i] = attention[i];
            }
        }

        // Concatenate all heads
        // Final linear projection
        vector<vector<float>> output = matmul(attention_heads, W_O);
        return output;
    }

private:
    // Function to generate random matrix
    vector<vector<float>> random_matrix(size_t rows, size_t cols) {
        vector<vector<float>> mat(rows, vector<float>(cols, 0.0f));
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                mat[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.02f - 0.01f; // Small random values
        return mat;
    }

    // Function to split heads
    vector<vector<float>> split_heads(const vector<vector<float>>& X) {
        // Assuming X has shape (seq_len, d_model)
        // We split into (num_heads * seq_len, d_k)
        size_t seq_len = X.size();
        vector<vector<float>> X_split;
        X_split.reserve(num_heads * seq_len);

        #pragma omp parallel for
        for (size_t h = 0; h < num_heads; ++h) {
            vector<vector<float>> temp;
            temp.reserve(seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                vector<float> head(d_k, 0.0f);
                for (size_t j = 0; j < d_k; ++j) {
                    head[j] = X[i][h * d_k + j];
                }
                #pragma omp critical
                X_split.push_back(head);
            }
        }
        return X_split;
    }

    // Function to get specific head
    vector<vector<float>> get_head(const vector<vector<float>>& X_heads, size_t head) {
        size_t seq_len = X_heads.size() / num_heads;
        vector<vector<float>> head_matrix(seq_len, vector<float>(d_k, 0.0f));
        for (size_t i = 0; i < seq_len; ++i) {
            head_matrix[i] = X_heads[head * seq_len + i];
        }
        return head_matrix;
    }

    // Function to transpose a matrix
    vector<vector<float>> transpose(const vector<vector<float>>& X) {
        if (X.empty()) return {};
        size_t rows = X.size();
        size_t cols = X[0].size();
        vector<vector<float>> X_T(cols, vector<float>(rows, 0.0f));

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                X_T[j][i] = X[i][j];
        return X_T;
    }

    // Function to concatenate heads
    // Not needed as attention_heads are already concatenated in forward()
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

    // Forward pass
    vector<vector<float>> forward(const vector<vector<float>>& input) {
        // Linear layer 1
        vector<vector<float>> hidden = matmul(input, W1);
        add_bias(hidden, b1);
        // ReLU activation
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < hidden.size(); ++i) {
            for (size_t j = 0; j < hidden[0].size(); ++j) {
                hidden[i][j] = max(0.0f, hidden[i][j]);
            }
        }
        // Linear layer 2
        vector<vector<float>> output = matmul(hidden, W2);
        add_bias(output, b2);
        return output;
    }

private:
    // Function to generate random matrix
    vector<vector<float>> random_matrix(size_t rows, size_t cols) {
        vector<vector<float>> mat(rows, vector<float>(cols, 0.0f));
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                mat[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.02f - 0.01f; // Small random values
        return mat;
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

    // Forward pass
    vector<vector<float>> forward(const vector<vector<float>>& input) {
        // Multi-Head Attention
        vector<vector<float>> mha_out = mha.forward(input);
        // Residual connection and layer normalization
        vector<vector<float>> add1 = add_vectors(input, mha_out);
        vector<vector<float>> norm1 = layer_norm(add1, gamma, beta);
        
        // Feed Forward
        vector<vector<float>> ff_out = ff.forward(norm1);
        // Residual connection and layer normalization
        vector<vector<float>> add2 = add_vectors(norm1, ff_out);
        vector<vector<float>> norm2 = layer_norm(add2, gamma, beta);

        return norm2;
    }

private:
    // Function to add two matrices element-wise with OpenMP
    vector<vector<float>> add_vectors(const vector<vector<float>>& a, const vector<vector<float>>& b) {
        assert(a.size() == b.size());
        assert(a[0].size() == b[0].size());
        size_t rows = a.size();
        size_t cols = a[0].size();
        vector<vector<float>> result(rows, vector<float>(cols, 0.0f));

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[i][j] = a[i][j] + b[i][j];

        return result;
    }
};

// Helper function: Load dataset from file
vector<vector<float>> load_dataset(const string& filename, size_t sequence_length, size_t embedding_dim) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    vector<vector<float>> dataset;
    string line;
    while (getline(file, line)) {
        istringstream stream(line);
        vector<float> sample;
        float value;
        while (stream >> value) {
            sample.push_back(value);
        }
        if (sample.size() != sequence_length * embedding_dim) {
            cerr << "Sample size mismatch. Expected " << sequence_length * embedding_dim
                 << ", got " << sample.size() << endl;
            continue; // Skip malformed samples
        }
        dataset.push_back(sample);
    }
    file.close();
    return dataset;
}

int main() {
    // Parameters
    string filename = "dataset_vectors.txt";
    size_t sequence_length = 10; // 与 Python 预处理一致
    size_t embedding_dim = 50;
    size_t d_model = 50; // Must match embedding_dim
    size_t num_heads = 5; // d_model should be divisible by num_heads
    size_t d_ff = 128; // Feed forward hidden size

    // Load dataset
    cout << "Loading dataset..." << endl;
    vector<vector<float>> dataset = load_dataset(filename, sequence_length, embedding_dim);
    if (dataset.empty()) {
        cerr << "No data loaded. Exiting." << endl;
        return EXIT_FAILURE;
    }
    cout << "Loaded " << dataset.size() << " samples." << endl;

    // For simplicity, process only the first sample
    // Reshape flat sample to (seq_len, embed_dim)
    vector<vector<float>> sample(sequence_length, vector<float>(embedding_dim, 0.0f));
    for (size_t i = 0; i < sequence_length; ++i)
        for (size_t j = 0; j < embedding_dim; ++j)
            sample[i][j] = dataset[0][i * embedding_dim + j];

    // Add positional encoding
    vector<vector<float>> pos_enc = positional_encoding(sequence_length, d_model);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < sequence_length; ++i)
        for (size_t j = 0; j < d_model; ++j)
            sample[i][j] += pos_enc[i][j];

    // Initialize Encoder Layer
    EncoderLayer encoder(d_model, num_heads, d_ff);

    // Forward pass through Encoder
    vector<vector<float>> encoder_output = encoder.forward(sample);

    // Print Encoder Output
    cout << "Encoder Output:" << endl;
    for (size_t i = 0; i < sequence_length; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            cout << encoder_output[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}