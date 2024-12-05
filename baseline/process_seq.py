import gensim.downloader as api
import numpy as np
import json

# 1. 加载预训练的 Word2Vec 模型
print("Loading Word2Vec model...")
word2vec = api.load("glove-wiki-gigaword-50")  # 使用 GloVe，向量维度为 50

# 2. 加载数据集（莎士比亚文集）
print("Loading dataset...")
with open("/Users/tanglu/csi596-project/baseline/process_seq.py", "r", encoding="utf-8") as f:
    dataset = f.readlines()

# 3. 数据预处理
def preprocess_text(text):
    # 简单预处理：小写、去除标点和多余空格
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text.strip().split()

# 4. 将文本转化为向量
def text_to_vector(text, model, max_len=10, embedding_dim=50):
    words = preprocess_text(text)
    vectors = []
    for word in words[:max_len]:  # 截断到 max_len
        if word in model:
            vectors.append(model[word])
        else:
            vectors.append(np.zeros(embedding_dim))  # 未知词用零填充
    # 填充不足长度
    while len(vectors) < max_len:
        vectors.append(np.zeros(embedding_dim))
    return np.array(vectors)

# 5. 处理数据集
print("Processing dataset...")
max_sequence_length = 10  # 固定序列长度
embedding_dim = 50  # 嵌入维度

processed_data = []
for line in dataset[:100]:  # 只处理前 100 行示例数据
    vectorized = text_to_vector(line, word2vec, max_sequence_length, embedding_dim)
    processed_data.append(vectorized)

processed_data = np.array(processed_data)
print(f"Processed data shape: {processed_data.shape}")  # (样本数, max_sequence_length, embedding_dim)

# 6. 保存为文件
output_file = "dataset_vectors.txt"
np.savetxt(output_file, processed_data.reshape(len(processed_data), -1))  # 保存为 2D 数组
print(f"Processed data saved to {output_file}")