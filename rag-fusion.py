# RAG Fusion 实现（Qwen3 Embedding + BM25 + Qwen Reranker）

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from sentence_transformers.util import cos_sim
import torch
import faiss

# ------------------ 模型初始化 ------------------
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding")
qwen_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding", device_map="auto")
rerank_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Reranker")
rerank_model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen-Reranker", device_map="auto")

# ------------------ 函数定义 ------------------
def encode_qwen(texts):
    inputs = qwen_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        emb = qwen_model(**inputs).last_hidden_state[:, 0]  # [CLS] 向量
        emb = torch.nn.functional.normalize(emb, dim=1)
    return emb.cpu()

def rerank(query, docs):
    inputs = rerank_tokenizer([query] * len(docs), docs, padding=True, truncation=True, return_tensors="pt").to(rerank_model.device)
    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze()
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return sorted_docs

def rrf_fusion(results_list, k=60):
    from collections import defaultdict
    scores = defaultdict(float)
    for result in results_list:
        for rank, doc in enumerate(result):
            scores[doc] += 1 / (k + rank)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# ------------------ 示例数据 ------------------
corpus = [
    "Qwen3 是一个开源的大语言模型。",
    "BM25 是一种经典的信息检索算法。",
    "RAG Fusion 是检索增强生成的高级策略。",
    "Qwen-Reranker 可用于文档重排序。",
    "向量检索在大模型应用中非常常见。"
]

# ------------------ 索引构建 ------------------
# 向量检索器（Qwen3 Embedding + FAISS）
corpus_embeddings = encode_qwen(corpus).numpy()
faiss_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
faiss_index.add(corpus_embeddings)

# BM25 检索器
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ------------------ Query 检索与融合 ------------------
query = "大模型中的文档检索方法"

# Qwen3 向量检索
query_embedding = encode_qwen([query]).numpy()
_, faiss_ids = faiss_index.search(query_embedding, k=3)
results_qwen = [corpus[i] for i in faiss_ids[0]]

# BM25 检索
results_bm25 = bm25.get_top_n(query.split(), corpus, n=3)

# RRF 融合
fused_docs = rrf_fusion([results_qwen, results_bm25])

# Reranker 精排
final_docs = rerank(query, fused_docs)

# 输出结果
print("\n【最终排序结果】")
for i, doc in enumerate(final_docs):
    print(f"{i+1}. {doc}")
