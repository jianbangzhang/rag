from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser

# 设置嵌入模型
# modelscope download --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  --local_dir ./embed
embed_model = HuggingFaceEmbedding(model_name="/home/whu/qwen3/embed")
Settings.embed_model = embed_model

# 设置语言模型
# modelscope download --model Qwen/Qwen3-1.7B README.md --local_dir ./model
llm = HuggingFaceLLM(
    model_name="/home/whu/qwen3/model",
    tokenizer_name="/home/whu/qwen3/model",
    model_kwargs={"trust_remote_code": True},
    tokenizer_kwargs={"trust_remote_code": True},
)
Settings.llm = llm

# 加载数据
documents = SimpleDirectoryReader("/home/whu/qwen3/data").load_data()

# 解析节点
node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
base_node = node_parser.get_nodes_from_documents(documents)

# 构建索引
index = VectorStoreIndex(nodes=base_node)

# 查询引擎
query_engine = index.as_query_engine()
rsp = query_engine.query("水平负载是什么")

print(rsp)
