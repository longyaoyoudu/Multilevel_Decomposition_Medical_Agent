# coding: utf-8
"""
向量检索模块
从知识图谱导出三元组，构建Chroma向量索引，支持ANN检索
"""
import os
import sys

# 禁用ChromaDB遥测
os.environ['CHROMA_TELEMETRY'] = '0'
os.environ['ANONYMIZED_TELEMETRY'] = '0'

import chromadb
from chromadb.config import Settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from utils.embedding_model import embedding_model


class VectorRetriever:
    """向量检索器，基于ChromaDB实现ANN相似度检索"""

    def __init__(self, persist_dir=None):
        """
        初始化向量检索器

        Args:
            persist_dir: 向量数据库持久化路径，默认从Config读取
        """
        self.persist_dir = persist_dir or Config.VECTOR_PERSIST_DIR
        self.collection_name = "medical_triples"

        # 确保存储目录存在
        os.makedirs(self.persist_dir, exist_ok=True)

        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 初始化collection
        self._ensure_collection()

        # Embedding模型
        self.encoder = embedding_model

        print(f"[VectorRetriever] 初始化完成，存储路径: {self.persist_dir}")

    def _ensure_collection(self):
        """确保collection存在"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"[VectorRetriever] 加载已有Collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Medical knowledge triples for RAG"}
            )
            print(f"[VectorRetriever] 创建新Collection: {self.collection_name}")

    def build_index(self, triples, batch_size=100):
        """
        从三元组列表构建向量索引

        Args:
            triples: 三元组列表 [(head, rel, tail), ...]
            batch_size: 批处理大小
        """
        if self.collection.count() > 0:
            print(f"⚠️ 向量索引已存在 ({self.collection.count()} 条)，将跳过构建")
            print("   如需重新构建，请先调用 delete_index()")
            return

        print(f"[VectorRetriever] 开始构建索引，共 {len(triples)} 条三元组...")

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i+batch_size]
            docs = []
            ids = []
            metadatas = []

            for j, (head, rel, tail) in enumerate(batch):
                # 构建文档字符串: "头实体的关系是尾实体"
                doc = f"{head}的{rel}是{tail}"
                docs.append(doc)
                ids.append(f"triple_{i+j}")
                metadatas.append({
                    "head": head,
                    "rel": rel,
                    "tail": tail,
                    "original_triple": f"<{head},{rel},{tail}>"
                })

            # 批量编码
            embeddings = self.encoder.encode(docs).tolist()

            # 添加到collection
            self.collection.add(
                embeddings=embeddings,
                documents=docs,
                ids=ids,
                metadatas=metadatas
            )

            progress = min(i + batch_size, len(triples))
            print(f"   构建进度: {progress}/{len(triples)}")

        print(f"✅ 向量索引构建完成，共 {self.collection.count()} 条")

    def search(self, query, top_k=50, filter_dict=None):
        """
        向量相似度检索

        Args:
            query: 查询文本
            top_k: 返回前K条结果
            filter_dict: 元数据过滤条件，e.g. {"head": "糖尿病"}

        Returns:
            list: 检索结果列表
            [{
                'id': str,
                'doc': str,
                'metadata': dict,
                'distance': float,
                'triple': tuple
            }, ...]
        """
        # 编码查询
        query_emb = self.encoder.encode_query(query)
        print(f"[VectorRetriever] Query编码完成，维度: {query_emb.shape}")

        # 构建where条件
        where = None
        if filter_dict:
            where = filter_dict

        # 执行检索
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # 格式化结果
        formatted_results = []
        for idx in range(len(results['ids'][0])):
            meta = results['metadatas'][0][idx]
            formatted_results.append({
                "id": results['ids'][0][idx],
                "doc": results['documents'][0][idx],
                "metadata": meta,
                "distance": results['distances'][0][idx],
                "triple": (meta['head'], meta['rel'], meta['tail'])
            })

        print(f"[VectorRetriever] 检索到 {len(formatted_results)} 条结果")
        return formatted_results

    def search_by_entity(self, entity_name, entity_type=None, top_k=50):
        """
        根据实体名称检索相关三元组

        Args:
            entity_name: 实体名称
            entity_type: 实体类型（可选）
            top_k: 返回前K条结果

        Returns:
            list: 检索结果列表
        """
        filter_dict = {"head": entity_name}
        if entity_type:
            filter_dict["entity_type"] = entity_type

        return self.search(entity_name, top_k=top_k, filter_dict=filter_dict)

    def get_by_id(self, doc_id):
        """
        根据ID获取单条记录

        Args:
            doc_id: 文档ID

        Returns:
            dict: 记录详情
        """
        result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if result['ids']:
            return {
                "id": result['ids'][0],
                "doc": result['documents'][0],
                "metadata": result['metadatas'][0]
            }
        return None

    def count(self):
        """获取索引中的记录数"""
        return self.collection.count()

    def delete_index(self):
        """删除向量索引"""
        self.client.delete_collection(self.collection_name)
        print("[VectorRetriever] 向量索引已删除")
        self._ensure_collection()

    def exists(self):
        """检查索引是否已构建"""
        try:
            count = self.collection.count()
            return count > 0
        except:
            return False


# 全局向量检索器实例（延迟初始化）
_vector_retriever = None


def get_vector_retriever():
    """获取全局向量检索器单例"""
    global _vector_retriever
    if _vector_retriever is None:
        _vector_retriever = VectorRetriever()
    return _vector_retriever


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试向量检索器")
    print("=" * 60)

    retriever = VectorRetriever()

    # 检查索引是否存在
    print(f"\n索引状态: {'已构建' if retriever.exists() else '未构建'}")
    print(f"索引记录数: {retriever.count()}")

    # 测试检索（如果索引已构建）
    if retriever.exists():
        print("\n测试检索...")
        results = retriever.search("糖尿病症状", top_k=3)
        for r in results:
            print(f"  [{r['distance']:.4f}] {r['doc']}")
    else:
        print("\n⚠️ 索引未构建，跳过检索测试")

    print("\n✅ 向量检索器测试完成")