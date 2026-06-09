# coding: utf-8
"""
Embedding模型封装
统一管理Embedding模型加载和编码，支持离线使用
"""
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
from utils.config import Config


class EmbeddingModel:
    """Embedding模型单例封装"""
    _instance = None
    _model = None

    # 本地模型路径
    LOCAL_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'paraphrase-multilingual-MiniLM-L12-v2'
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        """加载模型（延迟加载，单例模式）"""
        if self._model is None:
            # 优先使用本地模型路径
            local_model_path = self.LOCAL_MODEL_PATH

            if os.path.exists(local_model_path):
                print(f"[EmbeddingModel] 使用本地模型: {local_model_path}")
                try:
                    self._model = SentenceTransformer(
                        local_model_path,
                        trust_remote_code=True
                    )
                    test_emb = self._model.encode(["测试"])
                    print(f"[EmbeddingModel] 本地模型加载完成")
                    return self._model
                except Exception as e:
                    print(f"[EmbeddingModel] 本地模型加载失败: {e}")

            # 如果本地模型不存在，尝试在线加载
            model_name = Config.EMBEDDING_MODEL
            print(f"[EmbeddingModel] 尝试在线加载模型: {model_name}")
            try:
                self._model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True
                )
                print(f"[EmbeddingModel] 在线模型加载完成: {model_name}")
            except Exception as e:
                print(f"[EmbeddingModel] 在线模型加载失败: {e}")
                print("[EmbeddingModel] 尝试备选模型...")
                try:
                    self._model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
                    print(f"[EmbeddingModel] 备选模型加载完成")
                except Exception as e2:
                    print(f"[EmbeddingModel] 备选模型也加载失败: {e2}")
                    raise

        return self._model

    def reset(self):
        """重置模型单例（用于重新加载）"""
        self._model = None
        print("[EmbeddingModel] 模型单例已重置")

    def encode(self, texts, batch_size=32):
        """
        批量文本编码

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            numpy.ndarray: 文本 embeddings
        """
        model = self.load_model()
        return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    def encode_query(self, query):
        """
        单条查询编码

        Args:
            query: 查询文本

        Returns:
            numpy.ndarray: query embedding
        """
        model = self.load_model()
        return model.encode([query])[0]

    def get_model_name(self):
        """获取模型名称"""
        return Config.EMBEDDING_MODEL


# 全局单例
embedding_model = EmbeddingModel()


if __name__ == "__main__":
    # 测试代码
    print("测试Embedding模型...")

    # 测试单条查询
    query = "糖尿病有什么症状"
    emb = embedding_model.encode_query(query)
    print(f"查询: {query}")
    print(f"Embedding维度: {emb.shape}")

    # 测试批量编码
    texts = ["糖尿病症状", "高血压治疗", "心脏病原因"]
    embs = embedding_model.encode(texts)
    print(f"批量编码: {texts}")
    print(f"Embedding维度: {embs.shape}")

    print("✅ Embedding模型测试通过")