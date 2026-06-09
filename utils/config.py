# coding: utf-8
"""
项目配置管理模块
统一管理.env文件中的所有配置项
"""
import os
from dotenv import load_dotenv

def get_env_path():
    """获取.env文件路径"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    return env_path

load_dotenv(get_env_path())

class Config:
    """配置管理类"""
    
    # ============================================================
    # Neo4j 数据库配置
    # ============================================================
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '978626572')
    
    # ============================================================
    # LLM 中间服务配置
    # ============================================================
    LLM_SERVICE_URL = os.getenv('LLM_SERVICE_URL', 'http://127.0.0.1:3001/generate')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '3001'))
    
    # ============================================================
    # MiniMax API 配置
    # ============================================================
    MINIMAX_API_KEY = os.getenv('MINIMAX_API_KEY', '')
    MINIMAX_GROUP_ID = os.getenv('MINIMAX_GROUP_ID', '')
    MINIMAX_API_URL = os.getenv('MINIMAX_API_URL', 'https://api.minimax.chat/v1/text/chatcompletion_v2')
    MINIMAX_MODEL_NAME = os.getenv('MINIMAX_MODEL_NAME', 'abab6.5s-chat')
    
    # ============================================================
    # SiliconFlow（硅基流动）API 配置
    # ============================================================
    SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY', '')
    SILICONFLOW_API_URL = os.getenv('SILICONFLOW_API_URL', 'https://api.siliconflow.cn/v1/chat/completions')
    SILICONFLOW_MODEL_NAME = os.getenv('SILICONFLOW_MODEL_NAME', 'deepseek-ai/DeepSeek-V3')
    
    # ============================================================
    # Ollama 本地模型配置
    # ============================================================
    OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
    OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'qwen3:0.6b')
    
    # ============================================================
    # 生成参数配置
    # ============================================================
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2048'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
    TOP_P = float(os.getenv('TOP_P', '0.9'))
    TOP_K = int(os.getenv('TOP_K', '40'))
    REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', '1.1'))
    
    # ============================================================
    # Web 应用配置
    # ============================================================
    GRADIO_PORT = int(os.getenv('GRADIO_PORT', '7860'))
    GRADIO_HOST = os.getenv('GRADIO_HOST', '127.0.0.1')

    # ============================================================
    # 向量检索配置
    # ============================================================
    VECTOR_PERSIST_DIR = os.getenv('VECTOR_PERSIST_DIR', './vector_db')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
    VECTOR_TOP_K = int(os.getenv('VECTOR_TOP_K', '50'))

    # ============================================================
    # Reranker 配置
    # ============================================================
    RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', '20'))
    RERANK_FINAL_K = int(os.getenv('RERANK_FINAL_K', '5'))
    RERANK_MODEL = os.getenv('RERANK_MODEL', 'MiniMax-M2.7-highspeed')

    @classmethod
    def get_neo4j_config(cls):
        """获取Neo4j配置字典"""
        return {
            'uri': cls.NEO4J_URI,
            'user': cls.NEO4J_USER,
            'password': cls.NEO4J_PASSWORD
        }
    
    @classmethod
    def get_minimax_config(cls):
        """获取MiniMax配置字典"""
        return {
            'api_key': cls.MINIMAX_API_KEY,
            'group_id': cls.MINIMAX_GROUP_ID,
            'api_url': cls.MINIMAX_API_URL,
            'model_name': cls.MINIMAX_MODEL_NAME
        }
    
    @classmethod
    def get_siliconflow_config(cls):
        """获取SiliconFlow配置字典"""
        return {
            'api_key': cls.SILICONFLOW_API_KEY,
            'api_url': cls.SILICONFLOW_API_URL,
            'model_name': cls.SILICONFLOW_MODEL_NAME
        }
    
    @classmethod
    def get_ollama_config(cls):
        """获取Ollama配置字典"""
        return {
            'api_url': cls.OLLAMA_API_URL,
            'model_name': cls.OLLAMA_MODEL_NAME
        }
    
    @classmethod
    def get_generation_params(cls):
        """获取生成参数配置字典"""
        return {
            'max_tokens': cls.MAX_TOKENS,
            'temperature': cls.TEMPERATURE,
            'top_p': cls.TOP_P,
            'top_k': cls.TOP_K,
            'repetition_penalty': cls.REPETITION_PENALTY
        }

    @classmethod
    def get_vector_config(cls):
        """获取向量检索配置字典"""
        return {
            'persist_dir': cls.VECTOR_PERSIST_DIR,
            'embedding_model': cls.EMBEDDING_MODEL,
            'top_k': cls.VECTOR_TOP_K
        }

    @classmethod
    def get_rerank_config(cls):
        """获取Reranker配置字典"""
        return {
            'top_k': cls.RERANK_TOP_K,
            'final_k': cls.RERANK_FINAL_K,
            'model': cls.RERANK_MODEL
        }

    @classmethod
    def validate_config(cls):
        """验证关键配置项"""
        issues = []
        
        if not cls.NEO4J_PASSWORD:
            issues.append("Neo4j 密码未配置")
        
        if not cls.MINIMAX_API_KEY and not cls.SILICONFLOW_API_KEY:
            issues.append("未配置任何LLM API Key (MiniMax 或 SiliconFlow)")
        
        return issues
    
    @classmethod
    def print_config_summary(cls):
        """打印配置摘要（用于调试）"""
        print("=" * 60)
        print("配置摘要")
        print("=" * 60)
        
        print("\n[Neo4j 数据库]")
        print(f"  URI: {cls.NEO4J_URI}")
        print(f"  用户: {cls.NEO4J_USER}")
        print(f"  密码: {'***' if cls.NEO4J_PASSWORD else '未配置'}")
        
        print("\n[LLM 中间服务]")
        print(f"  服务地址: {cls.LLM_SERVICE_URL}")
        print(f"  端口: {cls.FLASK_PORT}")
        
        print("\n[MiniMax API]")
        print(f"  API Key: {'已配置' if cls.MINIMAX_API_KEY else '未配置'}")
        print(f"  Group ID: {'已配置' if cls.MINIMAX_GROUP_ID else '未配置'}")
        print(f"  模型: {cls.MINIMAX_MODEL_NAME}")
        
        print("\n[SiliconFlow API]")
        print(f"  API Key: {'已配置' if cls.SILICONFLOW_API_KEY else '未配置'}")
        print(f"  模型: {cls.SILICONFLOW_MODEL_NAME}")
        
        print("\n[Ollama 本地模型]")
        print(f"  API 地址: {cls.OLLAMA_API_URL}")
        print(f"  模型: {cls.OLLAMA_MODEL_NAME}")
        
        print("\n[生成参数]")
        print(f"  max_tokens: {cls.MAX_TOKENS}")
        print(f"  temperature: {cls.TEMPERATURE}")
        print(f"  top_p: {cls.TOP_P}")
        print(f"  top_k: {cls.TOP_K}")
        print(f"  repetition_penalty: {cls.REPETITION_PENALTY}")
        
        print("\n[Web 应用]")
        print(f"  Gradio 地址: {cls.GRADIO_HOST}:{cls.GRADIO_PORT}")

        print("\n[向量检索]")
        print(f"  持久化路径: {cls.VECTOR_PERSIST_DIR}")
        print(f"  Embedding模型: {cls.EMBEDDING_MODEL}")
        print(f"  召回数量: {cls.VECTOR_TOP_K}")

        print("\n[Reranker]")
        print(f"  融合候选数: {cls.RERANK_TOP_K}")
        print(f"  最终输出数: {cls.RERANK_FINAL_K}")
        print(f"  模型: {cls.RERANK_MODEL}")

        print("\n" + "=" * 60)
        
        issues = cls.validate_config()
        if issues:
            print("⚠️  配置警告:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ 关键配置项已配置完成")
        
        print("=" * 60)


config = Config()

if __name__ == '__main__':
    config.print_config_summary()
