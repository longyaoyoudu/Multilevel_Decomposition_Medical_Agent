# 环境准备
运行requirements.txt配置相关环境

本项目使用的硬件设备为nvidiaA800显卡以及4070显卡，部署时可以根据个人硬件情况进行调整

# 一、知识图谱构建

本项目所使用的数据是整合垂直型医药网站，构建起的包含7类规模为4.4万的知识实体，11类规模约30万实体关系的知识图谱。

运行build_medicalgraph.py文件建立图数据库。

<img width="865" height="341" alt="image" src="https://github.com/user-attachments/assets/e822272a-d39f-4796-999e-ddd15eb12e2d" />

<img width="865" height="348" alt="image" src="https://github.com/user-attachments/assets/65319d76-fa42-4e7a-aa04-98b99b7bf7c7" />

# 二、LLM模型配置

本项目选择三种方式进行模型配置

## 1.ollama本地部署

第一种方式是使用ollama在本地部署qwen3：0.6b，运行ollama.py

```python
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama 默认 API 地址
MODEL_NAME = "qwen3:0.6b"  # 替换为您使用的 Ollama 模型名称


def predict_ollama(data):
    """
    调用本地 Ollama API 进行模型推理
    """
    # 构造 Ollama API 请求体
    payload = {
        "model": MODEL_NAME,
        "prompt": data["message"][0]["content"],
        "stream": False,  # 非流式响应
        "options": {
            "temperature": data.get("temperature", 0.8),
            "top_p": data.get("top_p", 0.9),
            "top_k": data.get("top_k", 40),
            "num_predict": data.get("max_tokens", 512),
            "repeat_penalty": data.get("repetition_penalty", 1.1),
            "seed": data.get("seed", 42),
            "stop": data.get("stop", []),
        }
    }
```

## 2.开源模型本地部署

第二种方式是使用Qwen-7B-Chat-Int4开源模型本地部署，运行qwen7b_server.py

```python
os.environ['MODELSCOPE_ENDPOINT'] = 'https://mirror.sjtu.edu.cn'
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",  # ModelScope 的模型ID
    cache_dir="C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",  # ModelScope 的模型ID
    cache_dir="C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models",
    trust_remote_code=True
).cuda()
model = model.to(device)
```

## 3.接入硅基流动平台大模型

第三种方式是使用硅基流动平台上的大模型，运行SiliconFlow.py

```python
SILICONFLOW_API_KEY = ""  # 替换为你的API密钥
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"  # 硅基流动API端点
MODEL_NAME = "deepseek-ai/DeepSeek-V3"  # 或硅基平台上的其他模型名称


def predict_siliconflow(data):
    """调用硅基流动API进行模型推理"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": data["message"],
        "max_tokens": data.get("max_tokens", 512),
        "temperature": data.get("temperature", 0.8),
        "top_p": data.get("top_p", 0.9),
        "repetition_penalty": data.get("repetition_penalty", 1.1),
        "stream": False
    }

    response = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload)
    response.raise_for_status()  # 如果请求失败会抛出异常

    result = response.json()
    return result["choices"][0]["message"]["content"]
```

# 三、可视化对话

使用gradio进行可视化对话，运行chatapp.py

<img width="865" height="431" alt="image" src="https://github.com/user-attachments/assets/6020300a-dc7d-4d87-ac76-4e146a388ca2" />




