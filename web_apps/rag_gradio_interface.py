import gradio as gr
from vectordb.faiss_vector_store import FaissVectorStore
from llm.openllm import OpenLlm
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 LLM 和向量数据库
llm = OpenAI()
emb = OpenLlm()
vector_db = FaissVectorStore.from_persist_path("kb/index", "kb/docs")

# 修改 rag_pipeline 函数，支持多轮对话

def rag_pipeline(query, history):
    """
    基于用户输入的 query 和对话历史 history，执行多轮对话的 RAG 流程。
    """
    try:
        # 生成查询向量
        query_vector = emb.emb(model="text-embedding-v4", text=query)

        # 检索相关文档
        res = vector_db.query(query_emb=query_vector)
        retrieval_text = ["【文档{}】：\n{}".format(idx, text) for idx, text in enumerate(res)]

        # 构造提示词，包含对话历史
        history_text = "\n".join(["用户：{}\n助手：{}".format(h[0], h[1]) for h in history])
        ans_prompt = (
            "你是魔搭社区的答疑小助手，请你根据参考文档来回答用户的问题。 \n"
            "以下是用户和你的对话历史：\n{history}\n"
            "问题：{query} \n参考文档：\n{docs}"
        )

        # 调用 LLM 生成回答
        response = llm.chat.completions.create(
            model="qwen3-8b",
            messages=[
                {"role": "user", "content": ans_prompt.format(query=query, docs="\n".join(retrieval_text), history=history_text)}
            ],
            temperature=0,
            extra_body={"enable_thinking": False},
        )

        # 获取回答内容
        answer = response.choices[0].message.content

        # 更新对话历史
        history.append((query, answer))

        # 返回检索结果和更新后的对话历史
        return "\n".join(retrieval_text), history

    except Exception as e:
        return "检索或生成回答时出错：" + str(e), history

# 修改 Gradio 界面，支持多轮对话

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # 魔搭社区答疑助手
        输入你的问题，系统将从知识库中检索相关文档并生成回答。
        """)

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="多轮对话")
                query_input = gr.Textbox(label="请输入你的问题", placeholder="例如：魔搭平台是做什么的？")
                submit_button = gr.Button("发送")

            with gr.Column():
                retrieval_output = gr.Textbox(label="检索到的文档", lines=10, interactive=False)

        # 初始化对话历史
        history_state = gr.State([])

        submit_button.click(
            fn=rag_pipeline,
            inputs=[query_input, history_state],
            outputs=[retrieval_output, chatbot]
        )

    return demo

# 启动 Gradio 应用
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)