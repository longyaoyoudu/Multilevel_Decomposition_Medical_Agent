import sys
import os
import time
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from qa_llm_enhanced.chat_with_llm import KGRAG

# 初始化问答系统
chatbot = KGRAG()

# 全局对话历史管理
conversations = {}
current_conversation_id = None

# 初始化当前对话
if current_conversation_id is None:
    current_conversation_id = str(uuid.uuid4())
    conversations[current_conversation_id] = {
        "id": current_conversation_id,
        "title": "新对话",
        "history": [],
        "created_at": time.time()
    }

# 颜色编码的日志格式化
def color_log(text, color="black"):
    """为不同级别的日志添加颜色"""
    color_map = {
        "error": "#ff4d4d",
        "warning": "#ff9900",
        "success": "#33cc33",
        "info": "#3399ff",
        "system": "#666666"
    }
    color_code = color_map.get(color, "black")
    return f"<span style='color: {color_code};'>{text}</span>"


# 处理用户查询的核心函数
def process_query(query, conversation_id):
    """处理用户查询并返回响应和日志，支持对话历史"""
    global conversations
    
    # 获取或创建对话
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "id": conversation_id,
            "title": query[:30] + "..." if len(query) > 30 else query,
            "history": [],
            "created_at": time.time()
        }
    
    conversation = conversations[conversation_id]
    history = conversation["history"]
    
    start_time = time.time()
    logs = []

    # 初始化日志
    logs.append(color_log(f"用户查询: {query}", "info"))

    # 构建包含历史上下文的提示
    context = ""
    if history:
        context = "\n".join([f"用户: {h[0]}\n助手: {h[1]}" for h in history])
        context += f"\n用户: {query}"
        logs.append(color_log(f"对话历史上下文:\n{context}", "info"))

    try:
        # Step 1: 实体链接
        logs.append(color_log("【步骤1】实体识别...", "system"))
        entity_dict = chatbot.entity_linking(query)
        logs.append(color_log(f"✅ 识别到实体: {entity_dict}", "success"))

        if not entity_dict:
            logs.append(color_log("⚠️ 未识别到相关实体", "warning"))
            answer = "抱歉，这个问题超出了我的知识范围。"
            history.append((query, answer))
            return answer, "\n".join(logs), history

        # Step 2: 知识图谱召回
        depth = 1
        logs.append(color_log(f"【步骤2】知识检索 (深度={depth})...", "system"))
        facts = set()

        for entity_name, types in entity_dict.items():
            for entity_type in types:
                # 关系链接
                logs.append(color_log(f"🔍 处理实体: {entity_name}({entity_type})", "info"))
                rels = chatbot.link_entity_rel(query, entity_name, entity_type)
                logs.append(color_log(f"  相关关系: {rels}", "success"))

                # 事实召回
                entity_triples = chatbot.recall_facts(rels, entity_type, entity_name, depth)
                facts.update(entity_triples)
                logs.append(color_log(f"  检索到三元组: {len(entity_triples)}条", "success"))

        # 结果截断
        max_facts = 15
        facts_list = list(facts)
        if len(facts_list) > max_facts:
            logs.append(color_log(f"⚠️ 三元组数量超过{max_facts}，进行截断", "warning"))
            facts_list = facts_list[:max_facts]

        # Step 3: 构建Prompt
        logs.append(color_log("【步骤3】构建提示词...", "system"))
        fact_prompt = chatbot.format_prompt(query, facts_list)
        logs.append(color_log(f"提示词内容:\n{fact_prompt}", "info"))

        # Step 4: LLM生成答案
        logs.append(color_log("【步骤4】生成回答...", "system"))
        answer = chatbot.chat(fact_prompt)
        logs.append(color_log(f"✅ 回答生成成功", "success"))

        # 计算处理时间
        elapsed = time.time() - start_time
        logs.append(color_log(f"⏱️ 总处理时间: {elapsed:.2f}秒", "info"))

        # 更新对话历史
        history.append((query, answer))
        
        # 更新对话标题（如果是第一条消息）
        if len(history) == 1:
            conversation["title"] = query[:30] + "..." if len(query) > 30 else query

        return answer, "\n".join(logs), history

    except Exception as e:
        logs.append(color_log(f"❌ 处理失败: {str(e)}", "error"))
        error_msg = "系统处理出错，请稍后再试"
        history.append((query, error_msg))
        return error_msg, "\n".join(logs), history


# 生成聊天历史HTML
def generate_chat_html(history):
    """生成聊天历史HTML，参考ChatGPT风格"""
    if not history:
        return """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 40px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 20px;">🏥</div>
            <h2 style="margin: 0 0 10px 0; font-size: 24px; color: #1a1a1a;">医疗智能体</h2>
            <p style="margin: 0; color: #666; font-size: 16px;">基于LLM和知识图谱的私人医生智能体</p>
            <div style="margin-top: 30px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; max-width: 600px; width: 100%;">
                <div style="background: #f7f7f8; padding: 15px; border-radius: 10px; cursor: pointer; transition: background 0.2s;" class="example-card">
                    <div style="font-weight: 500; color: #1a1a1a;">糖尿病的早期症状有哪些？</div>
                </div>
                <div style="background: #f7f7f8; padding: 15px; border-radius: 10px; cursor: pointer; transition: background 0.2s;" class="example-card">
                    <div style="font-weight: 500; color: #1a1a1a;">高血压患者应该吃什么药？</div>
                </div>
                <div style="background: #f7f7f8; padding: 15px; border-radius: 10px; cursor: pointer; transition: background 0.2s;" class="example-card">
                    <div style="font-weight: 500; color: #1a1a1a;">乳腺癌有哪些典型症状？</div>
                </div>
            </div>
        </div>
        """
    
    chat_html = '<div style="display: flex; flex-direction: column; gap: 20px; padding: 20px 0;">'
    
    for user_msg, assistant_msg in history:
        # 用户消息（右侧）
        chat_html += f'''
        <div style="display: flex; justify-content: flex-end; width: 100%;">
            <div style="display: flex; align-items: flex-start; gap: 16px; max-width: 90%;">
                <div style="display: flex; flex-direction: column; align-items: flex-end;">
                    <div style="background: #007aff; color: white; padding: 14px 18px; border-radius: 18px; border-top-right-radius: 4px; word-wrap: break-word; line-height: 1.6; font-size: 15px;">
                        {user_msg}
                    </div>
                </div>
                <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #007aff 0%, #5856d6 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; flex-shrink: 0; font-size: 14px;">
                    您
                </div>
            </div>
        </div>
        '''
        
        # 助手消息（左侧）
        chat_html += f'''
        <div style="display: flex; justify-content: flex-start; width: 100%;">
            <div style="display: flex; align-items: flex-start; gap: 16px; max-width: 90%;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #34c759 0%, #30d158 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; flex-shrink: 0; font-size: 18px;">
                    🏥
                </div>
                <div style="display: flex; flex-direction: column;">
                    <div style="background: #f7f7f8; color: #1a1a1a; padding: 14px 18px; border-radius: 18px; border-top-left-radius: 4px; word-wrap: break-word; line-height: 1.7; font-size: 15px;">
                        {assistant_msg}
                    </div>
                </div>
            </div>
        </div>
        '''
    
    chat_html += '</div>'
    return chat_html


# 生成对话列表HTML
def generate_conversations_list():
    """生成对话历史列表HTML"""
    global conversations
    
    if not conversations:
        return '<div style="padding: 20px; text-align: center; color: #8e8e93;">暂无对话历史</div>'
    
    # 按时间排序（最新的在前）
    sorted_conversations = sorted(
        conversations.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )
    
    list_html = '<div style="display: flex; flex-direction: column; gap: 4px;">'
    
    for conv in sorted_conversations:
        conv_id = conv["id"]
        is_active = conv_id == current_conversation_id
        bg_color = "#e5e5ea" if is_active else "transparent"
        text_color = "#000" if is_active else "#8e8e93"
        
        list_html += f'''
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 8px; background: {bg_color}; cursor: pointer; transition: background 0.2s;" 
             class="conversation-item" 
             data-conversation-id="{conv_id}"
             onmouseover="this.style.background='#e5e5ea'"
             onmouseout="if(!this.classList.contains('active')) this.style.background='transparent'">
            <span style="flex-shrink: 0;">💬</span>
            <span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: {text_color};">{conv["title"]}</span>
        </div>
        '''
    
    list_html += '</div>'
    return list_html


# 创建新对话
def create_new_conversation():
    """创建新对话"""
    global current_conversation_id, conversations
    
    conversation_id = str(uuid.uuid4())
    current_conversation_id = conversation_id
    
    conversations[conversation_id] = {
        "id": conversation_id,
        "title": "新对话",
        "history": [],
        "created_at": time.time()
    }
    
    return (
        "",  # 输入框清空
        "",  # 日志清空
        [],  # 历史清空
        generate_chat_html([]),  # 聊天区域
        generate_conversations_list()  # 对话列表
    )


# 切换对话
def switch_conversation(conversation_id):
    """切换到指定对话"""
    global current_conversation_id, conversations
    
    if conversation_id in conversations:
        current_conversation_id = conversation_id
        conv = conversations[conversation_id]
        return (
            "",  # 输入框
            "",  # 日志
            conv["history"],  # 历史
            generate_chat_html(conv["history"]),  # 聊天区域
            generate_conversations_list()  # 对话列表
        )
    
    return "", "", [], generate_chat_html([]), generate_conversations_list()


# ChatGPT风格的CSS
chatgpt_style_css = """
:root {
    --primary: #007aff;
    --primary-dark: #0051d5;
    --background: #ffffff;
    --surface: #f7f7f8;
    --surface-hover: #e5e5ea;
    --text-primary: #1a1a1a;
    --text-secondary: #8e8e93;
    --border: #e5e5ea;
    --success: #34c759;
    --warning: #ff9500;
    --error: #ff3b30;
    --sidebar-width: 260px;
}

/* 全局样式 */
* {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    margin: 0;
    padding: 0;
    background: var(--background);
    color: var(--text-primary);
}

/* 侧边栏样式 */
.sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.sidebar-header {
    padding: 16px;
    border-bottom: 1px solid var(--border);
}

.new-chat-btn {
    width: 100%;
    background: var(--primary);
    color: white;
    border: none;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.new-chat-btn:hover {
    background: var(--primary-dark);
}

.conversations-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}

.conversation-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
    font-size: 14px;
}

.conversation-item:hover {
    background: var(--surface-hover);
}

.conversation-item.active {
    background: var(--surface-hover);
}

/* 主聊天区域 */
.main-content {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: var(--background);
}

.chat-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.chat-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
}

.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 0 20px;
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.chat-messages {
    flex: 1;
    max-width: 1000px;
    margin: 0 auto;
    width: 100%;
    overflow-y: auto;
}

/* 输入区域 */
.input-container {
    flex: 3;
    padding: 20px;
    border-top: 1px solid var(--border);
    background: var(--background);
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.input-wrapper {
    max-width: 1000px;
    margin: 0 auto;
    position: relative;
}

.input-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 60px 20px 24px;
    min-height: 150px;
    max-height: 500px;
    height: 100%;
    resize: none;
    width: 100%;
    font-size: 16px;
    line-height: 1.7;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.input-box:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1);
}

.send-btn {
    position: absolute;
    right: 8px;
    bottom: 8px;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: var(--primary);
    color: white;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s, opacity 0.2s;
    opacity: 0.5;
}

.send-btn:hover:not(:disabled) {
    background: var(--primary-dark);
}

.send-btn:disabled {
    cursor: not-allowed;
}

.send-btn.active {
    opacity: 1;
}

/* 页脚 */
.footer {
    text-align: center;
    padding: 10px 20px 20px;
    font-size: 12px;
    color: var(--text-secondary);
}

/* 加载动画 */
.loading-dots {
    display: inline-flex;
    gap: 4px;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-secondary);
    animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) {
    animation-delay: -0.32s;
}

.loading-dots span:nth-child(2) {
    animation-delay: -0.16s;
}

@keyframes bounce {
    0%, 80%, 100% {
        transform: scale(0);
    }
    40% {
        transform: scale(1);
    }
}

/* 示例卡片悬停效果 */
.example-card:hover {
    background: #e5e5ea !important;
}

/* 滚动条样式 */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #c7c7cc;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #8e8e93;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .sidebar {
        display: none;
    }
    
    .chat-messages {
        max-width: 100%;
    }
    
    .input-wrapper {
        max-width: 100%;
    }
}
"""

# JavaScript代码 - 处理示例点击和输入框交互
js_code = """
function() {
    // 模拟用户输入的函数（确保Gradio能够检测到值变化）
    function simulateInput(element, text) {
        // 聚焦元素
        element.focus();
        
        // 设置值
        element.value = text;
        
        // 触发多种事件以确保Gradio检测到变化
        const inputEvent = new Event('input', { bubbles: true });
        element.dispatchEvent(inputEvent);
        
        const changeEvent = new Event('change', { bubbles: true });
        element.dispatchEvent(changeEvent);
        
        // 使用更底层的方式触发事件（Gradio 5.x 可能需要）
        try {
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            nativeInputValueSetter.call(element, text);
            
            const inputEvent2 = new Event('input', { bubbles: true });
            element.dispatchEvent(inputEvent2);
        } catch (e) {
            console.log('Native input setter not available');
        }
        
        return true;
    }
    
    // 查找输入框元素（Gradio的Textbox会渲染为textarea）
    function findInputElement() {
        // 方法1: 通过elem_id查找（最可靠）
        let inputElement = document.getElementById('user_input');
        
        // 方法2: 如果找不到，查找内部的textarea
        if (!inputElement) {
            inputElement = document.querySelector('#user_input textarea');
        }
        
        // 方法3: 通过类名查找
        if (!inputElement) {
            inputElement = document.querySelector('.input-box textarea');
        }
        
        // 方法4: 直接查找textarea
        if (!inputElement) {
            const elements = document.querySelectorAll('textarea');
            for (let el of elements) {
                if (el.placeholder && el.placeholder.includes('输入')) {
                    inputElement = el;
                    break;
                }
            }
        }
        
        return inputElement;
    }
    
    // 查找提交按钮
    function findSubmitButton() {
        // 方法1: 通过elem_id查找
        let submitBtn = document.getElementById('submit_btn');
        
        // 方法2: 通过类名查找
        if (!submitBtn) {
            submitBtn = document.querySelector('.send-btn');
        }
        
        // 方法3: 查找包含特定文本的按钮
        if (!submitBtn) {
            const buttons = document.querySelectorAll('button');
            for (let btn of buttons) {
                if (btn.textContent.trim() === '➤' || btn.textContent.includes('发送')) {
                    submitBtn = btn;
                    break;
                }
            }
        }
        
        return submitBtn;
    }
    
    // 使用事件委托处理示例卡片点击事件（支持动态渲染）
    document.addEventListener('click', function(e) {
        // 查找点击的元素是否是示例卡片或其子元素
        let targetElement = e.target;
        while (targetElement && !targetElement.classList.contains('example-card')) {
            targetElement = targetElement.parentElement;
        }
        
        if (targetElement && targetElement.classList.contains('example-card')) {
            const question = targetElement.textContent.trim();
            console.log('点击示例问题:', question);
            
            const inputElement = findInputElement();
            const submitBtn = findSubmitButton();
            
            if (inputElement) {
                console.log('找到输入框元素');
                
                // 模拟用户输入
                simulateInput(inputElement, question);
                
                // 自动点击发送按钮
                if (submitBtn) {
                    console.log('找到提交按钮，准备点击');
                    setTimeout(() => {
                        submitBtn.click();
                    }, 200);
                } else {
                    console.log('未找到提交按钮');
                }
            } else {
                console.log('未找到输入框元素');
            }
        }
    });
    
    // 输入框动态高度调整
    function setupInputBox() {
        const inputBox = findInputElement();
        const sendBtn = findSubmitButton();
        
        if (inputBox && sendBtn) {
            // 添加输入事件监听器
            inputBox.addEventListener('input', function() {
                // 调整输入框高度
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 200) + 'px';
                
                // 更新发送按钮状态
                if (this.value.trim()) {
                    sendBtn.classList.add('active');
                    sendBtn.disabled = false;
                } else {
                    sendBtn.classList.remove('active');
                }
            });
            
            // 初始检查
            if (inputBox.value.trim()) {
                sendBtn.classList.add('active');
            }
        }
    }
    
    // 初始设置输入框
    setTimeout(setupInputBox, 500);
    
    // 定期检查并重新设置输入框（处理Gradio重新渲染的情况）
    setInterval(setupInputBox, 2000);
    
    // Ctrl+Enter 提交
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            const submitBtn = findSubmitButton();
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.click();
            }
        }
    });
    
    // 使用事件委托处理对话列表点击事件
    document.addEventListener('click', function(e) {
        let targetElement = e.target;
        while (targetElement && !targetElement.classList.contains('conversation-item')) {
            targetElement = targetElement.parentElement;
        }
        
        if (targetElement && targetElement.classList.contains('conversation-item')) {
            const convId = targetElement.dataset.conversationId;
            // 这里需要通过Gradio的事件系统来处理
            // 由于Gradio的限制，我们需要使用其他方式
        }
    });
    
    // 返回值（Gradio要求）
    return 'JavaScript initialized successfully';
}
"""

# 创建优化后的聊天界面
with gr.Blocks(
        title="医疗智能体 - 基于LLM和知识图谱",
        css=chatgpt_style_css,
        js=js_code,
        theme=gr.themes.Base()
) as demo:
    
    # 状态变量
    chat_history_state = gr.State([])
    conversation_id_state = gr.State(current_conversation_id)
    is_loading = gr.State(False)
    
    # 主布局
    with gr.Row(equal_height=True, variant="compact"):
        
        # 左侧边栏
        with gr.Column(scale=1, min_width=260, elem_classes="sidebar"):
            # 侧边栏头部 - 新对话按钮
            with gr.Column(elem_classes="sidebar-header"):
                new_chat_btn = gr.Button(
                    "➕ 新对话",
                    variant="primary",
                    elem_classes="new-chat-btn"
                )
            
            # 对话历史列表
            with gr.Column(elem_classes="conversations-list"):
                conversations_list = gr.HTML(
                    value=generate_conversations_list(),
                    label="对话历史"
                )
        
        # 右侧主聊天区域
        with gr.Column(scale=4, elem_classes="main-content"):
            
            # 聊天头部
            with gr.Row(elem_classes="chat-header"):
                gr.Markdown("""
                <div class="chat-title">🏥 医疗智能体</div>
                """)
            
            # 聊天消息区域
            with gr.Column(elem_classes="chat-container"):
                chat_display = gr.HTML(
                    value=generate_chat_html([]),
                    elem_classes="chat-messages",
                    elem_id="chat_display"
                )
            
            # 输入区域
            with gr.Column(elem_classes="input-container"):
                with gr.Column(elem_classes="input-wrapper"):
                    input_query = gr.Textbox(
                        placeholder="输入您的医疗问题...",
                        lines=3,
                        elem_classes="input-box",
                        elem_id="user_input",
                        show_label=False,
                        container=False
                    )
                    submit_btn = gr.Button(
                        "➤",
                        variant="primary",
                        elem_classes="send-btn",
                        elem_id="submit_btn"
                    )
                
                # 处理日志区域（折叠）
                with gr.Accordion("🔧 处理日志", open=False):
                    process_logs = gr.HTML(
                        value="<div style='padding: 10px; font-family: monospace; font-size: 12px; color: #666;'>系统准备就绪，日志将显示在这里...</div>"
                    )
            
            # 页脚
            gr.Markdown("""
            <div class="footer">
                医疗智能体 v1.0 | 基于知识图谱与大语言模型 | 仅供信息参考，医疗问题请咨询专业医生
            </div>
            """)
    
    # 事件处理函数
    
    # 提交查询
    def handle_submit(query, history, conversation_id):
        """处理用户提交的查询"""
        global current_conversation_id
        
        if not query.strip():
            return query, "", history, generate_chat_html(history), generate_conversations_list()
        
        # 处理查询
        answer, logs, new_history = process_query(query, conversation_id)
        
        # 更新当前对话ID
        current_conversation_id = conversation_id
        
        # 生成新的聊天HTML
        chat_html = generate_chat_html(new_history)
        
        # 更新对话列表
        conv_list = generate_conversations_list()
        
        return "", logs, new_history, chat_html, conv_list
    
    # 新对话
    def handle_new_chat():
        """处理新对话"""
        return create_new_conversation()
    
    # 绑定事件
    
    # 新对话按钮
    new_chat_btn.click(
        fn=handle_new_chat,
        inputs=[],
        outputs=[input_query, process_logs, chat_history_state, chat_display, conversations_list]
    )
    
    # 提交按钮
    submit_btn.click(
        fn=handle_submit,
        inputs=[input_query, chat_history_state, conversation_id_state],
        outputs=[input_query, process_logs, chat_history_state, chat_display, conversations_list]
    )
    
    # 输入框回车提交
    input_query.submit(
        fn=handle_submit,
        inputs=[input_query, chat_history_state, conversation_id_state],
        outputs=[input_query, process_logs, chat_history_state, chat_display, conversations_list]
    )


# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )
