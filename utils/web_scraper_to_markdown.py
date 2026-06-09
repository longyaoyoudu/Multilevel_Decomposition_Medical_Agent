import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 定义要爬取的网页列表
urls = {
    "文档中心": "https://www.modelscope.cn/docs/overview",
    "研习社": "https://modelscope.cn/learn",
    "GitHub": "https://github.com/modelscope",
    "模型库": "https://modelscope.cn/models",
    "数据集": "https://modelscope.cn/datasets",
    "创空间应用": "https://modelscope.cn/studios",
    "MCP": "https://www.modelscope.cn/mcp",
    "AIGC": "https://www.modelscope.cn/aigc",
}

# 创建保存 Markdown 文件和图片的目录
output_dir = "web_content_markdown"
os.makedirs(output_dir, exist_ok=True)

def download_image(img_url, save_dir):
    """下载图片并保存到指定目录"""
    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()
        filename = os.path.basename(img_url.split("?")[0])  # 去掉 URL 参数
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return filename
    except Exception as e:
        print(f"下载图片失败: {img_url}, 错误: {e}")
        return None

def fetch_and_save(url, name):
    """爬取网页内容并保存为 Markdown 文件，包含图片"""
    try:
        # 发送 HTTP 请求
        response = requests.get(url)
        response.raise_for_status()

        # 解析 HTML 内容
        soup = BeautifulSoup(response.text, 'html.parser')

        # 创建网页专属目录
        page_dir = os.path.join(output_dir, name)
        os.makedirs(page_dir, exist_ok=True)

        # 提取标题和主要内容
        title = soup.title.string if soup.title else name
        content = []

        # 提取所有段落、标题和图片
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'img']):
            if tag.name in ['h1', 'h2', 'h3']:
                content.append(f"# {tag.text.strip()}")
            elif tag.name == 'p':
                content.append(tag.text.strip())
            elif tag.name == 'img' and tag.get('src'):
                img_url = urljoin(url, tag['src'])  # 处理相对路径
                img_filename = download_image(img_url, page_dir)
                if img_filename:
                    content.append(f"![图片]({img_filename})")

        # 提取所有链接
        links = [a['href'] for a in soup.find_all('a', href=True)]
        if links:
            content.append("\n## 链接列表")
            content.extend([f"- {link}" for link in links])

        # 保存为 Markdown 文件
        markdown_content = f"# {title}\n\n" + "\n\n".join(content)
        file_path = os.path.join(page_dir, f"{name}.md")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"成功保存: {file_path}")

    except Exception as e:
        print(f"爬取 {name} 时出错: {e}")

# 遍历所有 URL 并爬取内容
for name, url in urls.items():
    fetch_and_save(url, name)

print("所有网页内容已爬取并保存为 Markdown 文件，包含图片。")