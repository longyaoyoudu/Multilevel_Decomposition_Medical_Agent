from modelscope import snapshot_download

# 清除旧缓存并重新下载
model_dir = snapshot_download(
    "qwen/Qwen-7B-Chat",
    cache_dir="/home/Tony2016Edu/.cache/modelscope/hub",
    revision='master',  # 强制从最新代码分支下载
    ignore_file_pattern="*.bin"  # 避免重复下载大文件（如有）
)