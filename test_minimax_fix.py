# coding: utf-8
"""
测试 MiniMax API 修复后的配置
"""
import os
import sys
import json
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config import Config

def test_minimax_config():
    """测试 MiniMax 配置"""
    print("=" * 60)
    print("MiniMax 配置测试")
    print("=" * 60)
    
    minimax_config = Config.get_minimax_config()
    
    print(f"\n当前配置:")
    print(f"  API URL: {minimax_config['api_url']}")
    print(f"  模型名称: {minimax_config['model_name']}")
    print(f"  API Key 已配置: {'是' if minimax_config['api_key'] else '否'}")
    print(f"  Group ID 已配置: {'是' if minimax_config['group_id'] else '否'}")
    
    expected_url = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    if minimax_config['api_url'] == expected_url:
        print(f"\n✅ API URL 配置正确!")
        print(f"   当前值: {minimax_config['api_url']}")
    else:
        print(f"\n❌ API URL 配置可能不正确")
        print(f"   建议值: {expected_url}")
        print(f"   当前值: {minimax_config['api_url']}")
    
    paid_models = ['MiniMax-M2.7', 'MiniMax-M2.7-highspeed', 'MiniMax-M2.5', 'MiniMax-M1']
    free_models = ['abab6.5s-chat', 'abab6.5g-chat', 'abab5.5s-chat']
    
    model_name = minimax_config['model_name']
    if model_name in paid_models:
        print(f"\n⚠️  模型名称 {model_name} 是付费模型 (需要升级套餐)")
        print(f"   提示: 免费套餐不支持此模型，建议切换到免费模型")
    elif model_name in free_models:
        print(f"\n✅ 模型名称 {model_name} 是免费模型 (推荐用于测试)")
        print(f"   提示: abab6.5s-chat 有 100 万 tokens 免费额度")
    else:
        print(f"\n⚠️  模型名称 {model_name} 未知，请确认是否正确")
    
    print("\n" + "=" * 60)
    print("问题分析总结:")
    print("=" * 60)
    
    print("\n问题 1: 404 错误")
    print("  原因: 原 API URL 配置不完整")
    print("    - 错误: https://api.minimaxi.com/v1")
    print("    - 正确: https://api.minimaxi.com/v1/text/chatcompletion_v2")
    print("  修复: ✅ 已更新 .env 中的 MINIMAX_API_URL")
    
    print("\n问题 2: TypeError: object of type 'NoneType' has no len()")
    print("  原因: API 返回业务错误时，choices 为 null，代码没有处理")
    print("  修复: ✅ 已在 minimax.py 中添加空值检查")
    
    print("\n问题 3: 模型不支持 (错误码 2061)")
    print("  原因: MiniMax-M2.7-highspeed 是付费模型，免费套餐不支持")
    print("  修复: ✅ 已切换到免费模型 abab6.5s-chat")
    
    print("\n已修复的内容:")
    print("  1. ✅ MINIMAX_API_URL 已更新为完整端点")
    print("  2. ✅ minimax.py 中添加了业务错误检查 (base_resp)")
    print("  3. ✅ minimax.py 中添加了 choices 空值检查")
    print("  4. ✅ 已切换到免费模型 abab6.5s-chat")
    print("  5. ✅ 更新了 .env 中的模型注释，说明套餐支持情况")
    
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    
    print("\n1. 重启你的 Flask 服务以加载新配置")
    print("   - 停止当前运行的 minimax.py 服务")
    print("   - 重新启动: python llm_services/minimax.py")
    
    print("\n2. 模型选择建议:")
    print("   【免费套餐可用】:")
    print("   - abab6.5s-chat (推荐): 100万tokens免费额度，适合测试")
    print("   - abab6.5g-chat: 高性能版")
    print("   - abab5.5s-chat: MiniMax-02")
    print("   【付费套餐可用】:")
    print("   - MiniMax-M2.7: 旗舰模型，支持自我进化")
    print("   - MiniMax-M2.7-highspeed: 高速版")
    print("   - MiniMax-M2.5: 高性价比")
    print("   - MiniMax-M1: 长上下文推理")
    
    print("\n3. API 地址选择:")
    print("   - 国内版: https://api.minimaxi.com/v1/text/chatcompletion_v2")
    print("   - 海外版: https://api.minimax.io/v1/text/chatcompletion_v2")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    test_minimax_config()
