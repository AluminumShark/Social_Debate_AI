"""
GPT 客戶端接口
提供與 OpenAI GPT 模型的交互功能
"""

import os
import openai

try:
    from ..utils.config_loader import Config
except ImportError:
    from utils.config_loader import Config

# 載入配置
cfg = Config.load()
openai.api_key = os.getenv('OPENAI_API_KEY')

def chat(prompt: str, model: str = None) -> str:
    """
    與 GPT 模型對話
    
    Args:
        prompt: 輸入提示文本
        model: 模型名稱（可選，預設使用配置中的模型）
        
    Returns:
        GPT 生成的回應文本
    """
    model = model or cfg.get('gpt', {}).get('model', 'gpt-4o')
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=800  # 增加到 800 以確保能生成 250 字的中文回應
        )
        return response.choices[0].message.content
    except Exception as e:
        # 如果 API 調用失敗，返回一個默認回應
        print(f"⚠️ GPT API 調用失敗: {e}")
        return f"[API Error] {str(e)}"


