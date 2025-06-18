import openai, os
try:
    from ..utils.config_loader import Config
except ImportError:
    from utils.config_loader import Config


cfg = Config.load()
openai.api_key = os.getenv('OPENAI_API_KEY')

def chat(prompt: str, model=None) -> str:
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
        return f"[API Error] {str(e)}"


