import yaml
from pathlib import Path

class Config:
    _cfg = None

    @classmethod
    def load(cls, path: str = "configs/debate.yaml"):
        if cls._cfg is None:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    cls._cfg = yaml.safe_load(f)
                    print(f"✅ 成功加載配置文件: {path}")
                    print(f"   主題: {cls._cfg.get('debate', {}).get('topic', 'N/A')}")
            except FileNotFoundError:
                print(f"⚠️  配置文件不存在: {path}，使用默認配置")
                # 如果配置文件不存在，使用默認配置
                cls._cfg = {
                    'debate': {
                        'rounds': 3,
                        'topic': 'Should we adopt universal basic income?'
                    }
                }
        return cls._cfg
    
    @classmethod
    def reset(cls):
        """重置配置緩存，強制重新加載"""
        cls._cfg = None