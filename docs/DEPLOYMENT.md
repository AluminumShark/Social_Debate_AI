# 🚀 Social Debate AI 部署指南

本指南說明如何將 Social Debate AI 部署到生產環境。

## 📋 部署選項

### 1. 本地部署
- 適合個人使用和測試
- 最簡單的部署方式

### 2. 雲端部署
- AWS EC2
- Google Cloud Platform
- Azure
- Heroku

### 3. Docker 部署
- 容器化部署
- 易於擴展

## 🏠 本地生產部署

### 1. 環境準備

```bash
# 創建生產環境
python -m venv venv_prod
source venv_prod/bin/activate  # Linux/Mac
# 或
venv_prod\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
pip install gunicorn  # 生產服務器
```

### 2. 環境變數配置

創建 `.env.production` 文件：

```bash
# API Keys
OPENAI_API_KEY=your-production-key

# Flask 配置
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key

# 安全設置
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax
```

### 3. 使用 Gunicorn 運行

```bash
# 基本運行
gunicorn -w 4 -b 0.0.0.0:5000 ui.app:app

# 推薦配置
gunicorn \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  ui.app:app
```

### 4. 使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支援（未來功能）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # 靜態文件
    location /static {
        alias /path/to/Social_Debate_AI/ui/static;
        expires 1d;
    }
}
```

## ☁️ AWS EC2 部署

### 1. 創建 EC2 實例

```bash
# 推薦配置
- 實例類型: t3.medium (最低) 或 t3.large (推薦)
- 作業系統: Ubuntu 20.04 LTS
- 存儲: 20GB SSD
- 安全組: 開放 80, 443, 22 端口
```

### 2. 初始化服務器

```bash
# 連接到 EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# 更新系統
sudo apt update && sudo apt upgrade -y

# 安裝必要軟體
sudo apt install python3.8 python3-pip nginx git -y
```

### 3. 部署應用

```bash
# 克隆專案
git clone https://github.com/your-username/Social_Debate_AI.git
cd Social_Debate_AI

# 設置虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
pip install gunicorn

# 設置環境變數
cp .env.example .env
nano .env  # 編輯配置
```

### 4. 設置 Systemd 服務

創建 `/etc/systemd/system/social-debate.service`:

```ini
[Unit]
Description=Social Debate AI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Social_Debate_AI
Environment="PATH=/home/ubuntu/Social_Debate_AI/venv/bin"
ExecStart=/home/ubuntu/Social_Debate_AI/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 ui.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

啟動服務：

```bash
sudo systemctl start social-debate
sudo systemctl enable social-debate
sudo systemctl status social-debate
```

## 🐳 Docker 部署

### 1. 創建 Dockerfile

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# 複製應用代碼
COPY . .

# 暴露端口
EXPOSE 5000

# 運行命令
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "ui.app:app"]
```

### 2. 創建 docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped
```

### 3. 構建和運行

```bash
# 構建映像
docker-compose build

# 運行容器
docker-compose up -d

# 查看日誌
docker-compose logs -f
```

## 🔒 安全配置

### 1. HTTPS 配置

使用 Let's Encrypt 免費證書：

```bash
# 安裝 Certbot
sudo apt install certbot python3-certbot-nginx

# 獲取證書
sudo certbot --nginx -d your-domain.com

# 自動更新
sudo certbot renew --dry-run
```

### 2. 防火牆設置

```bash
# UFW 防火牆
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 3. 環境變數安全

```bash
# 使用 python-dotenv
from dotenv import load_dotenv
load_dotenv('.env.production')

# 敏感信息加密
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)
```

## 📊 監控和日誌

### 1. 應用日誌

```python
# 在 app.py 中配置日誌
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/social-debate.log',
        maxBytes=10240000,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    app.logger.addHandler(file_handler)
```

### 2. 系統監控

```bash
# 使用 htop 監控資源
sudo apt install htop
htop

# 使用 netdata 進行全面監控
bash <(curl -Ss https://my-netdata.io/kickstart.sh)
```

### 3. 錯誤追蹤

考慮整合 Sentry：

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)
```

## 🚦 性能優化

### 1. 緩存配置

```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})

@cache.cached(timeout=300)
def expensive_function():
    pass
```

### 2. 資料庫優化

如果使用資料庫存儲辯論記錄：

```python
# 使用連接池
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    pool_recycle=3600
)
```

### 3. CDN 配置

使用 Cloudflare 或其他 CDN 服務加速靜態資源。

## 📈 擴展策略

### 1. 水平擴展

```nginx
# Nginx 負載均衡
upstream app_servers {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    location / {
        proxy_pass http://app_servers;
    }
}
```

### 2. 使用 Redis 共享狀態

```python
from redis import Redis
redis_client = Redis(host='localhost', port=6379, db=0)

# 存儲辯論狀態
redis_client.set(f'debate:{debate_id}', json.dumps(debate_state))
```

## 🔧 故障排除

### 常見問題

1. **記憶體不足**
   - 增加交換空間
   - 優化模型載入
   - 使用更大的實例

2. **響應超時**
   - 增加 Gunicorn timeout
   - 優化模型推理
   - 使用異步處理

3. **並發問題**
   - 使用 Redis 鎖
   - 實現請求排隊
   - 限制並發數

## 📝 部署檢查清單

- [ ] 環境變數配置完整
- [ ] HTTPS 證書已安裝
- [ ] 防火牆規則已設置
- [ ] 日誌系統正常運作
- [ ] 備份策略已實施
- [ ] 監控系統已啟用
- [ ] 錯誤追蹤已配置
- [ ] 性能測試已完成

---

💡 **提示**：部署前請先在測試環境驗證所有配置！ 