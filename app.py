"""
绿洲大模型 - 后端智能体服务（带会话管理）
========================================
用户发消息 → DeepSeek 推理 → 需要时调 PCSE 模拟 → 整合回答
支持多会话、历史记录（SQLite 本地 / PostgreSQL 云端双后端）
"""
import os, json, time, re, asyncio, traceback
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import httpx

# ========== 数据库引擎（SQLite 本地 / PostgreSQL 云端）==========
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Text, ForeignKey,
    select, insert, update, delete,
)
from sqlalchemy.pool import NullPool
import bcrypt
import jwt as pyjwt

# ========== 启动预热：连带唤醒机理模型 & 向量化服务 ==========
async def prewarm_oasis_api():
    """oasis-chat 启动时自动唤醒下游服务（并行+短超时，不阻塞启动）"""
    async def ping(url: str, name: str, headers=None):
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(url, headers=headers)
                print(f"[预热] {name} 已唤醒 (HTTP {r.status_code})")
        except Exception as e:
            print(f"[预热] {name} 唤醒失败: {e}")
    hdrs = {"X-API-Key": OASIS_API_KEY} if OASIS_API_KEY else {}
    await asyncio.gather(
        ping(f"{PCSE_API_URL}/", "机理模型服务", hdrs),
        ping(f"{EMBED_API_URL}/", "向量化服务"),
    )

# ========== 周期保活：每10分钟 ping 下游服务 ==========
async def keepalive_loop():
    """只要 oasis-chat 进程活着，每10分钟唤醒一次下游，保证通路"""
    await asyncio.sleep(60)  # 启动1分钟后开始
    while True:
        await asyncio.sleep(600)  # 10分钟
        try:
            await prewarm_oasis_api()
        except Exception as e:
            print(f"[保活] 下游唤醒失败: {e}")
        try:
            # ★ 每10分钟真读一次数据库：防 Supabase 免费版因长时间无活动被暂停
            with engine.connect() as conn:
                conn.execute(_sql_text("SELECT 1"))
        except Exception as e:
            print(f"[保活] 数据库心跳失败: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 后台预热下游服务（不阻塞启动，页面秒开）
    asyncio.create_task(prewarm_oasis_api())
    keepalive_task = asyncio.create_task(keepalive_loop())
    yield
    keepalive_task.cancel()

app = FastAPI(title="绿洲大模型 - 智能体后端", version="2.1.0", lifespan=lifespan)

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
PCSE_API_URL = os.environ.get("PCSE_API_URL", "https://oasis-api-phlm.onrender.com")
OASIS_API_KEY = os.environ.get("OASIS_API_KEY", "")  # 调用机理API的密钥
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "chat.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")
JWT_SECRET = os.environ.get("JWT_SECRET", "")
if not JWT_SECRET:
    import secrets
    JWT_SECRET = secrets.token_hex(32)
    print("[警告] 未设置 JWT_SECRET 环境变量，已生成随机密钥（重启后token将失效，生产请配置固定值）")
JWT_EXPIRE_DAYS = int(os.environ.get("JWT_EXPIRE_DAYS", "7"))

# 静态文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# ========== 数据库初始化 ==========
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, poolclass=NullPool, pool_pre_ping=True)
    print(f"[启动] 使用云端 PostgreSQL 数据库")
else:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    print(f"[启动] 使用本地 SQLite 数据库: {DB_PATH}")

metadata = MetaData()

conversations = Table(
    "conversations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False, default="anonymous"),
    Column("title", String(200), nullable=False, default="新对话"),
    Column("created_at", String(32), nullable=False),
)

messages = Table(
    "messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("conversation_id", Integer, ForeignKey("conversations.id"), nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("cards", Text, nullable=True),
    Column("created_at", String(32), nullable=False),
)

# 用户表（预留手机号字段，供以后短信登录接入）
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(50), nullable=False, unique=True),
    Column("password_hash", String(200), nullable=False),
    Column("phone", String(20), nullable=True),          # 预留：手机号
    Column("phone_verified", String(1), nullable=False, default="0"),  # 预留：手机号验证状态
    Column("created_at", String(32), nullable=False),
)

metadata.create_all(engine)
# 知识库表（documents/chunks）幂等建表：PostgreSQL 用 pgvector，SQLite 本地用 TEXT 占位。
# 修复：此前这两个表不在 metadata 中，本地 SQLite 模式 /api/knowledge 直接 500。
from sqlalchemy import text as _sql_text
if DATABASE_URL:
    # ★ 逐条执行 + 20 秒语句超时 + 单条失败不影响启动：
    #   建索引要拿表锁，若被别的长事务挡住，原来会无限期卡住整个服务启动（2026-10-12 实测踩到）
    _pg_ddl = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'upload',
            created_at TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content TEXT NOT NULL, embedding vector(512), created_at TEXT NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)",
    ]
    for _stmt in _pg_ddl:
        try:
            with engine.begin() as conn:
                conn.execute(_sql_text("set statement_timeout = '20000'"))
                conn.execute(_sql_text(_stmt))
        except Exception as _e:
            print(f"[启动] DDL 跳过（不影响启动）: {type(_e).__name__}: {str(_e)[:110]}")
else:
    with engine.begin() as conn:
        conn.execute(_sql_text("""CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'upload',
            created_at TEXT NOT NULL)"""))
        conn.execute(_sql_text("""CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id),
            content TEXT NOT NULL, embedding TEXT, created_at TEXT NOT NULL)"""))
print("[启动] 数据表就绪")

# ========== JWT / 密码工具 ==========
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False

def create_token(user_id: int, username: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")

def decode_token(token: str):
    """解析 token，失败返回 None"""
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except Exception:
        return None

def get_token_user(request):
    """从请求 Authorization 头解析用户，返回 (user_id, username) 或 None"""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    payload = decode_token(auth[7:])
    if not payload:
        return None
    try:
        return int(payload["sub"]), payload.get("username", "")
    except Exception:
        return None

# ========== 登录限流（内存）==========
# key: "user:username" 或 "ip:x.x.x.x", value: [fail_count, lock_until_ts]
_login_attempts = {}
LOGIN_MAX_FAILS = 5
LOGIN_LOCK_MINUTES = 10

def check_login_limited(request, username: str):
    """检查是否被锁定；未锁定则返回 None，锁定返回剩余秒数"""
    now = time.time()
    ip = request.client.host if request.client else "unknown"
    keys = [f"user:{username}", f"ip:{ip}"]
    for k in keys:
        rec = _login_attempts.get(k)
        if rec and rec[1] > now:
            return int(rec[1] - now)
    return None

def record_login_fail(request, username: str):
    """记录一次失败"""
    now = time.time()
    ip = request.client.host if request.client else "unknown"
    for k in [f"user:{username}", f"ip:{ip}"]:
        rec = _login_attempts.get(k, [0, 0])
        rec[0] += 1
        if rec[0] >= LOGIN_MAX_FAILS:
            rec[1] = now + LOGIN_LOCK_MINUTES * 60
            rec[0] = 0
        _login_attempts[k] = rec

def record_login_success(username: str):
    """登录成功后清除该用户名/IP的计数"""
    for k in list(_login_attempts.keys()):
        if k == f"user:{username}" or k.startswith("ip:"):
            _login_attempts.pop(k, None)

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ========== RAG 知识库核心 ==========
COMPILE_PROMPT = """你是一位资深农业知识编辑。请将以下资料整理成一份结构化的精编知识页：
1. 提取全部核心知识点，删除重复和冗余内容
2. 按主题组织成清晰的章节（如选地与整地、播种技术、水肥管理、病虫害防治、收获与储藏）
3. 保留所有关键数据（数字、剂量、时间、标准）
4. 语言精炼、专业、准确
5. 用 Markdown 格式输出，使用 ## 分章节

只输出整理后的知识页内容，不要添加任何前后说明。

资料如下：
{text}"""

async def _compile_once(text: str) -> str:
    """单段编译（≤15000字符）"""
    llm_msgs = [
        {"role": "system", "content": "你是绿洲大模型的资深农业知识编辑，负责把原始资料编译成高质量知识页。"},
        {"role": "user", "content": COMPILE_PROMPT.format(text=text)},
    ]
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                json={"model": DEEPSEEK_MODEL, "messages": llm_msgs, "temperature": 0.2, "max_tokens": 4000},
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                return text
            compiled = resp.json()["choices"][0]["message"]["content"].strip()
            return compiled if len(compiled) > 50 else text
    except Exception as e:
        print(f"[编译] 失败，回退原文: {e}")
        return text

async def compile_document(text: str) -> str:
    """用 DeepSeek 把原始文档编译成精编知识页（大文档分段编译，失败时回退原文）"""
    if not DEEPSEEK_API_KEY:
        return text
    max_chars = 15000
    if len(text) <= max_chars:
        return await _compile_once(text)
    # 大文档分段编译再拼接
    parts = []
    for i in range(0, len(text), max_chars - 800):
        seg = text[i:i + max_chars]
        parts.append(await _compile_once(seg))
    return "\n\n---\n\n".join(parts)

async def embed_texts(texts: list) -> list:
    """调用向量化服务，返回向量列表（带冷启动重试 + 缓存）"""
    if not texts:
        return []
    # 问题向量缓存（相同文本秒回，避免重复 embed）
    uncached = []
    for t in texts:
        if t not in _EMBED_CACHE:
            uncached.append(t)
    if uncached:
        last_err = None
        for attempt in range(3):  # 重试压缩：2+4+6=12s（原4次50s，聊天路径卡太久）
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{EMBED_API_URL}/embed",
                        json={"texts": uncached},
                    )
                    if resp.status_code != 200:
                        last_err = f"HTTP {resp.status_code}"
                    else:
                        vs = resp.json()["vectors"]
                        for t, v in zip(uncached, vs):
                            _EMBED_CACHE[t] = v
                        # 限制缓存大小（最多500条）
                        if len(_EMBED_CACHE) > 500:
                            for k in list(_EMBED_CACHE)[:100]:
                                _EMBED_CACHE.pop(k, None)
                        # 按输入顺序返回（修复：原 cached+vs 拼接会打乱顺序 → 向量错配）
                        return [_EMBED_CACHE[t] for t in texts]
            except Exception as e:
                last_err = str(e)
            await asyncio.sleep(2 * (attempt + 1))
        raise Exception(f"向量化失败: {last_err}")
    # 全部命中缓存：按输入顺序返回
    return [_EMBED_CACHE[t] for t in texts]

# 问题向量缓存（dict，最近500条）
_EMBED_CACHE = {}

def chunk_text(text: str, max_len: int = 500, overlap: int = 50) -> list:
    """文本切块：按段落优先，超长再按长度切，带重叠"""
    text = text.strip()
    if not text:
        return []
    # 按段落分
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= max_len:
            chunks.append(p)
        else:
            # 长段落按句子/长度切
            for i in range(0, len(p), max_len - overlap):
                chunks.append(p[i:i + max_len])
    return chunks

def parse_document(filename: str, data: bytes) -> str:
    """解析文档为文本（支持 PDF/Word/TXT/MD）"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".txt", ".md", ".csv"):
        for enc in ("utf-8", "gbk", "gb2312"):
            try:
                return data.decode(enc)
            except:
                continue
        return data.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(400, "服务端缺少PDF解析库")
        except Exception as e:
            raise HTTPException(400, f"PDF解析失败: {e}")
    elif ext in (".docx", ".doc"):
        try:
            import io
            from docx import Document
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise HTTPException(400, "服务端缺少Word解析库")
        except Exception as e:
            raise HTTPException(400, f"Word解析失败: {e}")
    elif ext in (".xlsx", ".xls"):
        try:
            import io
            from openpyxl import load_workbook
            wb = load_workbook(io.BytesIO(data), read_only=True)
            lines = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    vals = [str(v) for v in row if v is not None]
                    if vals:
                        lines.append(", ".join(vals))
            return "\n".join(lines)
        except ImportError:
            raise HTTPException(400, "服务端缺少Excel解析库")
        except Exception as e:
            raise HTTPException(400, f"Excel解析失败: {e}")
    else:
        raise HTTPException(400, f"不支持的文件类型: {ext}（支持 txt/md/csv/pdf/docx/xlsx）")

async def search_knowledge(query: str, top_k: int = None) -> list:
    """检索知识库：问题向量化 → 余弦相似度 → 返回 top_k 段落"""
    if not RAG_ENABLED:
        return []
    try:
        top_k = top_k or RAG_TOP_K
        q_vec = (await embed_texts([query]))[0]
        # pgvector 余弦相似度检索
        from sqlalchemy import text
        vec_str = "[" + ",".join(str(x) for x in q_vec) + "]"
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT c.id, c.content, d.title, 1 - (c.embedding <=> '{vec_str}'::vector) AS similarity
                    FROM chunks c JOIN documents d ON c.document_id = d.id
                    ORDER BY c.embedding <=> '{vec_str}'::vector
                    LIMIT {int(top_k)}
                """)
            ).fetchall()
        results = []
        for r in rows:
            sim = round(float(r.similarity), 4) if r.similarity is not None else 0
            if sim < 0.45:
                continue  # 低相似度视为噪音，过滤
            results.append({
                "content": r.content,
                "title": r.title,
                "similarity": sim,
            })
        return results
    except Exception as e:
        print(f"[RAG] 检索失败: {e}")
        return []

# ========== 数据模型 ==========
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    reply: str
    cards: Optional[list] = None
    text: Optional[str] = None  # 语音接口返回转写文字

# ========== 作物/地点识别 ==========
CROPS = {
    1: "冬小麦", 2: "玉米", 3: "春大麦", 5: "水稻",
    6: "甜菜", 7: "马铃薯", 8: "蚕豆",
    9: "大豆", 10: "冬油菜", 11: "向日葵", 12: "饲料玉米",
}
LOCATIONS = {
    "莎车": (38.48, 77.26), "喀什": (39.47, 75.99),
    "阿克苏": (41.17, 80.25), "石河子": (44.30, 86.03),
    "伊犁": (43.92, 81.32), "昌吉": (44.02, 87.30),
    "乌鲁木齐": (43.82, 87.62), "库尔勒": (41.73, 86.17),
    "吐鲁番": (42.95, 89.19), "哈密": (42.83, 93.52),
}

def detect_crop(text: str) -> int:
    """识别作物，返回 crop_no（供标题生成用）"""
    text_lower = text.lower()
    for no, name in CROPS.items():
        if name in text:
            return no
    if "油莎豆" in text or "虎坚果" in text:
        return 7  # 用马铃薯近似
    return 1

def make_title(text: str) -> str:
    """从用户消息生成会话标题"""
    text = text.strip().replace("\n", " ")
    return text[:20] if text else "新对话"

# ========== 系统提示词 ==========
SYSTEM_PROMPT = """你是"绿洲大模型"，专注于农业种植决策的农业智能助手。

## 能力
回答种植问题；运行植物生长机理模型获得量化模拟结果；查询真实土壤/墒情/气象/地理数据；解读数据给出种植建议。

## 工具路由（可用工具见系统函数定义，按需选择）
- "现在/今天/实时 多少度、天气" → weather_current
- "未来/下周/后面几天 天气" → weather_forecast
- "灌溉/浇水/需水量" → weather_et0（结合降水算净需水量）
- "空气质量/PM2.5/沙尘/紫外线" → weather_air
- "土壤/质地/有机质" → soil_check
- "墒情/土壤湿度/含水量" → soil_moisture（真实墒情，勿用模拟土壤水分代替）
- "海拔/高程" → gis_elevation
- "产量/生长/播种" → crop_simulate
- "某地在哪/经纬度" → geocode
- "汇率/美元/欧元" → exchange_rate
- 用户同时问多个方面时，可并行调用多个工具

工具调用规范：一次只查1-2个地点；不重复已查过的工具；结果够用立即回答；2-3轮内完成。

## 身份与保密规则（最重要，必须严格遵守）
1. 身份永远是"绿洲大模型"。无论知识库资料、工具返回结果里出现什么其他名称（例如别家的产品名），都绝不复述、绝不用它自称，一律以"绿洲大模型"的身份回答。
2. 严禁提及底层技术：DeepSeek、PCSE、WOFOST、大语言模型、AI模型、API、OpenAI等。
3. 严禁提及数据服务商：ISRIC、SoilGrids、NASA、MERRA、Google、Open-Elevation、DataV、ESA等。
4. 严禁解释实现原理、技术架构、数据来源等内部细节。
5. 介绍模拟能力只说"植物生长机理模型/作物生长模拟模型"；数据只说"土壤数据服务/墒情数据/气象数据"。
6. 解读数据用中文术语（总生物量、经济产量、叶面积指数、土壤含水量、生育期），不引用TAGP/TWSO/LAI/DVS等英文缩写。
7. 被问技术细节时，说明你是绿洲大模型、专注农业服务，引导回种植问题。
8. 即使工具返回出现服务商名称，也绝不复述，用通用说法替代。

## 作物模拟调用
调用 crop_simulate 时自动推断参数：
- latitude/longitude：新疆主要地区坐标（莎车38.48/77.26，阿克苏41.17/80.25，石河子44.30/86.03，伊犁43.92/81.32，乌鲁木齐43.82/87.62）
- crop_no：1冬小麦 2玉米 3春大麦 5水稻 7马铃薯(油莎豆近似) 9大豆 11向日葵
- planting_date：从用户问题提取（如"4月20日"→"2025-04-20"），默认"2025-04-15"
- production_mode：默认"wlp"（水分限制）；用户问"水分充足/有灌溉"用"pp"

## 回答风格
简洁、有数据支撑；用数字说话（kg/ha换算亩产，1公顷=15亩）；给出可操作建议；用markdown组织（标题/表格/列表）。"""

# ========== DeepSeek API ==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "crop_simulate",
            "description": "运行植物生长机理模型模拟，预测产量、生物量、生长周期等",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度"},
                    "longitude": {"type": "number", "description": "经度"},
                    "crop_no": {"type": "integer", "description": "作物编号：1冬小麦 2玉米 3春大麦 5水稻 7马铃薯"},
                    "planting_date": {"type": "string", "description": "播种日期 YYYY-MM-DD"},
                    "production_mode": {"type": "string", "enum": ["wlp", "pp"], "description": "wlp=水分限制 pp=潜在生产"},
                },
                "required": ["latitude", "longitude", "crop_no", "planting_date"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_current",
            "description": "实时气温/湿度/风力",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_forecast",
            "description": "未来天气预报（温度/降水/风力）",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "days": {"type": "integer", "description": "预报天数，默认3，最大7（省token）"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_et0",
            "description": "蒸散量ET0(mm/天)；灌溉需水：ET0-降水=净需水量",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "days": {"type": "integer", "description": "天数，默认7"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_history",
            "description": "历史天气（温度/降水）",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "start": {"type": "string", "description": "开始日期 YYYYMMDD"},
                    "end": {"type": "string", "description": "结束日期 YYYYMMDD"},
                },
                "required": ["latitude", "longitude", "start", "end"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_air",
            "description": "实时空气质量与紫外线",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "地名转经纬度",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "地名，如：喀什、莎车"},
                },
                "required": ["name"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "exchange_rate",
            "description": "人民币兑外币实时汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency": {"type": "string", "description": "货币代码，如USD、EUR，默认CNY"},
                },
                "required": ["currency"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "soil_check",
            "description": "土壤属性（质地/砂黏比例/有机碳）",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "soil_moisture",
            "description": "土壤墒情时间序列（表层/根区含水量）",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "start": {"type": "string", "description": "开始日期 YYYYMMDD，可选"},
                    "end": {"type": "string", "description": "结束日期 YYYYMMDD，可选"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "gis_elevation",
            "description": "查询某位置的海拔高程",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    },
]

async def call_deepseek(messages: list) -> dict:
    async with httpx.AsyncClient(timeout=90) as client:
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        resp = await client.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            json=payload, headers=headers,
        )
        if resp.status_code != 200:
            raise Exception(f"模型服务暂时不可用: {resp.status_code}")
        data = resp.json()
        # 记录用量（内存统计，不落库）
        usage = data.get("usage", {})
        record_usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        return data

async def call_deepseek_stream(messages: list):
    """DeepSeek 流式调用（最终回答轮，逐段 yield 文本增量）
    tool_choice=none 强制纯文本回答，避免模型误输出工具调用文本"""
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "stream": True,
        "tools": TOOLS,
        "tool_choice": "none",
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream(
            "POST", f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            json=payload, headers=headers,
        ) as resp:
            if resp.status_code != 200:
                raise Exception(f"模型服务暂时不可用: {resp.status_code}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    if obj.get("usage") and not obj.get("choices"):
                        # 流式结束前的用量汇总（修复：此前流式回答不计入用量面板）
                        u = obj["usage"]
                        record_usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                        continue
                    delta = obj["choices"][0].get("delta", {})
                except:
                    continue
                content = delta.get("content")
                if content:
                    yield content

# ========== DeepSeek 用量统计（内存）==========
usage_stats = {"day": "", "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

def record_usage(prompt_tokens: int, completion_tokens: int):
    today = datetime.now().strftime("%Y-%m-%d")
    if usage_stats["day"] != today:
        usage_stats["day"] = today
        usage_stats["calls"] = 0
        usage_stats["prompt_tokens"] = 0
        usage_stats["completion_tokens"] = 0
    usage_stats["calls"] += 1
    usage_stats["prompt_tokens"] += prompt_tokens
    usage_stats["completion_tokens"] += completion_tokens

@app.get("/api/stats")
def stats(request: Request):
    """今日模型用量统计——仅管理员可查看（修复：此前匿名可看）"""
    user = get_token_user(request)
    if not user:
        raise HTTPException(401, "请先登录")
    if user[1] != ADMIN_USERNAME:
        raise HTTPException(403, "仅管理员可查看用量")
    return {
        "date": usage_stats["day"],
        "calls": usage_stats["calls"],
        "prompt_tokens": usage_stats["prompt_tokens"],
        "completion_tokens": usage_stats["completion_tokens"],
        "total_tokens": usage_stats["prompt_tokens"] + usage_stats["completion_tokens"],
    }

# 清洗返回给模型的内容，防止暴露底层技术名称
def sanitize_text(text: str) -> str:
    if not text:
        return text
    # 清理模型误输出的工具调用文本标签（兜底）
    # 处理未闭合标签（回答可能被截断）
    text = re.sub(r'<(?:DSML)?invoke\b[^>]*>.*?(</(?:DSML)?invoke>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'<(?:DSML)?parameter\b[^>]*>.*?(</(?:DSML)?parameter>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'<(?:DSML)?tool_calls>\s*</(?:DSML)?tool_calls>', '', text, flags=re.DOTALL)
    text = re.sub(r'<(?:DSML)?tool_calls>', '', text)
    text = re.sub(r'</(?:DSML)?tool_calls>', '', text)
    text = re.sub(r'</(?:DSML)?invoke>', '', text)
    text = re.sub(r'</(?:DSML)?parameter>', '', text)
    text = re.sub(r'<tool\b[^>]*>.*?(</tool>|$)', '', text, flags=re.DOTALL)
    # 清理 DSML/工具标签残片（含被截断的结束标签如 </DSML）
    text = re.sub(r'</?DSML[a-zA-Z_]*>?', '', text)
    text = re.sub(r'</?invoke[a-zA-Z_]*>?', '', text)
    text = re.sub(r'</?parameter[a-zA-Z_]*>?', '', text)
    text = re.sub(r'</?tool_calls>?', '', text)
    # 清理全角竖线伪标签（模型偶发输出 ｜｜invoke name="..."｜｜parameter... 格式）
    text = re.sub(r'｜｜[a-zA-Z_]+[^｜]*?｜｜/[a-zA-Z_]+｜｜', '', text, flags=re.DOTALL)
    text = re.sub(r'｜｜[a-zA-Z_]+[^｜]*?$', '', text, flags=re.DOTALL)  # 未闭合的到结尾
    text = text.replace('｜｜', '')
    # 压缩多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace("WOFOST", "植物生长机理模型")
    text = text.replace("wofost", "植物生长机理模型")
    text = text.replace("PCSE", "植物生长机理模型")
    text = text.replace("pcse", "植物生长机理模型")
    text = text.replace("DeepSeek", "绿洲大模型")
    text = text.replace("deepseek", "绿洲大模型")
    text = text.replace("OpenAI", "绿洲大模型")
    text = text.replace("GPT", "大模型")
    text = text.replace("Claude", "绿洲大模型")
    text = text.replace("Anthropic", "绿洲大模型")
    text = text.replace("Gemini", "绿洲大模型")
    text = text.replace("Llama", "大模型")
    # ★ 品牌兜底（2026-10-12）：知识库里夹带的其他名称一律归一到绿洲大模型
    text = text.replace("万象耕耘大模型", "绿洲大模型")
    text = text.replace("万象耕耘", "绿洲大模型")
    text = text.replace("绿洲驼铃农业大模型", "绿洲大模型")
    text = re.sub(r"我是(?!绿洲)[^\s，。、！？]{2,10}?大模型", "我是绿洲大模型", text)
    text = text.replace("大语言模型", "智能模型")
    text = text.replace("语言模型", "智能模型")
    text = text.replace("LLM", "智能模型")
    text = text.replace("ISRIC SoilGrids", "全球土壤数据库")
    text = text.replace("SoilGrids", "土壤数据库")
    text = text.replace("ISRIC", "土壤数据库")
    text = text.replace("NASA POWER", "气象与墒情数据服务")
    text = text.replace("NASA", "气象数据服务")
    text = text.replace("MERRA-2", "再分析数据")
    text = text.replace("Open-Meteo CAMS", "空气质量数据服务")   # ★长串先替换，否则被下面一行抢先切掉
    text = text.replace("Open-Meteo", "气象数据服务")
    text = text.replace("ExchangeRate-API", "汇率数据服务")
    text = text.replace("Open-Elevation", "高程数据服务")
    text = text.replace("DataV GeoAtlas", "行政区划数据服务")
    text = text.replace("WOFOST72", "作物生长模型")
    text = text.replace("era5", "再分析数据")
    text = text.replace("ERA5", "再分析数据")
    return text

async def execute_tool(tool_call: dict) -> str:
    fn_name = tool_call["function"]["name"]
    headers = {"X-API-Key": OASIS_API_KEY} if OASIS_API_KEY else {}
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except Exception:
        return json.dumps({"status": "error", "message": f"工具参数解析失败: {fn_name}"}, ensure_ascii=False)

    if fn_name == "crop_simulate":
        # 播种日期：按作物特性匹配"最近一个完整生长季"的播种期。
        # 保留模型选择的月日（春播/秋播由作物特性决定），年份往前找最近一轮
        # 满足"播种+预估生育期<今天"（即完整走完的生长季，NASA 历史数据可用）。
        planting = args.get("planting_date", "")
        try:
            md = str(planting)[5:10] if len(str(planting)) >= 10 else "04-15"
            today = datetime.now()
            crop_days = 170  # 预估生育期上限（覆盖小麦/玉米/棉花等）
            chosen = None
            for y in range(today.year - 1, today.year - 4, -1):
                try:
                    sow = datetime(y, int(md[:2]), int(md[3:5]))
                except:
                    continue
                if sow + timedelta(days=crop_days) < today:
                    chosen = f"{y}-{md}"
                    break
            planting = chosen if chosen else f"{today.year - 1}-{md}"
        except:
            planting = f"{datetime.now().year - 1}-04-15"
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                f"{PCSE_API_URL}/simulate",
                json={
                    "latitude": args["latitude"],
                    "longitude": args["longitude"],
                    "crop_no": args["crop_no"],
                    "planting_date": planting,
                    "production_mode": args.get("production_mode", "wlp"),
                    "initial_soil_moisture": 100.0,
                },
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"模拟失败: {resp.text[:300]}")
            data = resp.json()
            # 精简返回给模型：只保留 summary（关键数据），丢弃全量时间序列（省大量token）
            compact = {
                "status": "success",
                "summary": data.get("summary", {}),
                "message": data.get("message", ""),
                "growth_days": len(data.get("timeseries", {}).get("day", [])),
            }
            return sanitize_text(json.dumps(compact, ensure_ascii=False))

    elif fn_name == "weather_check":
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/weather/{args['latitude']}/{args['longitude']}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "weather_current":
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/weather/current/{args['latitude']}/{args['longitude']}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"实时天气查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "weather_forecast":
        days = min(args.get("days", 3), 7)  # 默认3天（省token），最多7天
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/weather/forecast/{args['latitude']}/{args['longitude']}?days={days}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"预报查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "weather_et0":
        days = args.get("days", 7)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/weather/et0/{args['latitude']}/{args['longitude']}?days={days}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"蒸散量查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "weather_history":
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/weather/history/{args['latitude']}/{args['longitude']}/{args['start']}/{args['end']}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"历史天气查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "weather_air":
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/weather/air/{args['latitude']}/{args['longitude']}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"空气质量查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "geocode":
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{PCSE_API_URL}/geocode/{args['name']}", headers=headers)
            if resp.status_code != 200:
                return sanitize_text(f"地理编码失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "exchange_rate":
        currency = args.get("currency", "CNY")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{PCSE_API_URL}/exchange-rate/{currency}", headers=headers)
            if resp.status_code != 200:
                return sanitize_text(f"汇率查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "soil_check":
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/soil/{args['latitude']}/{args['longitude']}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"土壤查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    elif fn_name == "soil_moisture":
        query = ""
        if args.get("start"):
            query += f"?start={args['start']}&end={args.get('end', '')}"
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/soil-moisture/{args['latitude']}/{args['longitude']}{query}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"墒情查询失败: {resp.text[:300]}")
            data = resp.json()
            # 精简：只保留最近10天 + 均值（省token）
            compact = {"status": "success", "location": data.get("location", {})}
            series = data.get("series", data.get("daily", []))
            if isinstance(series, list) and series:
                compact["recent_days"] = series[-10:]
                compact["total_days"] = len(series)
            elif isinstance(series, dict):
                compact["recent_days"] = {k: (v[-10:] if isinstance(v, list) else v) for k, v in series.items()}
            return sanitize_text(json.dumps(compact, ensure_ascii=False))

    elif fn_name == "gis_elevation":
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{PCSE_API_URL}/gis/elevation/{args['latitude']}/{args['longitude']}",
                headers=headers
            )
            if resp.status_code != 200:
                return sanitize_text(f"高程查询失败: {resp.text[:300]}")
            return sanitize_text(json.dumps(resp.json(), ensure_ascii=False))

    return json.dumps({"error": f"未知工具: {fn_name}"})

async def _safe_execute_tool(tool_call: dict) -> str:
    """execute_tool 安全包装：任何异常（网络超时/下游故障/KeyError）转为错误文本喂回模型，
    而不是让 asyncio.gather 把整轮对话炸掉"""
    try:
        return await execute_tool(tool_call)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"工具调用失败: {type(e).__name__}: {str(e)[:200]}"}, ensure_ascii=False)

def extract_cards(result: dict) -> list:
    """从模拟结果提取数据卡片"""
    summary = result.get("summary", {})
    cards = []
    if summary:
        def num(v):
            return v if isinstance(v, (int, float)) else None  # 非数值（None/str）不格式化，防崩溃丢卡片
        card = {}
        tagp = num(summary.get("TAGP"))
        twso = num(summary.get("TWSO"))
        laimax = num(summary.get("LAIMAX"))
        if tagp is not None:
            card["总生物量"] = f"{tagp:.0f} kg/ha"
        if twso is not None:
            card["产量"] = f"{twso:.0f} kg/ha"
        if laimax is not None:
            card["最大叶面积"] = f"{laimax:.2f}"
        if summary.get("DOE"):
            card["出苗"] = summary["DOE"]
        if summary.get("DOM"):
            card["成熟"] = summary["DOM"]
        if card:
            cards.append(card)
    return cards

# ========== 用户识别（匿名 Cookie）==========
import uuid as uuid_lib
from fastapi import Cookie, Response

USER_COOKIE = "oasis_uid"
COOKIE_MAX_AGE = 3600 * 24 * 365  # 1年

def get_uid(response: Response, uid: Optional[str] = Cookie(default=None, alias=USER_COOKIE)) -> str:
    """读取或生成匿名用户ID"""
    if not uid:
        uid = "u_" + uuid_lib.uuid4().hex[:16]
        response.set_cookie(
            USER_COOKIE, uid,
            max_age=COOKIE_MAX_AGE,
            httponly=True, samesite="lax",
        )
    return uid

def check_conversation_owner(conn, conv_id: int, uid: str) -> bool:
    """校验会话归属"""
    result = conn.execute(
        select(conversations.c.id).where(
            conversations.c.id == conv_id,
            conversations.c.user_id == uid,
        )
    )
    return result.first() is not None

def conv_row_to_dict(r):
    return {
        "id": r.id,
        "user_id": r.user_id,
        "title": r.title,
        "created_at": r.created_at,
    }

# ========== 匿名用户内存存储（不落库，服务重启即清）==========
# 登录用户（acc_*）数据存 PostgreSQL；匿名用户（u_*）数据只存内存
anon_store = {}  # uid -> {"conv_seq": int, "conversations": {cid: dict}, "messages": {cid: list}}

def is_anon(uid: str) -> bool:
    return uid.startswith("u_")

def anon_bucket(uid: str) -> dict:
    if uid not in anon_store:
        anon_store[uid] = {"conv_seq": 0, "conversations": {}, "messages": {}}
    return anon_store[uid]

def anon_list(uid: str):
    b = anon_bucket(uid)
    convs = list(b["conversations"].values())
    convs.sort(key=lambda c: c["id"], reverse=True)
    return convs

def anon_create(uid: str, title="新对话"):
    b = anon_bucket(uid)
    b["conv_seq"] += 1
    cid = b["conv_seq"]
    conv = {"id": cid, "user_id": uid, "title": title, "created_at": now_str()}
    b["conversations"][cid] = conv
    b["messages"][cid] = []
    return conv

def anon_get_messages(uid: str, conv_id: int):
    b = anon_bucket(uid)
    return b["messages"].get(conv_id, None)  # None 表示不存在

def anon_add_message(uid: str, conv_id: int, role: str, content: str, cards):
    b = anon_bucket(uid)
    msgs = b["messages"].get(conv_id)
    if msgs is None:
        return False
    msgs.append({
        "id": len(msgs) + 1,
        "conversation_id": conv_id,
        "role": role,
        "content": content,
        "cards": cards,
        "created_at": now_str(),
    })
    return True

def anon_delete(uid: str, conv_id: int):
    b = anon_bucket(uid)
    if conv_id in b["conversations"]:
        del b["conversations"][conv_id]
        b["messages"].pop(conv_id, None)
        return True
    return False

def anon_rename(uid: str, conv_id: int, title: str):
    b = anon_bucket(uid)
    conv = b["conversations"].get(conv_id)
    if conv:
        conv["title"] = title

# ========== 用户认证 API ==========

class RegisterRequest(BaseModel):
    username: str
    password: str
    phone: Optional[str] = None  # 预留：手机号

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/register")
def register(req: RegisterRequest, request: Request):
    """注册新用户"""
    username = req.username.strip()
    if not username or len(username) < 2:
        raise HTTPException(400, "用户名至少2个字符")
    if len(req.password) < 6:
        raise HTTPException(400, "密码至少6位")
    # 注册也限流（防止批量注册）
    limited = check_login_limited(request, username)
    if limited:
        raise HTTPException(429, f"尝试过于频繁，请 {limited} 秒后再试")
    try:
        with engine.begin() as conn:
            # 检查重名
            exists = conn.execute(
                select(users.c.id).where(users.c.username == username)
            ).first()
            if exists:
                raise HTTPException(400, "用户名已存在")
            result = conn.execute(
                insert(users).values(
                    username=username,
                    password_hash=hash_password(req.password),
                    phone=req.phone,
                    phone_verified="0",
                    created_at=now_str(),
                )
            )
            user_id = result.inserted_primary_key[0]
    except HTTPException:
        raise
    token = create_token(user_id, username)
    return {"token": token, "username": username, "user_id": user_id}

@app.post("/api/login")
def login(req: LoginRequest, request: Request):
    """用户登录"""
    username = req.username.strip()
    limited = check_login_limited(request, username)
    if limited:
        raise HTTPException(429, f"尝试过于频繁，请 {limited} 秒后再试")
    with engine.connect() as conn:
        row = conn.execute(
            select(users).where(users.c.username == username)
        ).first()
    if not row:
        record_login_fail(request, username)
        raise HTTPException(401, "用户名或密码错误")
    if not verify_password(req.password, row.password_hash):
        record_login_fail(request, username)
        raise HTTPException(401, "用户名或密码错误")
    record_login_success(username)
    token = create_token(row.id, row.username)
    return {"token": token, "username": row.username, "user_id": row.id}

@app.get("/api/me")
def me(request: Request):
    """获取当前登录用户信息"""
    user = get_token_user(request)
    if not user:
        raise HTTPException(401, "未登录")
    user_id, username = user
    with engine.connect() as conn:
        row = conn.execute(
            select(users.c.phone).where(users.c.id == user_id)
        ).first()
    return {"user_id": user_id, "username": username, "phone": row.phone if row else None, "is_admin": username == ADMIN_USERNAME}

@app.get("/api/ping")
def api_ping():
    """★ 保活探针（公开、无鉴权）：真读一次数据库，防止 Supabase 免费版因长时间无活动被暂停。
    被本机 ~/.hermes/scripts/oasis-keepalive.sh 与 GitHub Actions 定时访问。"""
    try:
        with engine.connect() as conn:
            conn.execute(_sql_text("SELECT 1"))
            docs = conn.execute(_sql_text("SELECT COUNT(*) FROM documents")).scalar()
            chunks = conn.execute(_sql_text("SELECT COUNT(*) FROM chunks")).scalar()
        return {"status": "ok", "db": "ok", "documents": docs, "chunks": chunks}
    except Exception as e:
        raise HTTPException(503, f"数据库不可用: {str(e)[:120]}")

def resolve_uid(request: Request, response: Response, cookie_uid: Optional[str]) -> str:
    """统一身份解析：优先 token 账号，否则匿名 cookie。
    登录用户返回 'acc_{id}'，匿名返回 'u_{xxx}'
    """
    user = get_token_user(request)
    if user:
        return f"acc_{user[0]}"
    return get_uid(response, cookie_uid)

def migrate_anon_to_account(anon_uid: str, acc_uid: str):
    """把匿名用户的会话迁移到账号名下"""
    if not anon_uid or anon_uid == acc_uid or not anon_uid.startswith("u_"):
        return 0
    with engine.begin() as conn:
        result = conn.execute(
            update(conversations)
            .where(conversations.c.user_id == anon_uid)
            .values(user_id=acc_uid)
        )
        return result.rowcount or 0

# ========== 知识库管理 API ==========
from sqlalchemy import text as sql_text

@app.post("/api/knowledge/upload")
async def upload_knowledge(request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """上传文档到知识库（仅管理员）"""
    user = get_token_user(request)
    if not user:
        raise HTTPException(401, "请先登录")
    if user[1] != ADMIN_USERNAME:
        raise HTTPException(403, "仅管理员可更新知识库")
    try:
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise HTTPException(400, "未收到文件")
        data = await file.read()
        filename = getattr(file, "filename", None) or "document.txt"
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(400, "文件过大（限制10MB）")

        # 解析文本
        text_content = parse_document(filename, data)
        if len(text_content) < 20:
            raise HTTPException(400, "文档内容过少或无法解析")

        # AI 编译：整理成精编知识页（LLM-wiki 模式）
        compiled = await compile_document(text_content)

        # 切块（编译后的精编页）
        chunks = chunk_text(compiled)
        if not chunks:
            raise HTTPException(400, "无法从文档中提取文本")

        # 向量化（分批，每批10条，避免下游内存压力）
        vectors = []
        for i in range(0, len(chunks), 10):
            batch = chunks[i:i+10]
            vectors.extend(await embed_texts(batch))

        # 入库
        with engine.begin() as conn:
            result = conn.execute(
                sql_text("INSERT INTO documents (title, source, created_at) VALUES (:t, :s, :c) RETURNING id"),
                {"t": filename, "s": f"uploaded_by_{user[1]}", "c": now_str()},
            )
            doc_id = result.fetchone()[0]
            for content, vec in zip(chunks, vectors):
                vec_str = "[" + ",".join(str(x) for x in vec) + "]"
                conn.execute(
                    sql_text(f"INSERT INTO chunks (document_id, content, embedding, created_at) VALUES (:d, :c, '{vec_str}'::vector, :t)"),
                    {"d": doc_id, "c": content, "t": now_str()},
                )

        return {"status": "ok", "document_id": doc_id, "title": filename, "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"上传失败: {str(e)}")

@app.get("/api/knowledge")
def list_knowledge(request: Request):
    """文档列表（含块数）——仅管理员可查看（修复：此前无鉴权，匿名可枚举全部文档）"""
    user = get_token_user(request)
    if not user:
        raise HTTPException(401, "请先登录")
    if user[1] != ADMIN_USERNAME:
        raise HTTPException(403, "仅管理员可查看知识库")
    with engine.connect() as conn:
        rows = conn.execute(
            sql_text("""
                SELECT d.id, d.title, d.source, d.created_at,
                       (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
                FROM documents d ORDER BY d.id DESC
            """)
        ).fetchall()
    return [
        {"id": r.id, "title": r.title, "source": r.source, "created_at": r.created_at, "chunks": r.chunk_count}
        for r in rows
    ]

@app.delete("/api/knowledge/{doc_id}")
def delete_knowledge(doc_id: int, request: Request):
    """删除文档及其块（仅管理员）"""
    user = get_token_user(request)
    if not user:
        raise HTTPException(401, "请先登录")
    if user[1] != ADMIN_USERNAME:
        raise HTTPException(403, "仅管理员可更新知识库")
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
    return {"status": "ok"}

@app.post("/api/knowledge/search")
async def search_knowledge_api(request: Request):
    """测试检索：返回相关段落——仅管理员可调用（修复：此前匿名可检索知识库全文）"""
    user = get_token_user(request)
    if not user:
        raise HTTPException(401, "请先登录")
    if user[1] != ADMIN_USERNAME:
        raise HTTPException(403, "仅管理员可检索知识库")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "请求体必须是 JSON")
    query = body.get("query", "")
    if not query:
        raise HTTPException(400, "缺少 query")
    results = await search_knowledge(query)
    return {"results": results}

# ========== 会话 API ==========

@app.get("/api/conversations")
def list_conversations(request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """会话列表（登录用户查云端，匿名用户查内存）"""
    uid = resolve_uid(request, response, uid)
    if is_anon(uid):
        return anon_list(uid)
    with engine.connect() as conn:
        result = conn.execute(
            select(conversations)
            .where(conversations.c.user_id == uid)
            .order_by(conversations.c.id.desc())
        )
        return [conv_row_to_dict(r) for r in result]

@app.post("/api/conversations")
def create_conversation(request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """创建新会话（匿名存内存，登录存云端）"""
    uid = resolve_uid(request, response, uid)
    if is_anon(uid):
        conv = anon_create(uid)
        return {"id": conv["id"], "title": conv["title"], "created_at": conv["created_at"]}
    with engine.begin() as conn:
        result = conn.execute(
            insert(conversations).values(user_id=uid, title="新对话", created_at=now_str())
        )
        conv_id = result.inserted_primary_key[0]
    return {"id": conv_id, "title": "新对话", "created_at": now_str()}

@app.get("/api/conversations/{conv_id}/messages")
def get_messages(conv_id: int, request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """获取某会话的消息（匿名查内存，登录查云端）"""
    uid = resolve_uid(request, response, uid)
    if is_anon(uid):
        msgs = anon_get_messages(uid, conv_id)
        if msgs is None:
            raise HTTPException(404, "会话不存在")
        return {"conversation_id": conv_id, "messages": msgs}
    with engine.connect() as conn:
        if not check_conversation_owner(conn, conv_id, uid):
            raise HTTPException(404, "会话不存在")
        result = conn.execute(
            select(messages)
            .where(messages.c.conversation_id == conv_id)
            .order_by(messages.c.id)
        )
        rows = result.fetchall()
    messages_list = []
    for r in rows:
        m = {
            "id": r.id,
            "conversation_id": r.conversation_id,
            "role": r.role,
            "content": r.content,
            "cards": None,
            "created_at": r.created_at,
        }
        try:
            m["cards"] = json.loads(r.cards) if r.cards else None
        except:
            m["cards"] = None
        messages_list.append(m)
    return {"conversation_id": conv_id, "messages": messages_list}

@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: int, request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """删除会话及其消息（匿名删内存，登录删云端）"""
    uid = resolve_uid(request, response, uid)
    if is_anon(uid):
        if not anon_delete(uid, conv_id):
            raise HTTPException(404, "会话不存在")
        return {"status": "ok"}
    with engine.begin() as conn:
        if not check_conversation_owner(conn, conv_id, uid):
            raise HTTPException(404, "会话不存在")
        conn.execute(delete(messages).where(messages.c.conversation_id == conv_id))
        conn.execute(delete(conversations).where(conversations.c.id == conv_id))
    return {"status": "ok"}

@app.post("/api/migrate")
def migrate_conversations(request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """登录后调用：把匿名内存会话迁移到账号云端存储"""
    user = get_token_user(request)
    if not user:
        raise HTTPException(401, "未登录")
    anon_uid = uid or get_uid(response, uid)
    acc_uid = f"acc_{user[0]}"
    if not is_anon(anon_uid) or anon_uid not in anon_store:
        return {"status": "ok", "migrated": 0}
    bucket = anon_store.get(anon_uid)  # 先读不 pop：DB 事务失败时匿名数据不丢
    migrated = 0
    convs = sorted(bucket["conversations"].values(), key=lambda c: c["id"])
    with engine.begin() as conn:
        for conv in convs:
            result = conn.execute(
                insert(conversations).values(
                    user_id=acc_uid, title=conv["title"], created_at=conv["created_at"]
                )
            )
            new_id = result.inserted_primary_key[0]
            for m in bucket["messages"].get(conv["id"], []):
                conn.execute(
                    insert(messages).values(
                        conversation_id=new_id, role=m["role"], content=m["content"],
                        cards=json.dumps(m["cards"], ensure_ascii=False) if m.get("cards") else None,
                        created_at=m["created_at"],
                    )
                )
            migrated += 1
    anon_store.pop(anon_uid, None)  # 事务成功后清除匿名数据（失败时保留，可重试）
    return {"status": "ok", "migrated": migrated}

# ========== 主页面 ==========

@app.get("/", response_class=HTMLResponse)
async def chat_page():
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ========== 对话 ==========

# ========== 工具步骤描述（SSE 实时展示）==========
# 已知新疆地名坐标映射（显示友好地名）
KNOWN_PLACES = {
    (38.48, 77.26): "莎车", (41.17, 80.25): "阿克苏", (44.30, 86.03): "石河子",
    (43.92, 81.32): "伊犁", (43.82, 87.62): "乌鲁木齐", (42.95, 89.19): "吐鲁番",
    (43.90, 87.50): "乌鲁木齐", (39.47, 75.99): "喀什",
}

def parse_text_tool_calls(content: str):
    """解析模型输出的文本工具调用（兜底：DeepSeek V4 偶发输出 DSML 格式
    <DSMLtool_calls><DSMLinvoke name="x"><DSMLparameter name="y">v</DSMLparameter></DSMLinvoke></DSMLtool_calls>
    而非标准 JSON tool_calls）"""
    calls = []
    if not content:
        return calls
    # 支持 <invoke name="x"> 与 <DSMLinvoke name="x"> 两种标签
    pattern = re.compile(r'<(?:DSML)?invoke\s+name="([^"]+)"[^>]*>(.*?)</(?:DSML)?invoke>', re.DOTALL)
    for m in pattern.finditer(content):
        name = m.group(1).strip()
        args_text = m.group(2).strip()
        if not name:
            continue
        args = {}
        try:
            args = json.loads(args_text)
        except:
            # 支持 <parameter name="key">value</parameter> 与 <DSMLparameter name="key">value</DSMLparameter>
            for pm in re.finditer(r'<(?:DSML)?parameter\s+name="([^"]+)"[^>]*>(.*?)</(?:DSML)?parameter>', args_text, re.DOTALL):
                val = pm.group(2).strip()
                try:
                    args[pm.group(1)] = float(val) if '.' in val else int(val)
                except:
                    args[pm.group(1)] = val
            # 兜底：key="value" 格式
            if not args:
                for km in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_text):
                    args[km.group(1)] = km.group(2)
            if not args and args_text:
                args = {"text": args_text}
        calls.append({
            "id": f"textcall_{len(calls)}",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    return calls

def describe_tool_step(fn_name: str, args: dict) -> str:
    """把工具调用翻译成人类可读的步骤描述"""
    place = args.get("location") or args.get("name") or ""
    lat = args.get("latitude")
    lon = args.get("longitude")
    crop = args.get("crop_name") or args.get("crop_no", "")
    if lat and lon:
        # 尝试匹配已知地名
        place = KNOWN_PLACES.get((round(lat, 2), round(lon, 2)), "") or f"({lat}, {lon})"
    steps = {
        "crop_simulate": f"正在运行植物生长机理模拟（{place}{crop}）…",
        "weather_current": f"正在查询{place}实时天气…",
        "weather_forecast": f"正在查询{place}天气预报…",
        "weather_et0": f"正在计算{place}蒸散量与灌溉需水量…",
        "weather_history": "正在查询历史天气…",
        "weather_air": f"正在查询{place}空气质量与紫外线…",
        "weather_check": f"正在检查{place}气象数据…",
        "geocode": f"正在定位{args.get('name', '')}坐标…",
        "exchange_rate": "正在查询实时汇率…",
        "soil_check": f"正在查询{place}土壤数据…",
        "soil_moisture": f"正在查询{place}土壤墒情…",
        "gis_elevation": f"正在查询{place}海拔高程…",
    }
    return steps.get(fn_name, f"正在处理{fn_name}…")

def _is_smalltalk(text: str) -> bool:
    """寒暄/极短消息检测：跳过 RAG 检索（无意义且浪费 embed 调用时间）。
    仅整句匹配寒暄词才跳过——不能用 startswith("你好")，否则"你好，小麦何时播种"这类实质问题会被误伤。"""
    t = (text or "").strip().lower()
    if not t or len(t) <= 2:
        return True
    small = {"你好", "您好", "hi", "hello", "hey", "在吗", "谢谢", "感谢", "好的", "ok", "嗯", "再见", "bye",
             "早上好", "中午好", "晚上好", "你是谁", "介绍一下", "你能做什么", "你会什么", "可以帮我吗"}
    return t in small

async def handle_chat(uid: str, conv_id, user_text: str, step_cb=None, stream_cb=None):
    """核心对话逻辑（文本/语音共用）：确保会话 → 存用户消息 → DeepSeek(含工具) → 存AI消息
    step_cb: 可选异步回调，工具调用时推送步骤描述（SSE 实时展示）
    stream_cb: 可选异步回调，最终回答流式推送文本增量（打字机效果）"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(500, "服务未完成配置，请联系管理员")

    # 1. 确保会话存在（匿名存内存，登录存云端）
    if is_anon(uid):
        if conv_id is None:
            conv = anon_create(uid, title=make_title(user_text))
            conv_id = conv["id"]
        else:
            if anon_get_messages(uid, conv_id) is None:
                raise HTTPException(404, "会话不存在")
            anon_rename(uid, conv_id, make_title(user_text))
    else:
        if conv_id is None:
            with engine.begin() as conn:
                result = conn.execute(
                    insert(conversations).values(user_id=uid, title=make_title(user_text), created_at=now_str())
                )
                conv_id = result.inserted_primary_key[0]
        else:
            with engine.begin() as conn:
                if not check_conversation_owner(conn, conv_id, uid):
                    raise HTTPException(404, "会话不存在")
                conn.execute(
                    update(conversations)
                    .where(conversations.c.id == conv_id, conversations.c.title == "新对话")
                    .values(title=make_title(user_text))
                )

    # 2. 保存用户消息
    if is_anon(uid):
        anon_add_message(uid, conv_id, "user", user_text, None)
    else:
        with engine.begin() as conn:
            conn.execute(
                insert(messages).values(
                    conversation_id=conv_id, role="user",
                    content=user_text, cards=None, created_at=now_str(),
                )
            )

    # 2.5 RAG 检索（启用时）：检索相关知识点拼入上下文（精简每段，省token）
    # 优化：寒暄/简短消息跳过 RAG——"你好"这类问题检索无意义，省一次 embed+DB 查询（1-3s）
    rag_context = ""
    if RAG_ENABLED and not _is_smalltalk(user_text):
        try:
            hits = await search_knowledge(user_text)
            if hits:
                parts = []
                for h in hits[:RAG_TOP_K]:
                    # 每段截断到250字，避免上下文膨胀
                    content = h['content'] if len(h['content']) <= 250 else h['content'][:250] + "…"
                    parts.append(f"【{h['title']}】{content}")
                rag_context = "\n\n参考知识库内容（用户提问前请先参考这些资料）：\n" + "\n\n".join(parts)
        except Exception as e:
            print(f"[RAG] 集成失败: {e}")

    # 2.6 加载会话历史（多轮对话记忆，按字符截断控制上下文长度）
    history = []
    try:
        if is_anon(uid):
            msgs = anon_get_messages(uid, conv_id) or []
            history = [{"role": m["role"], "content": m["content"]} for m in msgs[-30:]]
        else:
            with engine.connect() as conn:
                rows = conn.execute(
                    select(messages.c.role, messages.c.content)
                    .where(messages.c.conversation_id == conv_id)
                    .order_by(messages.c.id.desc())
                    .limit(30)
                ).fetchall()
                history = [{"role": r.role, "content": r.content} for r in reversed(rows)]
        # 去掉最后一条（当前用户消息，避免重复）
        if history and history[-1]["role"] == "user" and history[-1]["content"] == user_text:
            history = history[:-1]
        # 按字符截断（约6000字 ≈ 4000 token，控制推理速度；保留最近消息）
        total_chars = 0
        trimmed = []
        for h in reversed(history):  # 从最新往旧累计
            total_chars += len(h["content"])
            trimmed.append(h)
            if total_chars > 6000:
                break
        history = list(reversed(trimmed))
    except Exception as e:
        print(f"[记忆] 历史加载失败: {e}")
        history = []

    # 3. 调用 DeepSeek（带完整上下文）
    system_content = SYSTEM_PROMPT
    # 动态当前日期：让模型知道今天日期，播种/出苗/成熟预测用未来日期
    system_content += f"\n\n【当前日期：{datetime.now().strftime('%Y年%m月%d日')}】当用户询问播种、出苗、成熟时间时，请选择当前日期之后的最近适宜时间。注意：作物模拟返回的生长日期是上一个完整生长季（去年）的数据，作为'上一轮'参考；回答时要换算成今年/明年的预计时间。"
    if rag_context:
        system_content += rag_context
    llm_msgs = [{"role": "system", "content": system_content}]
    llm_msgs.extend(history)
    llm_msgs.append({"role": "user", "content": user_text})

    response = await call_deepseek(llm_msgs)
    choice = response["choices"][0]
    msg = choice["message"]

    # 4. 处理工具调用（循环执行，最多10轮，直到模型给出最终回答）
    #    支持 JSON tool_calls 和文本 <invoke> 两种格式（兜底防乱码）
    cards = []
    max_rounds = 10
    rounds = 0
    while True:
        tool_calls = msg.get("tool_calls") or parse_text_tool_calls(msg.get("content") or "")
        if not tool_calls:
            break
        rounds += 1
        llm_msgs.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
        })

        # 逐步推送"正在做什么"（SSE 步骤显示）
        for tool_call in tool_calls:
            try:
                fn_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                step_text = describe_tool_step(fn_name, args)
                if step_cb:
                    await step_cb(step_text)
            except:
                pass

        # 并行执行同轮的所有工具调用（大幅提速）
        results = await asyncio.gather(*[_safe_execute_tool(tc) for tc in tool_calls])
        for tool_call, result in zip(tool_calls, results):
            llm_msgs.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })
            try:
                result_data = json.loads(result)
                if result_data.get("status") == "success":
                    cards = extract_cards(result_data)
            except:
                pass

        if rounds >= max_rounds:
            # 达到轮次上限：非流式拿最终回答
            response = await call_deepseek(llm_msgs)
            msg = response["choices"][0]["message"]
            break

        response = await call_deepseek(llm_msgs)
        choice = response["choices"][0]
        msg = choice["message"]

    # 5. 最终回答（有工具轮且启用流式 → 打字机效果）
    if msg.get("tool_calls"):
        # 到达轮次上限仍未结束：非流式兜底，最多再追 2 轮，防空回复
        for _ in range(3):
            response = await call_deepseek(llm_msgs)
            msg = response["choices"][0]["message"]
            if not msg.get("tool_calls"):
                break
            llm_msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]})
            results = await asyncio.gather(*[_safe_execute_tool(tc) for tc in msg["tool_calls"]])
            for tc, r in zip(msg["tool_calls"], results):
                llm_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": r})
        reply = sanitize_text(msg.get("content") or "")
        if not reply.strip():
            reply = "抱歉，回答未能生成，请换个问法再试。"
    elif stream_cb and rounds > 0:
        reply = ""
        dsml_text = ""
        suspect = ""
        # 源头抑制：显式告知模型最终回答不要输出任何工具调用标签
        stream_msgs = [{"role": "system", "content": "直接输出最终回答文本。禁止输出任何 XML/DSML 标签（如 <invoke>、<DSMLinvoke>、<tool_calls> 等）。如果还需要查询数据，直接省略或简要说明即可。回答简洁精炼，一般不超过500字，用markdown组织。"}] + list(llm_msgs)
        async for chunk in call_deepseek_stream(stream_msgs):
            if not chunk:
                continue
            suspect += chunk
            # 收紧后的检测：只匹配真正的工具调用标签（DSMLinvoke/invoke/tool_calls/全角双竖线变体），
            # 不再用裸 < ｜ 〈 字符类——正常中文的 | 分隔符、〈〉书名号、x<5 不再误判截断回答
            if not dsml_text and re.search(r'<(?:DSML)?[a-zA-Z/!]*invoke|<(?:DSML)?tool_calls|<DSML|｜｜\s*(?:DSML)?invoke|DSMLtool_calls|DSMLinvoke', suspect):
                # 模型开始输出工具调用（DSML文本，含全角变体）→ 停止推送，收集用于解析
                dsml_text = suspect
                continue
            if dsml_text:
                dsml_text += chunk
                continue
            reply += chunk
            if stream_cb:
                await stream_cb(sanitize_text(chunk))
        # 流式输出可能仍含 DSML 工具调用（模型还想调用工具）→ 执行后继续
        dsml_calls = parse_text_tool_calls(dsml_text or reply)
        if dsml_calls:
            try:
                assistant_text = sanitize_text(reply).strip()
                llm_msgs.append({"role": "assistant", "content": assistant_text or "继续分析"})
                results = await asyncio.gather(*[_safe_execute_tool(tc) for tc in dsml_calls])
                for tc, r in zip(dsml_calls, results):
                    llm_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": r})
                    try:
                        rd = json.loads(r)
                        if rd.get("status") == "success":
                            cards = extract_cards(rd)
                    except:
                        pass
                response = await call_deepseek(llm_msgs)
                reply = sanitize_text(response["choices"][0]["message"]["content"] or "")
            except Exception as e:
                print(f"[DSML继续] 失败: {e}")
                reply = sanitize_text(reply)
        elif dsml_text:
            # 检测到工具调用文本但解析失败（被截断）→ 让模型重新以 JSON 格式调用
            try:
                assistant_text = sanitize_text(reply).strip()
                if assistant_text:
                    llm_msgs.append({"role": "assistant", "content": assistant_text})
                response = await call_deepseek(llm_msgs)
                msg2 = response["choices"][0]["message"]
                if msg2.get("tool_calls"):
                    llm_msgs.append({"role": "assistant", "content": msg2.get("content") or "", "tool_calls": msg2["tool_calls"]})
                    results = await asyncio.gather(*[_safe_execute_tool(tc) for tc in msg2["tool_calls"]])
                    for tc, r in zip(msg2["tool_calls"], results):
                        llm_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": r})
                        try:
                            rd = json.loads(r)
                            if rd.get("status") == "success":
                                cards = extract_cards(rd)
                        except:
                            pass
                    response = await call_deepseek(llm_msgs)
                reply = sanitize_text(response["choices"][0]["message"]["content"] or "")
            except Exception as e:
                print(f"[DSML重试] 失败: {e}")
                reply = sanitize_text(reply)
        else:
            reply = sanitize_text(reply)
    else:
        reply = sanitize_text(msg.get("content") or "")

    # 5. 保存 AI 消息
    if is_anon(uid):
        anon_add_message(uid, conv_id, "assistant", reply, cards if cards else None)
    else:
        with engine.begin() as conn:
            conn.execute(
                insert(messages).values(
                    conversation_id=conv_id, role="assistant",
                    content=reply,
                    cards=json.dumps(cards, ensure_ascii=False) if cards else None,
                    created_at=now_str(),
                )
            )

    return conv_id, reply, cards

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """处理用户文本消息"""
    uid = resolve_uid(request, response, uid)
    try:
        conv_id, reply, cards = await handle_chat(uid, req.conversation_id, req.message)
        return ChatResponse(reply=reply, cards=cards if cards else None)
    except HTTPException:
        raise  # 404 会话不存在/越权等保持原状态码，不伪装成 200
    except Exception as e:
        traceback.print_exc()
        return ChatResponse(reply="抱歉，服务暂时不可用，请稍后再试。")

# ========== 流式对话（SSE：实时推送"正在做什么"）==========
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """SSE 流式对话：实时推送步骤（工具调用）与最终回答"""
    uid = resolve_uid(request, response, uid)

    async def event_gen():
        def sse(data: dict):
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        try:
            yield sse({"type": "start"})

            async def step_cb(text: str):
                yield_text = sse({"type": "step", "text": text})
                # 通过队列传递给生成器（简化：直接 yield 不可行，用队列）
                await step_queue.put(yield_text)

            step_queue = asyncio.Queue()

            async def run():
                try:
                    await step_queue.put(sse({"type": "step", "text": "正在分析问题…"}))
                    async def token_cb(text: str):
                        # 流式 token 也过滤（避免打字机阶段显示工具标签）
                        clean = sanitize_text(text)
                        if clean:
                            await step_queue.put(sse({"type": "token", "content": clean}))
                    conv_id, reply, cards = await handle_chat(uid, req.conversation_id, req.message, step_cb=step_cb, stream_cb=token_cb)
                    if not cards:
                        await step_queue.put(sse({"type": "step", "text": "正在整合数据，生成回答…"}))
                    await step_queue.put(sse({"type": "result", "reply": reply, "cards": cards or []}))
                except Exception as e:
                    traceback.print_exc()
                    await step_queue.put(sse({"type": "result", "reply": "抱歉，服务暂时不可用，请稍后再试。", "cards": []}))

            task = asyncio.create_task(run())
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(step_queue.get(), timeout=0.5)
                        yield item
                    except asyncio.TimeoutError:
                        if task.done():
                            # 取完队列剩余（保证 result 不丢）
                            while not step_queue.empty():
                                yield await step_queue.get()
                            break
                        continue
            finally:
                # 客户端断开（GeneratorExit）时取消后台任务，避免 DeepSeek/PCSE 调用孤儿化
                if not task.done():
                    task.cancel()
        except Exception as e:
            traceback.print_exc()
            yield sse({"type": "result", "reply": "抱歉，服务暂时不可用，请稍后再试。", "cards": []})

    resp = StreamingResponse(event_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    # 修复：直接返回 StreamingResponse 时注入 response 的 set_cookie 不生效，
    # 匿名用户首次直连 /chat/stream 会拿不到 cookie → 下次请求变新 uid → 会话 404
    if uid.startswith("u_") and not request.cookies.get(USER_COOKIE):
        resp.set_cookie(USER_COOKIE, uid, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
    return resp

# ========== 语音对话 ==========
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_ASR_MODEL = os.environ.get("SILICONFLOW_ASR_MODEL", "FunAudioLLM/SenseVoiceSmall")
# ========== RAG 向量化服务 ==========
EMBED_API_URL = os.environ.get("EMBED_API_URL", "https://oasis-embed-4jsf.onrender.com")
EMBED_DIM = 512
RAG_ENABLED = os.environ.get("RAG_ENABLED", "1") == "1"
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
# 管理员账号（唯一可更新知识库的账号）
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "guanliyuan")

@app.post("/api/voice", response_model=ChatResponse)
async def voice_chat(request: Request, response: Response, uid: str = Cookie(default=None, alias=USER_COOKIE)):
    """语音输入：接收音频 → 语音转文字 → 走对话流程"""
    uid = resolve_uid(request, response, uid)
    if not SILICONFLOW_API_KEY:
        return ChatResponse(reply="语音识别服务暂未配置，请稍后再试。")
    try:
        form = await request.form()
        audio_file = form.get("audio")
        conv_id_raw = form.get("conversation_id")
        conv_id = int(conv_id_raw) if conv_id_raw else None
        if audio_file is None:
            return ChatResponse(reply="未收到音频文件。")
        # 读取音频数据
        audio_bytes = await audio_file.read()
        filename = getattr(audio_file, "filename", None) or "voice.webm"
        if len(audio_bytes) < 100:
            return ChatResponse(reply="录音太短，请再说一次。")

        # 调用语音识别（硅基流动 Whisper，OpenAI 兼容）
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.siliconflow.cn/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
                files={"file": (filename, audio_bytes, "application/octet-stream")},
                data={"model": SILICONFLOW_ASR_MODEL, "language": "zh"},
            )
            if resp.status_code != 200:
                return ChatResponse(reply=f"语音识别失败，请重试。")
            asr = resp.json()
            user_text = (asr.get("text") or "").strip()
            if not user_text:
                return ChatResponse(reply="没听清您说什么，请再说一次。")

        # 走正常对话流程
        conv_id, reply, cards = await handle_chat(uid, conv_id, user_text)
        return ChatResponse(reply=reply, cards=cards if cards else None, text=user_text)
    except Exception as e:
        traceback.print_exc()
        return ChatResponse(reply="抱歉，服务暂时不可用，请稍后再试。")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
