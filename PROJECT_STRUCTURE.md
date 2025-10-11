# 🏗️ 项目结构说明

## 📁 目录结构

```
video_translator/
├── domain/                          # 领域层（核心）
│   ├── __init__.py
│   ├── entities.py                  # 实体定义
│   ├── value_objects.py             # 值对象
│   ├── services.py                  # 领域服务（纯函数）
│   ├── ports.py                     # 接口定义（Port）
│   └── exceptions.py                # 领域异常
│
├── application/                     # 应用层（用例）
│   ├── __init__.py
│   ├── use_cases/                   # 用例实现
│   │   ├── __init__.py
│   │   ├── generate_subtitles.py   # 字幕生成用例
│   │   ├── clone_voice.py          # 语音克隆用例
│   │   ├── synthesize_video.py     # 视频合成用例
│   │   ├── batch_process.py        # 批量处理用例
│   │   └── test_component.py       # 组件测试用例
│   └── dto.py                       # 数据传输对象
│
├── infrastructure/                  # 基础设施层（适配器）
│   ├── __init__.py
│   ├── adapters/                    # 适配器实现
│   │   ├── __init__.py
│   │   ├── asr/                     # ASR 适配器
│   │   │   ├── whisper_adapter.py
│   │   │   └── faster_whisper_adapter.py
│   │   ├── translation/             # 翻译适配器
│   │   │   ├── qwen_adapter.py
│   │   │   └── gpt_adapter.py
│   │   ├── tts/                     # TTS 适配器
│   │   │   ├── f5tts_adapter.py
│   │   │   └── xtts_adapter.py
│   │   ├── video/                   # 视频处理适配器
│   │   │   └── ffmpeg_adapter.py
│   │   ├── subtitle/                # 字幕写入适配器
│   │   │   └── pysrt_adapter.py
│   │   └── storage/                 # 存储适配器
│   │       ├── cache_adapter.py
│   │       └── file_adapter.py
│   │
│   ├── ui/                          # 用户界面
│   │   ├── __init__.py
│   │   ├── webui.py                # Gradio WebUI
│   │   └── cli.py                  # 命令行界面
│   │
│   └── config/                      # 配置
│       ├── __init__.py
│       ├── dependency_injection.py  # 依赖注入容器
│       └── settings.py              # 配置管理
│
├── tests/                           # 测试
│   ├── unit/                        # 单元测试
│   │   ├── test_domain/
│   │   ├── test_application/
│   │   └── test_infrastructure/
│   ├── integration/                 # 集成测试
│   └── e2e/                         # 端到端测试
│
├── scripts/                         # 脚本
│   ├── install.sh                   # 安装脚本
│   └── run_webui.py                # 启动脚本
│
├── docs/                            # 文档
│   ├── ARCHITECTURE.md              # 架构说明
│   ├── API.md                       # API 文档
│   └── DEVELOPMENT.md               # 开发指南
│
├── requirements.txt                 # Python 依赖
├── setup.py                         # 安装配置
├── README.md                        # 项目说明
└── .env.example                     # 环境变量示例
```

---

## 🎯 架构层次详解

### 1. Domain Layer（领域层）- 最内层

**职责**: 纯业务逻辑，无副作用，不依赖外部

#### 文件组织

```python
# domain/entities.py
@dataclass(frozen=True)
class Video:
    """视频实体（不可变）"""
    path: Path
    duration: float
    has_audio: bool

# domain/value_objects.py
@dataclass(frozen=True)
class TimeRange:
    """时间范围值对象"""
    start_seconds: float
    end_seconds: float

# domain/services.py
def merge_bilingual_subtitles(
    primary: Subtitle,
    secondary: Subtitle
) -> Subtitle:
    """合并双语字幕（纯函数）"""
    pass

# domain/ports.py
class ASRProvider(Protocol):
    """ASR 提供者接口"""
    def transcribe(self, audio_path: Path) -> tuple[tuple[TextSegment, ...], LanguageCode]:
        ...
```

**关键特性**:
- ✅ 所有类使用 `@dataclass(frozen=True)` 保证不可变
- ✅ 所有函数都是纯函数
- ✅ 使用 `Protocol` 定义接口
- ✅ 不导入任何外层模块

---

### 2. Application Layer（应用层）

**职责**: 编排领域服务，定义用例，纯函数实现

#### 文件组织

```python
# application/use_cases/generate_subtitles.py
def generate_subtitles_use_case(
    video: Video,
    asr_provider: ASRProvider,          # 依赖接口
    translation_provider: TranslationProvider,
    cache_repo: CacheRepository,
    progress: ProgressCallback = None
) -> SubtitleGenerationResult:
    """
    字幕生成用例（纯函数）
    
    通过依赖注入接收所有外部依赖
    """
    # 1. 检查缓存
    # 2. 调用 ASR
    # 3. 调用翻译
    # 4. 返回结果
    pass
```

**关键特性**:
- ✅ 所有用例都是纯函数
- ✅ 通过参数接收依赖（依赖注入）
- ✅ 只依赖领域层接口
- ✅ 返回不可变的结果对象

---

### 3. Infrastructure Layer（基础设施层）- 最外层

**职责**: 实现接口，处理副作用，集成外部服务

#### 3.1 Adapters（适配器）

```python
# infrastructure/adapters/asr/whisper_adapter.py
class WhisperASRAdapter:
    """实现 ASRProvider 接口"""
    
    def __init__(self, model_size: str = "large-v3"):
        self.model_size = model_size
        self._model = None  # 懒加载
    
    def transcribe(self, audio_path: Path, ...) -> tuple[tuple[TextSegment, ...], LanguageCode]:
        """有副作用的实现：加载模型、GPU 计算"""
        model = self._load_model()  # 副作用
        result = model.transcribe(str(audio_path))  # 副作用
        return self._convert_to_domain(result)  # 转换为领域对象
```

#### 3.2 UI（用户界面）

```python
# infrastructure/ui/webui.py
def process_single_video_ui(video_file, ...):
    """UI 处理函数（有副作用）"""
    
    # 1. 创建依赖
    container = DependencyContainer()
    
    # 2. 创建领域对象
    video = Video(path=Path(video_file.name), ...)
    
    # 3. 调用用例（依赖注入）
    result = generate_subtitles_use_case(
        video=video,
        asr_provider=container.get_asr(),
        translation_provider=container.get_translator(),
        cache_repo=container.cache_repo,
    )
    
    # 4. 返回 UI 格式
    return format_for_ui(result)
```

#### 3.3 Dependency Injection（依赖注入容器）

```python
# infrastructure/config/dependency_injection.py
class DependencyContainer:
    """依赖注入容器"""
    
    def __init__(self):
        self.cache_repo = FileCacheRepositoryAdapter()
        self.video_processor = FFmpegVideoProcessorAdapter()
        self._asr = None  # 懒加载
    
    def get_asr(self, model_size: str) -> ASRProvider:
        """返回 ASR 提供者（实现接口）"""
        if self._asr is None:
            self._asr = WhisperASRAdapter(model_size)
        return self._asr
```

---

## 🔄 数据流示例

### 完整流程：用户上传视频 → 生成字幕

```
┌─────────────────────────────────────────────────────────────┐
│ 1. UI Layer (Infrastructure)                                │
│    webui.py: process_single_video_ui()                      │
│    - 接收用户输入（Gradio File）                              │
│    - 创建依赖容器                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 调用用例
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Application Layer                                         │
│    use_cases/generate_subtitles.py                          │
│    generate_subtitles_use_case(video, asr, translator, ...) │
│    - 纯函数编排                                               │
│    - 调用 asr.transcribe()                                   │
│    - 调用 translator.translate()                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 调用接口
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Domain Layer                                              │
│    ports.py: ASRProvider, TranslationProvider (接口定义)     │
│    services.py: merge_bilingual_subtitles() (纯函数)        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ 实现接口
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Infrastructure Adapters                                   │
│    adapters/asr/whisper_adapter.py: WhisperASRAdapter       │
│    - 加载 Whisper 模型（副作用）                              │
│    - GPU 计算（副作用）                                       │
│    - 返回领域对象                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 测试策略

### 1. 单元测试（Domain & Application）

```python
# tests/unit/test_domain/test_services.py
def test_merge_bilingual_subtitles():
    """测试纯函数（无需 mock）"""
    primary = Subtitle(...)
    secondary = Subtitle(...)
    
    result = merge_bilingual_subtitles(primary, secondary)
    
    assert len(result.segments) == len(primary.segments)
    assert "translated\noriginal" in result.segments[0].text

# tests/unit/test_application/test_use_cases.py
def test_generate_subtitles_use_case():
    """测试用例（mock 接口）"""
    # Mock 依赖
    mock_asr = Mock(spec=ASRProvider)
    mock_translator = Mock(spec=TranslationProvider)
    mock_cache = Mock(spec=CacheRepository)
    
    # 设置返回值
    mock_asr.transcribe.return_value = (segments, LanguageCode.ENGLISH)
    mock_translator.translate.return_value = translated_segments
    
    # 调用用例
    result = generate_subtitles_use_case(
        video=test_video,
        asr_provider=mock_asr,
        translation_provider=mock_translator,
        cache_repo=mock_cache
    )
    
    # 断言
    assert result.detected_language == LanguageCode.ENGLISH
    mock_asr.transcribe.assert_called_once()
```

### 2. 集成测试（Infrastructure）

```python
# tests/integration/test_adapters.py
def test_whisper_adapter_integration():
    """测试适配器集成（真实模型）"""
    adapter = WhisperASRAdapter(model_size="tiny")
    
    segments, lang = adapter.transcribe(Path("test_audio.wav"))
    
    assert len(segments) > 0
    assert lang in [LanguageCode.ENGLISH, LanguageCode.CHINESE]
    
    # 验证返回的是领域对象
    assert isinstance(segments[0], TextSegment)
```

### 3. 端到端测试

```python
# tests/e2e/test_full_pipeline.py
def test_full_video_processing():
    """端到端测试（真实依赖）"""
    container = DependencyContainer()
    
    video = Video(path=Path("test.mp4"), duration=10.0, has_audio=True)
    
    result = generate_subtitles_use_case(
        video=video,
        asr_provider=container.get_asr("tiny"),
        translation_provider=container.get_translator(),
        cache_repo=container.cache_repo
    )
    
    assert result.translated_subtitle is not None
    assert len(result.translated_subtitle.segments) > 0
```

---

## 🔧 如何添加新功能

### 示例：添加对口型功能

#### Step 1: 在 Domain Layer 定义接口

```python
# domain/ports.py
class LipSyncProvider(Protocol):
    """对口型提供者接口"""
    
    def sync_lips(
        self,
        video: Video,
        audio: AudioSample,
        subtitle: Subtitle
    ) -> Video:
        """对口型处理"""
        ...
```

#### Step 2: 在 Application Layer 创建用例

```python
# application/use_cases/lip_sync.py
def lip_sync_use_case(
    video: Video,
    audio_track: AudioTrack,
    subtitle: Subtitle,
    lip_sync_provider: LipSyncProvider,
    cache_repo: CacheRepository,
    progress: ProgressCallback = None
) -> Video:
    """对口型用例（纯函数）"""
    
    # 检查缓存
    cache_key = calculate_cache_key(video.path, "lip_sync", {})
    if cache_repo.exists(cache_key):
        cached = cache_repo.get(cache_key)
        return Video(**cached)
    
    # 执行对口型
    synced_video = lip_sync_provider.sync_lips(
        video=video,
        audio=audio_track.audio,
        subtitle=subtitle
    )
    
    # 保存缓存
    cache_repo.set(cache_key, {
        "path": str(synced_video.path),
        "duration": synced_video.duration,
        "has_audio": synced_video.has_audio
    })
    
    return synced_video
```

#### Step 3: 在 Infrastructure Layer 实现适配器

```python
# infrastructure/adapters/lip_sync/wav2lip_adapter.py
class Wav2LipAdapter:
    """Wav2Lip 对口型适配器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
    
    def sync_lips(
        self,
        video: Video,
        audio: AudioSample,
        subtitle: Subtitle
    ) -> Video:
        """实现对口型（有副作用）"""
        # 加载模型
        model = self._load_model()
        
        # 执行对口型处理
        output_path = self._process_video(model, video, audio)
        
        return Video(
            path=output_path,
            duration=video.duration,
            has_audio=True
        )
```

#### Step 4: 在 UI 中集成

```python
# infrastructure/ui/webui.py
def process_with_lip_sync_ui(video_file, ...):
    """UI 处理函数"""
    container = DependencyContainer()
    
    # 调用用例
    result = lip_sync_use_case(
        video=video,
        audio_track=audio_track,
        subtitle=subtitle,
        lip_sync_provider=container.get_lip_sync(),  # 新增
        cache_repo=container.cache_repo
    )
    
    return str(result.path)
```

---

## 📊 依赖管理

### requirements.txt 组织

```txt
# requirements.txt

# ===== Core (Domain & Application) =====
# 无外部依赖！纯 Python 标准库

# ===== Infrastructure - ASR =====
openai-whisper>=20231117
faster-whisper>=0.10.0

# ===== Infrastructure - Translation =====
transformers>=4.35.0
torch>=2.1.0
bitsandbytes>=0.41.0

# ===== Infrastructure - TTS =====
f5-tts>=0.1.0  # IndexTTS 2.0

# ===== Infrastructure - Video =====
ffmpeg-python>=0.2.0

# ===== Infrastructure - UI =====
gradio>=4.0.0

# ===== Infrastructure - Storage =====
# 使用标准库 json, pathlib

# ===== Testing =====
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# ===== Development =====
black>=23.0.0
mypy>=1.5.0
```

---

## 🚀 快速开始

### 1. 安装

```bash
# 克隆项目
git clone <repo-url>
cd video_translator

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装 F5-TTS
pip install f5-tts
```

### 2. 运行

```bash
# 启动 WebUI
python scripts/run_webui.py

# 或使用 CLI
python infrastructure/ui/cli.py process video.mp4 \
    --whisper medium \
    --translator Qwen/Qwen2.5-7B \
    --enable-voice
```

### 3. 测试

```bash
# 运行所有测试
pytest

# 运行特定层测试
pytest tests/unit/test_domain/
pytest tests/unit/test_application/
pytest tests/integration/

# 测试覆盖率
pytest --cov=video_translator --cov-report=html
```

---

## 💡 设计原则总结

### 1. 依赖倒置原则（DIP）
- ✅ 外层依赖内层
- ✅ 内层定义接口，外层实现
- ✅ 核心业务不依赖技术细节

### 2. 单一职责原则（SRP）
- ✅ Domain: 业务规则
- ✅ Application: 用例编排
- ✅ Infrastructure: 技术实现

### 3. 开闭原则（OCP）
- ✅ 易于扩展（添加新适配器）
- ✅ 无需修改核心代码

### 4. 函数式编程
- ✅ 纯函数（Domain & Application）
- ✅ 不可变数据（frozen dataclass）
- ✅ 副作用隔离（Infrastructure）

### 5. 可测试性
- ✅ 纯函数易测试（无 mock）
- ✅ 接口易 mock
- ✅ 分层测试策略

---

## 📚 推荐阅读

- **Clean Architecture** - Robert C. Martin
- **Domain-Driven Design** - Eric Evans
- **Functional Programming in Python** - David Mertz
- **Dependency Injection in Python** - Injection Patterns

---

## 🎯 下一步

1. ✅ 理解架构层次
2. ✅ 熟悉依赖注入
3. ✅ 学习纯函数编写
4. ✅ 实践测试驱动开发（TDD）
5. ✅ 扩展新功能（如对口型）

Happy Coding! 🚀