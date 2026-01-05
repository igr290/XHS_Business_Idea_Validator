# 小红书商业调研 Agent 重新开发计划

## 一、现有系统分析

### 1.1 完整数据流程追踪

```
用户输入 → Streamlit UI → validate_business_idea_cn() → 最终报告
    ↓                    ↓
[业务创意]          [validation_worker 线程]
                         ↓
                    ┌─────────────────────────────────────┐
                    │   Step 1: 生成关键词 (LLM)           │
                    │   keyword_generator.py               │
                    │   checkpoint: 01_keywords.json       │
                    └─────────────────────────────────────┘
                         ↓
                    ┌─────────────────────────────────────┐
                    │   Step 2-3: 抓取小红书帖子            │
                    │   scrapers/xhs.py                   │
                    │   API: TikHub                       │
                    │   checkpoint: 03_xhs_posts_*.json   │
                    └─────────────────────────────────────┘
                         ↓
                    ┌─────────────────────────────────────┐
                    │   Step 4: 抓取评论                   │
                    │   scrapers/xhs.py                   │
                    │   checkpoint: 04_xhs_comments_*.json│
                    └─────────────────────────────────────┘
                         ↓
                    ┌─────────────────────────────────────┐
                    │   Step 5-6: AI分析帖子+评论          │
                    │   analyzers/xhs_analyzer.py         │
                    │   checkpoint: 06_xhs_analyses_*.json│
                    └─────────────────────────────────────┘
                         ↓
                    ┌─────────────────────────────────────┐
                    │   Step 7: 生成最终报告               │
                    │   analyzers/combined_analyzer.py    │
                    │   checkpoint: 07_final_analysis.json│
                    └─────────────────────────────────────┘
                         ↓
                    [CombinedAnalysis 对象返回 UI]
                         ↓
                    [展示结果 + AI对话功能]
```

### 1.2 核心模块职责

| 模块 | 职责 | 关键类/函数 |
|------|------|-------------|
| `validator_cn.py` | 编排整个验证流程 | `validate_business_idea_cn()` |
| `keyword_generator.py` | LLM生成搜索关键词 | `generate_keywords_cn()` |
| `xhs.py` | TikHub API调用抓取 | `scrape_xhs_search()`, `scrape_xhs_post_comments()` |
| `xhs_analyzer.py` | 单帖子AI分析 | `analyze_xhs_post()` |
| `combined_analyzer.py` | 汇总生成最终报告 | `generate_final_analysis_cn()` |
| `business_validator_ui_cn.py` | Streamlit界面 | `run_validation_with_progress()` |

### 1.3 数据模型 (Pydantic)

```python
# business_validator/models.py
class KeywordModel:
    keywords: List[str]

class XhsPostAnalysis:
    relevant: bool
    pain_points: List[str]
    solutions_mentioned: List[str]
    market_signals: List[str]
    sentiment: str  # positive/negative/neutral
    engagement_score: int  # 1-10

class CombinedAnalysis:
    overall_score: int  # 1-100
    market_validation_summary: str
    key_pain_points: List[str]
    existing_solutions: List[str]
    market_opportunities: List[str]
    platform_insights: List[PlatformInsight]
    recommendations: List[str]
```

---

## 二、现有系统痛点分析

### 2.1 架构痛点

| 痛点 | 描述 | 影响 |
|------|------|------|
| **硬编码流程** | 步骤固定在 `validator_cn.py` 中，无法动态调整 | 扩展性差 |
| **线程监控脆弱** | 通过检查文件系统来监控进度，不可靠 | 进度显示不准确 |
| **无并行处理** | 所有步骤串行执行，效率低 | 处理时间长 |
| **LLM调用分散** | 每个分析器独立调用LLM，无统一管理 | 成本高、无批处理 |
| **错误处理粗糙** | Fallback分析机制简单，信息丢失多 | 结果质量低 |
| **无智能重试** | API失败直接返回空结果 | 数据不完整 |

### 2.2 可用性痛点

| 痛点 | 描述 | 影响 |
|------|------|------|
| **无交互式调整** | 用户无法中途修改参数或方向 | 体验差 |
| **结果展示静态** | 只能看最终报告，无法深入探索 | 价值受限 |
| **AI对话简单** | 仅仅是把评论塞给GPT，无上下文管理 | 效果一般 |
| **无历史对比** | 虽然保存历史，但无对比分析 | 洞察受限 |

### 2.3 技术债务

| 债务 | 描述 | 风险 |
|------|------|------|
| **API密钥硬编码** | 密钥写在源代码里 | 安全风险 |
| **中文处理混乱** | encode/decode 到处都是 | 维护困难 |
| **无日志聚合** | logging散落各处，无结构化日志 | 调试困难 |
| **无配置验证** | 启动时不检查配置完整性 | 运行时失败 |

---

## 三、新Agent架构设计

### 3.1 设计理念

采用 **Claude Code Agent** 技术栈重新设计，实现：
- **Agent化编排**: 用Agent替代硬编码流程
- **Subagent并行**: 独立任务并行执行
- **Skills可扩展**: 用户可调用特定技能
- **MCP集成**: 外部服务标准化接入

### 3.2 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户界面 (Streamlit)                       │
│  - 输入业务创意  - 选择分析维度  - 查看进度  - 探索结果       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Main Orchestrator Agent                    │
│  职责: 理解用户意图, 制定计划, 分配任务, 汇总结果            │
│  技能: plan, delegate, aggregate, explain                    │
└─────────────────────────────────────────────────────────────┘
         ↓                ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Keyword    │  │    Scraper   │  │   Analyzer   │  │   Reporter   │
│   Subagent   │  │   Subagent   │  │   Subagent   │  │   Subagent   │
│              │  │              │  │              │  │              │
│ Skills:      │  │ Skills:      │  │ Skills:      │  │ Skills:      │
│ - generate   │  │ - search_xhs │  │ - analyze    │  │ - summarize  │
│ - refine     │  │ - get_posts  │  │ - compare    │  │ - visualize  │
│ - validate   │  │ - get_comments│ │ - cluster    │  │ - export     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
         ↓                ↓                ↓                ↓
┌─────────────────────────────────────────────────────────────┐
│                    Shared Context Store                      │
│  - keywords  - raw_posts  - analyses  - user_preferences    │
└─────────────────────────────────────────────────────────────┘
         ↓                ↓                ↓                ↓
┌─────────────────────────────────────────────────────────────┐
│                      MCP Servers                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   XHS-MCP   │  │  LLM-MCP    │  │  Storage-MCP│         │
│  │  (TikHub)   │  │  (OpenAI)   │  │  (JSON/DB)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 核心组件设计

#### 3.3.1 Main Orchestrator Agent

**类型**: `general-purpose` agent

**职责**:
1. 理解用户输入的业务创意
2. 询问澄清问题（可选）
3. 制定验证计划（需要哪些关键词、抓取范围等）
4. 分配任务给Subagents
5. 监控执行进度
6. 汇总结果并生成报告

**提示词模板**:
```
You are a Business Research Orchestrator. Your task is to help validate a business idea by:

1. Understanding the business idea from the user
2. Asking clarifying questions if needed (target market, location, budget)
3. Delegating tasks to specialist agents:
   - KeywordGenerator: Generate search keywords
   - Scraper: Collect data from Xiaohongshu
   - Analyzer: Analyze collected content
   - Reporter: Generate insights and recommendations

4. Monitoring progress and handling errors
5. Synthesizing results into actionable insights

Business Idea: {business_idea}
User Context: {user_context}

Plan your approach and begin execution.
```

#### 3.3.2 KeywordGenerator Subagent

**类型**: `general-purpose` agent

**Skills**:
| Skill | 描述 | 参数 |
|-------|------|------|
| `/generate-keywords` | 根据业务创意生成关键词 | business_idea, count=3 |
| `/refine-keywords` | 根据已有结果优化关键词 | existing_keywords, feedback |
| `/validate-keywords` | 检查关键词是否适合搜索 | keywords |

**MCP调用**: `LLM-MCP` 用于生成关键词

#### 3.3.3 Scraper Subagent

**类型**: `general-purpose` agent

**Skills**:
| Skill | 描述 | 参数 |
|-------|------|------|
| `/search-posts` | 搜索小红书帖子 | keywords, pages=2 |
| `/get-comments` | 获取帖子评论 | post_ids, limit=50 |
| `/get-stats` | 获取帖子统计数据 | post_ids |

**MCP调用**: `XHS-MCP` (TikHub API)

#### 3.3.4 Analyzer Subagent

**类型**: `general-purpose` agent

**Skills**:
| Skill | 描述 | 参数 |
|-------|------|------|
| `/analyze-post` | 分析单个帖子 | post_data, business_idea |
| `/batch-analyze` | 批量分析帖子 | posts_data, business_idea |
| `/cluster-insights` | 聚类分析洞察 | analyses |
| `/compare-sentiment` | 对比情感分析 | analyses |

**MCP调用**: `LLM-MCP` 用于分析

#### 3.3.5 Reporter Subagent

**类型**: `general-purpose` agent

**Skills**:
| Skill | 描述 | 参数 |
|-------|------|------|
| `/summarize` | 生成执行摘要 | all_analyses |
| `/score-idea` | 计算验证评分 | analyses |
| `/visualize` | 生成可视化数据 | data |
| `/export-report` | 导出报告 | format='json/html' |

**MCP调用**: `LLM-MCP` 用于生成摘要

### 3.4 MCP Server 设计

#### 3.4.1 XHS-MCP Server

```python
# mcp_servers/xhs_server.py
"""
小红书数据获取 MCP Server
提供统一的TikHub API调用接口
"""

from mcp import Server
from business_validator.scrapers.xhs import scrape_xhs_search, scrape_xhs_post_comments

server = Server("xhs-server")

@server.tool()
def search_notes(keyword: str, page: int = 1) -> list:
    """搜索小红书笔记"""
    return scrape_xhs_search(keyword, page)

@server.tool()
def get_comments(note_id: str) -> list:
    """获取笔记评论"""
    return scrape_xhs_post_comments(note_id)

@server.tool()
def get_note_stats(note_id: str) -> dict:
    """获取笔记统计数据"""
    # 实现获取统计数据
    pass
```

#### 3.4.2 LLM-MCP Server

```python
# mcp_servers/llm_server.py
"""
LLM调用 MCP Server
提供统一的LLM调用接口，支持结构化输出
"""

from mcp import Server
from SimplerLLM.langauge.llm import LLM, LLMProvider

server = Server("llm-server")

@server.tool()
def generate_structured(prompt: str, schema: dict) -> dict:
    """生成结构化输出"""
    llm = LLM.create(provider=LLMProvider.OPENAI)
    return llm.generate_json_with_pydantic(prompt, schema)

@server.tool()
def generate_text(prompt: str, max_tokens: int = 1000) -> str:
    """生成文本"""
    llm = LLM.create(provider=LLMProvider.OPENAI)
    return llm.generate_text(prompt, max_tokens=max_tokens)
```

#### 3.4.3 Storage-MCP Server

```python
# mcp_servers/storage_server.py
"""
数据存储 MCP Server
提供统一的数据持久化接口
"""

from mcp import Server
import json
from pathlib import Path

server = Server("storage-server")

@server.tool()
def save_checkpoint(data: dict, run_id: str, step: str) -> str:
    """保存检查点"""
    path = Path(f"validation_data/{run_id}/{step}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))
    return str(path)

@server.tool()
def load_checkpoint(run_id: str, step: str) -> dict:
    """加载检查点"""
    path = Path(f"validation_data/{run_id}/{step}.json")
    return json.loads(path.read_text())
```

---

## 四、实施计划

### Phase 1: 基础架构搭建 (Week 1)

| 任务 | 描述 | 交付物 |
|------|------|--------|
| 1.1 | 创建Agent框架目录结构 | 新目录结构 |
| 1.2 | 实现MCP Servers | XHS-MCP, LLM-MCP, Storage-MCP |
| 1.3 | 创建Base Agent类 | agent/base_agent.py |
| 1.4 | 实现Context Store | agent/context_store.py |
| 1.5 | 配置管理重构 | agent/config.py |

```bash
# 新目录结构
agents/
├── __init__.py
├── base_agent.py          # Agent基类
├── orchestrator.py        # 主编排Agent
├── subagents/
│   ├── __init__.py
│   ├── keyword_agent.py
│   ├── scraper_agent.py
│   ├── analyzer_agent.py
│   └── reporter_agent.py
├── context_store.py       # 共享上下文
├── skills/
│   ├── __init__.py
│   ├── keyword_skills.py
│   ├── scraper_skills.py
│   ├── analyzer_skills.py
│   └── reporter_skills.py
└── config.py             # Agent配置

mcp_servers/
├── xhs_server.py         # XHS MCP Server
├── llm_server.py         # LLM MCP Server
└── storage_server.py     # Storage MCP Server
```

### Phase 2: Subagents实现 (Week 2)

| 任务 | 描述 | 交付物 |
|------|------|--------|
| 2.1 | KeywordGenerator Agent | agents/subagents/keyword_agent.py |
| 2.2 | Scraper Agent | agents/subagents/scraper_agent.py |
| 2.3 | Analyzer Agent | agents/subagents/analyzer_agent.py |
| 2.4 | Reporter Agent | agents/subagents/reporter_agent.py |
| 2.5 | Skills实现 | agents/skills/*.py |

### Phase 3: Orchestrator实现 (Week 3)

| 任务 | 描述 | 交付物 |
|------|------|--------|
| 3.1 | 主编排Agent | agents/orchestrator.py |
| 3.2 | 任务分配逻辑 | delegation机制 |
| 3.3 | 进度监控 | progress tracking |
| 3.4 | 错误处理 | error handling & recovery |
| 3.5 | 结果汇总 | result aggregation |

### Phase 4: UI集成 (Week 4)

| 任务 | 描述 | 交付物 |
|------|------|--------|
| 4.1 | 新Streamlit UI | agent_ui.py |
| 4.2 | 实时进度展示 | WebSocket/轮询 |
| 4.3 | 交互式参数调整 | 动态配置 |
| 4.4 | 结果探索界面 | 深度分析视图 |
| 4.5 | AI对话增强 | 上下文管理 |

### Phase 5: 测试与优化 (Week 5)

| 任务 | 描述 | 交付物 |
|------|------|--------|
| 5.1 | 单元测试 | tests/ |
| 5.2 | 集成测试 | 端到端测试 |
| 5.3 | 性能优化 | 并行处理 |
| 5.4 | 错误恢复测试 | 异常场景 |
| 5.5 | 文档完善 | README, API文档 |

---

## 五、关键技术实现

### 5.1 Agent基类设计

```python
# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class BaseAgent(ABC):
    """所有Agent的基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.context_store = config.get('context_store')

    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务的核心方法"""
        pass

    async def use_mcp(self, server: str, tool: str, **kwargs) -> Any:
        """调用MCP Server的工具"""
        # MCP调用实现
        pass

    async def delegate_to(self, agent_name: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """委托任务给子Agent"""
        # 委托实现
        pass

    def update_progress(self, step: str, progress: float, message: str):
        """更新进度"""
        if self.context_store:
            self.context_store.set_progress(self.name, step, progress, message)
```

### 5.2 编排Agent实现

```python
# agents/orchestrator.py
from .base_agent import BaseAgent
from .subagents import KeywordAgent, ScraperAgent, AnalyzerAgent, ReporterAgent

class OrchestratorAgent(BaseAgent):
    """主编排Agent"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("orchestrator", config)
        self.subagents = {
            'keyword': KeywordAgent(config),
            'scraper': ScraperAgent(config),
            'analyzer': AnalyzerAgent(config),
            'reporter': ReporterAgent(config)
        }

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        business_idea = context.get('business_idea')

        # Step 1: 生成关键词
        self.update_progress('generating_keywords', 0.1, "正在生成关键词...")
        keywords_result = await self.delegate_to(
            'keyword',
            'generate',
            {'business_idea': business_idea, 'count': 3}
        )
        keywords = keywords_result['keywords']
        self.context_store.set(context['run_id'], 'keywords', keywords)

        # Step 2: 抓取数据
        self.update_progress('scraping', 0.3, "正在抓取小红书数据...")
        posts_result = await self.delegate_to(
            'scraper',
            'search_posts',
            {'keywords': keywords, 'pages': 2}
        )
        posts = posts_result['posts']

        # Step 3: 分析数据
        self.update_progress('analyzing', 0.6, "正在AI分析...")
        analysis_result = await self.delegate_to(
            'analyzer',
            'batch_analyze',
            {'posts': posts, 'business_idea': business_idea}
        )
        analyses = analysis_result['analyses']

        # Step 4: 生成报告
        self.update_progress('reporting', 0.9, "正在生成报告...")
        report = await self.delegate_to(
            'reporter',
            'summarize',
            {'analyses': analyses, 'business_idea': business_idea}
        )

        self.update_progress('complete', 1.0, "验证完成!")
        return report
```

### 5.3 并行处理示例

```python
# 并行抓取多个关键词
async def parallel_scrape(self, keywords: list) -> list:
    tasks = [
        self.delegate_to('scraper', 'search_posts', {'keyword': kw})
        for kw in keywords
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

---

## 六、预期收益

### 6.1 架构收益

| 指标 | 现状 | 目标 | 改进 |
|------|------|------|------|
| 模块化程度 | 低 (紧耦合) | 高 (Agent独立) | +80% |
| 扩展性 | 差 (硬编码) | 好 (可插拔) | +90% |
| 并行处理 | 无 | 有 | 速度+300% |
| 错误恢复 | 粗糙 | 智能重试 | 可靠性+60% |

### 6.2 功能收益

| 功能 | 现状 | 新系统 |
|------|------|--------|
| 交互式调整 | 无 | 支持中途修改 |
| 智能关键词 | 固定生成 | 动态优化 |
| 深度分析 | 单次报告 | 可探索洞察 |
| 多平台支持 | 仅XHS | 易扩展 |

### 6.3 开发收益

| 方面 | 改进 |
|------|------|
| 代码复用 | Agent可复用 |
| 测试难度 | 降低 (Agent独立测试) |
| 维护成本 | 降低 (解耦) |
| 新人上手 | 降低 (清晰架构) |

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Agent通信复杂 | 开发周期长 | 使用成熟框架模板 |
| 状态管理困难 | 数据不一致 | 使用Redis做上下文存储 |
| 并发控制问题 | 资源竞争 | 使用异步锁 |
| LLM成本增加 | 运营成本 | 添加批处理和缓存 |

---

## 八、下一步行动

1. **确认计划**: 与团队讨论确认技术方案
2. **环境准备**: 设置开发环境和依赖
3. **MVP开发**: 先实现Keyword和Scraper Agent
4. **迭代优化**: 根据测试结果逐步完善

---

*文档版本: 1.0*
*创建日期: 2025-01-02*
*作者: Claude Code Agent Development Team*
