# 小红书商业调研 Agent 系统技术规格说明书

## 文档信息

| 项目 | 内容 |
|------|------|
| 文档版本 | 1.0 |
| 创建日期 | 2025-01-02 |
| 作者 | Claude Code Development Team |
| 审核状态 | 待审核 |

---

## 1. 技术选型

### 1.1 核心技术栈

| 技术层 | 选型 | 版本 | 说明 |
|--------|------|------|------|
| **语言** | Python | 3.9+ | 主要开发语言 |
| **异步框架** | asyncio | 内置 | 异步任务处理 |
| **Agent框架** | Claude Code Agent SDK | Latest | Agent编排与管理 |
| **LLM** | OpenAI API | GPT-4o | 结构化输出 |
| **数据源API** | TikHub API | v1 | 小红书数据 |
| **UI框架** | Streamlit | 1.31+ | Web界面 |
| **数据验证** | Pydantic | 2.5+ | 数据模型 |
| **MCP协议** | MCP SDK | Latest | 服务间通信 |

### 1.2 依赖库清单

```txt
# 核心依赖
streamlit==1.31.0
pydantic==2.5.3
requests==2.31.0
aiohttp==3.9.0
asyncio==3.4.3

# LLM相关
openai==1.12.0
instructor==1.3.0

# Claude Code Agent
anthropic-sdk>=0.5.0

# MCP相关
mcp>=0.1.0

# 数据处理
pandas==2.1.4
numpy==1.26.3

# 可视化
plotly==5.18.0

# 工具库
python-dotenv==1.0.0
loguru==0.7.2
tenacity==8.2.3

# 存储
redis==5.0.1  # 上下文存储
```

### 1.3 技术选型理由

| 选型 | 理由 |
|------|------|
| **asyncio** | 原生协程支持，适合I/O密集型任务 |
| **Claude Code Agent** | 提供完整的Agent编排能力 |
| **MCP协议** | 标准化服务通信，易于扩展 |
| **Pydantic V2** | 类型安全，运行时验证 |
| **Redis** | 高性能上下文共享，支持过期策略 |
| **Streamlit** | 快速原型，Python原生支持 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Presentation Layer                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Streamlit Web UI (agent_ui.py)                   │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │ │
│  │  │ Input Form │ │ Progress   │ │ Results    │ │ Chat AI    │       │ │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                            Agent Orchestration Layer                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                  Orchestrator Agent (orchestrator.py)               │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────────┐  │ │
│  │  │   Task       │ │   Context    │ │      Progress              │  │ │
│  │  │   Planner    │ │   Manager    │ │      Monitor               │  │ │
│  │  └──────────────┘ └──────────────┘ └────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    ↓                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        Subagent Pool                                │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │ │
│  │  │  Keyword    │ │   Scraper   │ │  Analyzer   │ │   Reporter  │  │ │
│  │  │   Agent     │ │   Agent     │ │   Agent     │ │   Agent     │  │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                              Skills Layer                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │ Keyword      │ │ Scraper      │ │ Analyzer     │ │ Reporter     │   │
│  │ Skills       │ │ Skills       │ │ Skills       │ │ Skills       │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                                MCP Layer                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │   XHS-MCP    │ │   LLM-MCP    │ │  Storage-MCP │ │  Cache-MCP   │   │
│  │   Server     │ │   Server     │ │   Server     │ │   Server     │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                            External Services                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  TikHub API  │ │  OpenAI API  │ │    Redis     │ │  File System │   │
│  │  (小红书)    │ │   (LLM)      │ │  (Context)   │ │  (Checkpoints)│  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构设计

```
Business_Idea_Validator_Agent/
├── agents/                          # Agent模块
│   ├── __init__.py
│   ├── base_agent.py               # Agent基类
│   ├── orchestrator.py             # 主编排Agent
│   ├── config.py                   # Agent配置
│   ├── context_store.py            # 上下文管理
│   ├── subagents/                  # 子Agent
│   │   ├── __init__.py
│   │   ├── keyword_agent.py        # 关键词生成Agent
│   │   ├── scraper_agent.py        # 数据抓取Agent
│   │   ├── analyzer_agent.py       # 数据分析Agent
│   │   └── reporter_agent.py       # 报告生成Agent
│   └── skills/                     # Skills定义
│       ├── __init__.py
│       ├── keyword_skills.py       # 关键词相关Skills
│       ├── scraper_skills.py       # 抓取相关Skills
│       ├── analyzer_skills.py      # 分析相关Skills
│       └── reporter_skills.py      # 报告相关Skills
│
├── mcp_servers/                    # MCP服务器
│   ├── __init__.py
│   ├── base_server.py              # MCP基类
│   ├── xhs_server.py               # 小红书MCP服务
│   ├── llm_server.py               # LLM MCP服务
│   ├── storage_server.py           # 存储MCP服务
│   └── cache_server.py             # 缓存MCP服务
│
├── models/                         # 数据模型
│   ├── __init__.py
│   ├── agent_models.py             # Agent相关模型
│   ├── business_models.py          # 业务模型(复用现有)
│   └── mcp_models.py               # MCP通信模型
│
├── ui/                             # 用户界面
│   ├── __init__.py
│   ├── agent_ui.py                 # 新Agent UI
│   ├── components/                 # UI组件
│   │   ├── __init__.py
│   │   ├── progress_monitor.py    # 进度监控组件
│   │   ├── result_viewer.py       # 结果展示组件
│   │   └── chat_interface.py      # AI对话组件
│   └── utils/
│       ├── __init__.py
│       └── formatters.py          # 格式化工具
│
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── logger.py                  # 日志配置
│   ├── config.py                  # 全局配置
│   ├── retry.py                   # 重试机制
│   └── validators.py              # 验证器
│
├── tests/                          # 测试
│   ├── __init__.py
│   ├── test_agents/               # Agent测试
│   ├── test_mcp_servers/          # MCP测试
│   └── test_integration/          # 集成测试
│
├── scripts/                        # 脚本
│   ├── start_mcp_servers.py       # 启动MCP服务
│   ├── run_agent.py               # 运行Agent
│   └── migrate_data.py            # 数据迁移
│
├── config/                         # 配置文件
│   ├── default.yaml               # 默认配置
│   ├── development.yaml           # 开发配置
│   └── production.yaml            # 生产配置
│
├── docs/                           # 文档
│   ├── TECHNICAL_SPECIFICATION.md # 本文档
│   ├── AGENT_REDEVELOPMENT_PLAN.md
│   └── API.md                     # API文档
│
├── requirements.txt                # 依赖
├── .env.example                   # 环境变量示例
├── pyproject.toml                 # 项目配置
└── README.md                      # 项目说明
```

---

## 3. 核心模块设计

### 3.1 BaseAgent 基类设计

```python
# agents/base_agent.py
"""
所有Agent的抽象基类
定义Agent的生命周期和基本行为
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from agents.context_store import ContextStore
from agents.config import AgentConfig
from models.agent_models import AgentState, TaskResult, ProgressUpdate


class AgentStatus(Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMetrics:
    """Agent运行指标"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    mcp_calls: int = 0
    llm_calls: int = 0


class BaseAgent(ABC):
    """
    Agent抽象基类

    所有Agent必须继承此类并实现execute方法

    Attributes:
        name: Agent名称
        config: Agent配置
        context_store: 共享上下文存储
        status: 当前状态
        metrics: 运行指标
    """

    def __init__(
        self,
        name: str,
        config: AgentConfig,
        context_store: ContextStore,
        mcp_clients: Dict[str, Any]
    ):
        self.name = name
        self.config = config
        self.context_store = context_store
        self.mcp_clients = mcp_clients
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"agent.{name}")

        # 进度回调函数
        self._progress_callback: Optional[Callable[[ProgressUpdate], None]] = None

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> TaskResult:
        """
        执行任务的核心抽象方法

        Args:
            task: 任务描述
            context: 执行上下文
            **kwargs: 额外参数

        Returns:
            TaskResult: 任务执行结果
        """
        pass

    async def use_mcp(
        self,
        server_name: str,
        tool_name: str,
        **kwargs
    ) -> Any:
        """
        调用MCP服务器工具

        Args:
            server_name: MCP服务器名称
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果

        Raises:
            MCPConnectionError: MCP连接失败
            MCPToolError: 工具执行失败
        """
        self.metrics.mcp_calls += 1

        if server_name not in self.mcp_clients:
            raise ValueError(f"MCP server '{server_name}' not available")

        client = self.mcp_clients[server_name]
        try:
            result = await client.call_tool(tool_name, **kwargs)
            self.logger.debug(f"MCP call: {server_name}.{tool_name} = {type(result)}")
            return result
        except Exception as e:
            self.logger.error(f"MCP call failed: {server_name}.{tool_name}: {e}")
            raise

    async def use_llm(
        self,
        prompt: str,
        response_model: Optional[type] = None,
        max_tokens: int = 2000
    ) -> Any:
        """
        调用LLM生成内容

        Args:
            prompt: 提示词
            response_model: Pydantic响应模型(结构化输出)
            max_tokens: 最大token数

        Returns:
            LLM生成结果
        """
        self.metrics.llm_calls += 1

        llm_client = self.mcp_clients.get('llm')
        if not llm_client:
            raise RuntimeError("LLM MCP client not available")

        if response_model:
            # 结构化输出
            return await llm_client.generate_structured(
                prompt=prompt,
                schema=response_model.model_json_schema()
            )
        else:
            # 文本输出
            return await llm_client.generate_text(
                prompt=prompt,
                max_tokens=max_tokens
            )

    async def delegate_to(
        self,
        agent_name: str,
        task: str,
        context: Dict[str, Any],
        timeout: float = 300.0
    ) -> TaskResult:
        """
        委托任务给其他Agent

        Args:
            agent_name: 目标Agent名称
            task: 任务描述
            context: 任务上下文
            timeout: 超时时间(秒)

        Returns:
            TaskResult: 委托任务的执行结果
        """
        self.logger.info(f"Delegating to {agent_name}: {task}")

        # 从context_store获取Agent实例
        target_agent = self.context_store.get_agent(agent_name)
        if not target_agent:
            raise ValueError(f"Agent '{agent_name}' not found in context store")

        # 异步执行并等待结果
        try:
            result = await asyncio.wait_for(
                target_agent.execute(task, context),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"Delegation to {agent_name} timed out")
            raise TimeoutError(f"Agent {agent_name} did not respond within {timeout}s")

    def update_progress(
        self,
        step: str,
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        更新执行进度

        Args:
            step: 当前步骤名称
            progress: 进度百分比(0-1)
            message: 进度消息
            details: 额外详情
        """
        update = ProgressUpdate(
            agent_name=self.name,
            step=step,
            progress=progress,
            message=message,
            timestamp=datetime.now(),
            details=details or {}
        )

        # 保存到context_store
        run_id = self.config.get('run_id')
        if run_id:
            self.context_store.set_progress(run_id, update)

        # 调用回调(如果有)
        if self._progress_callback:
            self._progress_callback(update)

        self.logger.debug(f"Progress: {step} - {progress*100:.1f}% - {message}")

    def set_progress_callback(
        self,
        callback: Callable[[ProgressUpdate], None]
    ):
        """设置进度回调函数"""
        self._progress_callback = callback

    async def save_checkpoint(
        self,
        run_id: str,
        step: str,
        data: Dict[str, Any]
    ):
        """
        保存检查点

        Args:
            run_id: 运行ID
            step: 步骤名称
            data: 要保存的数据
        """
        await self.use_mcp(
            'storage',
            'save_checkpoint',
            run_id=run_id,
            step=step,
            data=data
        )
        self.logger.info(f"Checkpoint saved: {run_id}/{step}")

    async def load_checkpoint(
        self,
        run_id: str,
        step: str
    ) -> Optional[Dict[str, Any]]:
        """
        加载检查点

        Args:
            run_id: 运行ID
            step: 步骤名称

        Returns:
            检查点数据，如果不存在返回None
        """
        try:
            data = await self.use_mcp(
                'storage',
                'load_checkpoint',
                run_id=run_id,
                step=step
            )
            self.logger.info(f"Checkpoint loaded: {run_id}/{step}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint {run_id}/{step}: {e}")
            return None

    def get_status(self) -> AgentStatus:
        """获取当前状态"""
        return self.status

    def get_metrics(self) -> AgentMetrics:
        """获取运行指标"""
        return self.metrics

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: Agent是否健康
        """
        try:
            # 检查MCP连接
            for server_name, client in self.mcp_clients.items():
                await client.ping()
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
```

### 3.2 Orchestrator Agent 设计

```python
# agents/orchestrator.py
"""
主编排Agent
负责任务规划、分配和结果汇总
"""

from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from agents.base_agent import BaseAgent, AgentStatus
from agents.config import AgentConfig
from agents.context_store import ContextStore
from models.agent_models import (
    TaskResult, ExecutionPlan, OrchestratorState,
    ValidationPlan, PlanStep
)
from utils.logger import get_logger


class OrchestratorAgent(BaseAgent):
    """
    主编排Agent

    职责:
    1. 理解用户意图
    2. 生成执行计划
    3. 分配任务给Subagents
    4. 监控执行进度
    5. 汇总结果
    6. 处理错误和重试
    """

    def __init__(
        self,
        config: AgentConfig,
        context_store: ContextStore,
        mcp_clients: Dict[str, Any]
    ):
        super().__init__("orchestrator", config, context_store, mcp_clients)
        self.logger = get_logger("orchestrator")

        # 初始化Subagents
        self.subagents = {
            'keyword': None,  # KeywordAgent实例
            'scraper': None,  # ScraperAgent实例
            'analyzer': None,  # AnalyzerAgent实例
            'reporter': None,  # ReporterAgent实例
        }

        # 执行状态
        self.state = OrchestratorState(
            current_run_id=None,
            current_plan=None,
            completed_steps=[],
            failed_steps=[],
            start_time=None,
            end_time=None
        )

    def register_subagent(self, name: str, agent: BaseAgent):
        """注册Subagent"""
        if name not in self.subagents:
            raise ValueError(f"Unknown subagent: {name}")
        self.subagents[name] = agent
        self.logger.info(f"Registered subagent: {name}")

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> TaskResult:
        """
        执行编排任务

        Args:
            task: 任务类型 (validate/analyze/compare)
            context: 任务上下文，包含:
                - business_idea: 业务创意
                - user_preferences: 用户偏好
                - run_id: 运行ID

        Returns:
            TaskResult: 编排结果
        """
        self.status = AgentStatus.RUNNING
        self.state.start_time = datetime.now()

        # 初始化运行上下文
        run_id = context.get('run_id')
        business_idea = context.get('business_idea')

        if not run_id:
            run_id = self._generate_run_id(business_idea)
            context['run_id'] = run_id

        self.state.current_run_id = run_id

        try:
            # Step 1: 理解任务并生成计划
            self.update_progress('planning', 0.05, "正在分析任务并生成执行计划...")
            plan = await self._create_execution_plan(task, context)

            # Step 2: 询问澄清问题(如果需要)
            if plan.needs_clarification:
                clarifications = await self._ask_clarifications(plan.questions, context)
                context.update(clarifications)
                # 重新生成计划
                plan = await self._create_execution_plan(task, context)

            # Step 3: 执行计划
            self.update_progress('execution', 0.1, "开始执行验证流程...")
            execution_result = await self._execute_plan(plan, context)

            # Step 4: 汇总结果
            self.update_progress('finalizing', 0.95, "正在汇总分析结果...")
            final_report = await self._finalize_results(execution_result, context)

            self.state.end_time = datetime.now()
            self.status = AgentStatus.COMPLETED

            return TaskResult(
                success=True,
                data=final_report,
                run_id=run_id,
                execution_time=(self.state.end_time - self.state.start_time).total_seconds()
            )

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}", exc_info=True)
            self.status = AgentStatus.FAILED
            self.state.end_time = datetime.now()

            return TaskResult(
                success=False,
                error=str(e),
                run_id=run_id,
                execution_time=(self.state.end_time - self.state.start_time).total_seconds()
            )

    async def _create_execution_plan(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        创建执行计划

        使用LLM分析任务并生成详细的执行计划
        """
        business_idea = context.get('business_idea')
        user_prefs = context.get('user_preferences', {})

        prompt = f"""
        You are a business research planning expert. Create a detailed execution plan for:

        Task: {task}
        Business Idea: {business_idea}
        User Preferences: {user_prefs}

        Available Subagents:
        - keyword: Generate and refine search keywords
        - scraper: Collect data from Xiaohongshu
        - analyzer: Analyze collected content with AI
        - reporter: Generate insights and reports

        Create an optimal execution plan considering:
        1. Which subagents to use and in what order
        2. Which tasks can run in parallel
        3. What data flows between steps
        4. Fallback strategies for each step

        Return a structured execution plan.
        """

        # 使用LLM生成计划
        plan_schema = ExecutionPlan.model_json_schema()
        plan = await self.use_llm(prompt, response_model=ExecutionPlan)

        self.logger.info(f"Generated execution plan with {len(plan.steps)} steps")
        self.state.current_plan = plan

        return plan

    async def _ask_clarifications(
        self,
        questions: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        询问用户澄清问题
        """
        # 在实际实现中，这里会暂停执行，等待用户响应
        # 简化版本返回默认值
        return {}

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行计划

        按照计划步骤执行，支持并行和错误恢复
        """
        results = {}
        run_id = context['run_id']

        for step in plan.steps:
            step_key = f"step_{step.order}"
            self.state.completed_steps.append(step_key)

            # 检查是否可以并行执行
            parallel_tasks = self._get_parallel_tasks(step, plan.steps)
            if len(parallel_tasks) > 1:
                self.logger.info(f"Executing {len(parallel_tasks)} tasks in parallel")
                step_results = await self._execute_parallel(parallel_tasks, context)
                for i, result in enumerate(step_results):
                    results[f"{step_key}_{i}"] = result
            else:
                # 串行执行
                self.update_progress(
                    step.name,
                    step.progress_weight,
                    f"执行: {step.description}"
                )

                # 检查是否有检查点
                checkpoint_data = await self.load_checkpoint(run_id, step.name)
                if checkpoint_data and plan.resume_from_checkpoint:
                    self.logger.info(f"Resuming from checkpoint: {step.name}")
                    results[step_key] = checkpoint_data
                    continue

                # 执行任务
                try:
                    result = await self._execute_step(step, context)
                    results[step_key] = result

                    # 保存检查点
                    await self.save_checkpoint(run_id, step.name, result)

                except Exception as e:
                    self.logger.error(f"Step {step.name} failed: {e}")

                    # 尝试恢复
                    if step.fallback_strategy:
                        result = await self._execute_fallback(step, e, context)
                        results[step_key] = result
                    else:
                        raise

        return results

    async def _execute_step(
        self,
        step: PlanStep,
        context: Dict[str, Any]
    ) -> Any:
        """
        执行单个步骤
        """
        agent = self.subagents.get(step.agent)
        if not agent:
            raise ValueError(f"Subagent '{step.agent}' not available")

        # 委托给Subagent
        result = await self.delegate_to(
            step.agent,
            step.task,
            context,
            timeout=step.timeout
        )

        # 更新上下文
        if step.output_key:
            context[step.output_key] = result

        return result

    async def _execute_parallel(
        self,
        steps: List[PlanStep],
        context: Dict[str, Any]
    ) -> List[Any]:
        """
        并行执行多个步骤
        """
        tasks = [
            self._execute_step(step, context.copy())
            for step in steps
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _get_parallel_tasks(
        self,
        current_step: PlanStep,
        all_steps: List[PlanStep]
    ) -> List[PlanStep]:
        """
        获取可以并行执行的任务
        """
        # 简化版本：只返回当前步骤
        # 实际实现需要分析依赖关系
        return [current_step]

    async def _execute_fallback(
        self,
        step: PlanStep,
        error: Exception,
        context: Dict[str, Any]
    ) -> Any:
        """
        执行失败后的恢复策略
        """
        self.logger.warning(f"Executing fallback for {step.name}")

        fallback_type = step.fallback_strategy.get('type')

        if fallback_type == 'retry':
            # 重试
            retries = step.fallback_strategy.get('retries', 3)
            for i in range(retries):
                try:
                    await asyncio.sleep(step.fallback_strategy.get('delay', 2))
                    return await self._execute_step(step, context)
                except Exception as e:
                    if i == retries - 1:
                        raise

        elif fallback_type == 'skip':
            # 跳过此步骤
            return None

        elif fallback_type == 'default_value':
            # 使用默认值
            return step.fallback_strategy.get('default_value')

        else:
            raise error

    async def _finalize_results(
        self,
        execution_results: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        汇总最终结果
        """
        # 调用Reporter Agent生成最终报告
        report_result = await self.delegate_to(
            'reporter',
            'generate_final_report',
            {
                'business_idea': context.get('business_idea'),
                'execution_results': execution_results,
                'run_id': context.get('run_id')
            }
        )

        return report_result.data

    def _generate_run_id(self, business_idea: str) -> str:
        """生成运行ID"""
        import hashlib
        from datetime import datetime

        idea_hash = hashlib.md5(business_idea.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{business_idea[:20]}_{timestamp}_{idea_hash}"

    def get_state(self) -> OrchestratorState:
        """获取当前编排状态"""
        return self.state
```

### 3.3 Keyword Agent 设计

```python
# agents/subagents/keyword_agent.py
"""
关键词生成Agent
负责生成和优化搜索关键词
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from agents.config import AgentConfig
from agents.context_store import ContextStore
from models.agent_models import TaskResult
from models.business_models import KeywordModel


class KeywordRefinement(BaseModel):
    """关键词优化结果"""
    original_keywords: List[str] = Field(description="原始关键词")
    refined_keywords: List[str] = Field(description="优化后的关键词")
    refinement_reason: str = Field(description="优化原因")
    suggested_additions: List[str] = Field(description="建议添加的关键词")


class KeywordAgent(BaseAgent):
    """
    关键词生成Agent

    Skills:
    - generate_keywords: 生成初始关键词
    - refine_keywords: 根据反馈优化关键词
    - validate_keywords: 验证关键词质量
    """

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> TaskResult:
        """
        执行关键词任务

        Args:
            task: 任务类型 (generate/refine/validate)
            context: 上下文，包含:
                - business_idea: 业务创意
                - count: 生成数量
                - existing_keywords: 现有关键词(refine时)
                - feedback: 反馈信息(refine时)
        """
        business_idea = context.get('business_idea')
        count = kwargs.get('count', 3)

        try:
            if task == 'generate':
                result = await self._generate_keywords(business_idea, count)
            elif task == 'refine':
                existing = context.get('existing_keywords', [])
                feedback = context.get('feedback', '')
                result = await self._refine_keywords(existing, feedback, business_idea)
            elif task == 'validate':
                keywords = context.get('keywords', [])
                result = await self._validate_keywords(keywords, business_idea)
            else:
                raise ValueError(f"Unknown task: {task}")

            return TaskResult(
                success=True,
                data=result,
                agent_name=self.name
            )

        except Exception as e:
            self.logger.error(f"Keyword task failed: {e}")
            return TaskResult(
                success=False,
                error=str(e),
                agent_name=self.name
            )

    async def _generate_keywords(
        self,
        business_idea: str,
        count: int
    ) -> Dict[str, Any]:
        """生成关键词"""
        self.update_progress('generating', 0.3, "正在生成搜索关键词...")

        prompt = f"""
        你是市场调研专家。请为以下业务创意生成 {count} 个搜索关键词：

        业务创意：{business_idea}

        要求：
        1. 关键词应该是小红书用户会搜索的短语
        2. 每个关键词3-6个字
        3. 覆盖不同角度：产品名、用途、场景、人群等
        4. 返回中文关键词
        5. 考虑地域信息(如果业务创意中包含地点)

        示例：
        - 输入："在深圳卖陈皮"
        - 输出：["新会陈皮深圳", "陈皮茶深圳", "深圳陈皮店", "陈皮养生", "深圳特产陈皮"]
        """

        self.update_progress('generating', 0.6, "正在调用LLM生成...")

        result = await self.use_llm(
            prompt,
            response_model=KeywordModel
        )

        self.update_progress('generating', 1.0, f"生成了 {len(result.keywords)} 个关键词")

        return {
            'keywords': result.keywords,
            'count': len(result.keywords),
            'business_idea': business_idea
        }

    async def _refine_keywords(
        self,
        existing_keywords: List[str],
        feedback: str,
        business_idea: str
    ) -> Dict[str, Any]:
        """优化关键词"""
        self.update_progress('refining', 0.5, "正在优化关键词...")

        prompt = f"""
        作为市场调研专家，请根据反馈优化以下关键词：

        原始关键词：{existing_keywords}
        用户反馈：{feedback}
        业务创意：{business_idea}

        请提供：
        1. 优化后的关键词列表
        2. 优化的原因
        3. 建议额外添加的关键词
        """

        result = await self.use_llm(
            prompt,
            response_model=KeywordRefinement
        )

        self.update_progress('refining', 1.0, "关键词优化完成")

        return {
            'original_keywords': existing_keywords,
            'refined_keywords': result.refined_keywords,
            'reason': result.refinement_reason,
            'suggested_additions': result.suggested_additions
        }

    async def _validate_keywords(
        self,
        keywords: List[str],
        business_idea: str
    ) -> Dict[str, Any]:
        """验证关键词质量"""
        self.update_progress('validating', 0.5, "正在验证关键词...")

        validation_prompt = f"""
        验证以下关键词是否适合用于搜索关于"{business_idea}"的小红书内容：

        关键词：{keywords}

        评估标准：
        1. 相关性：与业务创意的相关程度 (1-10)
        2. 搜索量：可能的搜索热度 (1-10)
        3. 精准度：返回结果的精准程度 (1-10)
        4. 建议：是否保留

        返回每个关键词的评估结果。
        """

        # 简化版本返回基础验证
        validation_results = []
        for kw in keywords:
            validation_results.append({
                'keyword': kw,
                'valid': len(kw) >= 2 and len(kw) <= 10,
                'score': 8,
                'suggestion': 'keep'
            })

        self.update_progress('validating', 1.0, "关键词验证完成")

        return {
            'validation_results': validation_results,
            'valid_count': sum(1 for v in validation_results if v['valid'])
        }
```

### 3.4 Scraper Agent 设计

```python
# agents/subagents/scraper_agent.py
"""
数据抓取Agent
负责从小红书抓取帖子和评论
"""

from typing import Dict, Any, List
import asyncio
from datetime import datetime

from agents.base_agent import BaseAgent
from agents.config import AgentConfig
from agents.context_store import ContextStore
from models.agent_models import TaskResult
from utils.retry import async_retry, RetryConfig


class ScrapeProgress(BaseModel):
    """抓取进度"""
    keyword: str
    page: int
    total_pages: int
    posts_found: int
    is_complete: bool


class ScraperAgent(BaseAgent):
    """
    数据抓取Agent

    Skills:
    - search_posts: 搜索帖子
    - get_comments: 获取评论
    - get_stats: 获取统计
    - batch_scrape: 批量抓取
    """

    def __init__(self, config: AgentConfig, context_store: ContextStore, mcp_clients: Dict[str, Any]):
        super().__init__("scraper", config, context_store, mcp_clients)

        # 抓取配置
        self.max_pages_per_keyword = config.get('max_pages_per_keyword', 2)
        self.max_posts_to_analyze = config.get('max_posts_to_analyze', 20)
        self.request_delay = config.get('request_delay', 1.0)
        self.retry_config = RetryConfig(
            max_retries=3,
            base_delay=2.0,
            max_delay=10.0
        )

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> TaskResult:
        """执行抓取任务"""
        try:
            if task == 'search_posts':
                result = await self._search_posts(context, kwargs)
            elif task == 'get_comments':
                result = await self._get_comments(context, kwargs)
            elif task == 'batch_scrape':
                result = await self._batch_scrape(context, kwargs)
            else:
                raise ValueError(f"Unknown task: {task}")

            return TaskResult(
                success=True,
                data=result,
                agent_name=self.name
            )

        except Exception as e:
            self.logger.error(f"Scraper task failed: {e}")
            return TaskResult(
                success=False,
                error=str(e),
                agent_name=self.name
            )

    @async_retry
    async def _search_posts(
        self,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """搜索帖子"""
        keywords = context.get('keywords', [])
        all_posts = []
        scrape_progress = []

        total_keywords = len(keywords)

        for i, keyword in enumerate(keywords):
            self.update_progress(
                f'searching_{keyword}',
                (i / total_keywords) * 0.7,
                f"正在搜索关键词: {keyword}"
            )

            # 并行抓取多页
            posts = await self._scrape_keyword_pages(keyword, self.max_pages_per_keyword)
            all_posts.extend(posts)

            scrape_progress.append(ScrapeProgress(
                keyword=keyword,
                page=self.max_pages_per_keyword,
                total_pages=self.max_pages_per_keyword,
                posts_found=len(posts),
                is_complete=True
            ))

            # 延迟避免限流
            await asyncio.sleep(self.request_delay)

        # 去重
        unique_posts = self._deduplicate_posts(all_posts)

        self.update_progress('search_complete', 1.0, f"共找到 {len(unique_posts)} 个帖子")

        return {
            'posts': unique_posts,
            'total_count': len(unique_posts),
            'keywords_used': keywords,
            'scrape_progress': [p.model_dump() for p in scrape_progress]
        }

    async def _scrape_keyword_pages(
        self,
        keyword: str,
        max_pages: int
    ) -> List[Dict[str, Any]]:
        """抓取单个关键词的多页结果"""
        posts = []
        note_ids = set()

        # 并行抓取多页
        tasks = [
            self._scrape_page(keyword, page)
            for page in range(1, max_pages + 1)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Page scrape failed: {result}")
                continue

            for post in result.get('posts', []):
                if post['note_id'] not in note_ids:
                    note_ids.add(post['note_id'])
                    posts.append(post)

        return posts

    @async_retry
    async def _scrape_page(
        self,
        keyword: str,
        page: int
    ) -> Dict[str, Any]:
        """抓取单页"""
        result = await self.use_mcp(
            'xhs',
            'search_notes',
            keyword=keyword,
            page=page
        )
        return result

    def _deduplicate_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重帖子"""
        seen = set()
        unique = []
        for post in posts:
            note_id = post.get('note_id')
            if note_id and note_id not in seen:
                seen.add(note_id)
                unique.append(post)
        return unique

    async def _get_comments(
        self,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取评论"""
        posts = context.get('posts', [])
        max_posts = kwargs.get('max_posts', self.max_posts_to_analyze)

        # 选择热度最高的帖子
        top_posts = sorted(
            posts,
            key=lambda p: p.get('liked_count', 0),
            reverse=True
        )[:max_posts]

        posts_with_comments = []
        total = len(top_posts)

        for i, post in enumerate(top_posts):
            self.update_progress(
                'getting_comments',
                (i / total) * 0.8 + 0.2,
                f"正在获取评论 {i+1}/{total}"
            )

            try:
                comments = await self.use_mcp(
                    'xhs',
                    'get_comments',
                    note_id=post['note_id']
                )
                post['comments_data'] = comments
                posts_with_comments.append(post)

                await asyncio.sleep(self.request_delay)

            except Exception as e:
                self.logger.warning(f"Failed to get comments for post {post.get('note_id')}: {e}")
                post['comments_data'] = []
                posts_with_comments.append(post)

        return {
            'posts_with_comments': posts_with_comments,
            'total_count': len(posts_with_comments)
        }

    async def _batch_scrape(
        self,
        context: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """批量抓取：搜索+评论"""
        # 先搜索
        search_result = await self._search_posts(context, kwargs)

        # 再获取评论
        context['posts'] = search_result['posts']
        comments_result = await self._get_comments(context, kwargs)

        return {
            'posts': search_result['posts'],
            'posts_with_comments': comments_result['posts_with_comments'],
            'summary': {
                'total_posts': search_result['total_count'],
                'posts_with_comments': comments_result['total_count']
            }
        }
```

---

## 4. 数据模型设计

### 4.1 Agent相关模型

```python
# models/agent_models.py
"""
Agent系统的数据模型
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressUpdate(BaseModel):
    """进度更新"""
    agent_name: str
    step: str
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    timestamp: datetime
    details: Dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """任务执行结果"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    agent_name: Optional[str] = None
    run_id: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    """执行计划的步骤"""
    order: int
    name: str
    description: str
    agent: str  # keyword/scraper/analyzer/reporter
    task: str
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    progress_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    timeout: float = Field(default=300.0)
    fallback_strategy: Optional[Dict[str, Any]] = None


class ExecutionPlan(BaseModel):
    """执行计划"""
    steps: List[PlanStep]
    needs_clarification: bool = False
    questions: List[str] = Field(default_factory=list)
    resume_from_checkpoint: bool = True
    estimated_duration: float = Field(default=600.0)  # 秒


class OrchestratorState(BaseModel):
    """编排器状态"""
    current_run_id: Optional[str] = None
    current_plan: Optional[ExecutionPlan] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ValidationPlan(BaseModel):
    """验证计划"""
    business_idea: str
    target_keywords: List[str]
    search_scope: Dict[str, Any]
    analysis_depth: str = Field(default="standard")  # basic/standard/deep
    output_format: str = Field(default="detailed")  # summary/standard/detailed


class AgentMessage(BaseModel):
    """Agent间通信消息"""
    from_agent: str
    to_agent: str
    message_type: str  # request/response/notification
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
```

### 4.2 上下文存储模型

```python
# models/context_models.py
"""
上下文存储的数据模型
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta


class RunContext(BaseModel):
    """运行上下文"""
    run_id: str
    business_idea: str
    user_preferences: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    status: str = "running"

    # 各阶段数据
    keywords: Optional[List[str]] = None
    posts: Optional[List[Dict[str, Any]]] = None
    analyses: Optional[List[Dict[str, Any]]] = None
    final_report: Optional[Dict[str, Any]] = None

    # 进度信息
    progress_updates: List[Dict[str, Any]] = []

    # 元数据
    metadata: Dict[str, Any] = {}


class ContextQuery(BaseModel):
    """上下文查询"""
    run_id: Optional[str] = None
    business_idea: Optional[str] = None
    status: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 10
```

---

## 5. MCP服务器设计

### 5.1 XHS MCP Server

```python
# mcp_servers/xhs_server.py
"""
小红书数据获取MCP服务器
"""

from mcp import Server, ServerInstance
from typing import List, Dict, Any
import asyncio
import logging

from business_validator.config import XHZ_AUTH_TOKEN
from business_validator.scrapers.xhs import (
    scrape_xhs_search,
    scrape_xhs_post_comments
)


server = Server("xhs-server", version="1.0.0")
logger = logging.getLogger("mcp.xhs")


@server.tool()
async def search_notes(
    keyword: str,
    page: int = 1,
    sort: str = "general",
    note_type: str = "_0"
) -> Dict[str, Any]:
    """
    搜索小红书笔记

    Args:
        keyword: 搜索关键词
        page: 页码(从1开始)
        sort: 排序方式 (general/popular/time)
        note_type: 笔记类型 (_0:全部, _1:视频, _2:图文)

    Returns:
        {
            "posts": List[Dict],  # 帖子列表
            "keyword": str,       # 搜索关键词
            "page": int,          # 当前页
            "has_more": bool      # 是否有更多
        }
    """
    logger.info(f"Searching XHS: keyword={keyword}, page={page}")

    try:
        result = await asyncio.to_thread(
            scrape_xhs_search,
            keyword=keyword,
            page=page
        )
        return result
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise


@server.tool()
async def get_note_comments(
    note_id: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    获取笔记评论

    Args:
        note_id: 笔记ID
        limit: 最多返回评论数

    Returns:
        评论列表
    """
    logger.info(f"Getting comments for note: {note_id}")

    try:
        comments = await asyncio.to_thread(
            scrape_xhs_post_comments,
            post_url=note_id
        )
        return comments[:limit]
    except Exception as e:
        logger.error(f"Get comments failed: {e}")
        raise


@server.tool()
async def get_note_stats(note_id: str) -> Dict[str, Any]:
    """
    获取笔记统计数据

    Args:
        note_id: 笔记ID

    Returns:
        统计数据
    """
    # 实现获取统计数据
    pass


@server.tool()
async def batch_get_comments(
    note_ids: List[str],
    limit_per_note: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    """
    批量获取评论

    Args:
        note_ids: 笔记ID列表
        limit_per_note: 每个笔记最多评论数

    Returns:
        {note_id: comments}
    """
    results = {}

    tasks = [
        get_note_comments(note_id, limit_per_note)
        for note_id in note_ids
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for note_id, response in zip(note_ids, responses):
        if isinstance(response, Exception):
            logger.warning(f"Failed to get comments for {note_id}: {response}")
            results[note_id] = []
        else:
            results[note_id] = response

    return results


def create_xhs_server(config: Dict[str, Any]) -> ServerInstance:
    """创建XHS MCP服务器实例"""
    return server.create_instance(
        host=config.get('host', 'localhost'),
        port=config.get('port', 8001)
    )
```

### 5.2 LLM MCP Server

```python
# mcp_servers/llm_server.py
"""
LLM调用MCP服务器
"""

from mcp import Server, ServerInstance
from typing import Any, Dict, Optional
import logging
from pydantic import BaseModel

from SimplerLLM.langauge.llm import LLM, LLMProvider


server = Server("llm-server", version="1.0.0")
logger = logging.getLogger("mcp.llm")

# LLM实例
_llm_instance: Optional[LLM] = None


def get_llm() -> LLM:
    """获取LLM实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLM.create(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o"
        )
    return _llm_instance


@server.tool()
async def generate_text(
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.7
) -> str:
    """
    生成文本

    Args:
        prompt: 提示词
        max_tokens: 最大token数
        temperature: 温度参数

    Returns:
        生成的文本
    """
    logger.debug(f"Generating text, tokens={max_tokens}")

    llm = get_llm()
    result = await llm.generate_text_async(
        user_prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return result


@server.tool()
async def generate_structured(
    prompt: str,
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    生成结构化输出

    Args:
        prompt: 提示词
        schema: JSON Schema

    Returns:
        符合schema的结构化数据
    """
    logger.debug("Generating structured output")

    llm = get_llm()

    # 将schema转换为Pydantic模型
    from pydantic import create_model
    model_name = "DynamicModel"
    DynamicModel = create_model(model_name, **schema)

    result = await llm.generate_json_with_pydantic_async(
        user_prompt=prompt,
        pydantic_model=DynamicModel,
        model_name="gpt-4o"
    )

    return result.model_dump()


@server.tool()
async def batch_generate(
    prompts: List[str],
    max_tokens: int = 1000
) -> List[str]:
    """
    批量生成文本

    Args:
        prompts: 提示词列表
        max_tokens: 每个请求的最大token数

    Returns:
        生成结果列表
    """
    logger.info(f"Batch generating {len(prompts)} prompts")

    llm = get_llm()

    tasks = [
        llm.generate_text_async(
            user_prompt=prompt,
            max_tokens=max_tokens
        )
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    outputs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Prompt {i} failed: {result}")
            outputs.append("")
        else:
            outputs.append(result)

    return outputs


def create_llm_server(config: Dict[str, Any]) -> ServerInstance:
    """创建LLM MCP服务器实例"""
    return server.create_instance(
        host=config.get('host', 'localhost'),
        port=config.get('port', 8002)
    )
```

### 5.3 Storage MCP Server

```python
# mcp_servers/storage_server.py
"""
存储MCP服务器
提供检查点保存/加载功能
"""

from mcp import Server, ServerInstance
from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path


server = Server("storage-server", version="1.0.0")
logger = logging.getLogger("mcp.storage")

# 配置
_data_dir: Optional[Path] = None


def get_data_dir() -> Path:
    """获取数据目录"""
    global _data_dir
    if _data_dir is None:
        _data_dir = Path("validation_data")
        _data_dir.mkdir(parents=True, exist_ok=True)
    return _data_dir


@server.tool()
async def save_checkpoint(
    run_id: str,
    step: str,
    data: Dict[str, Any]
) -> str:
    """
    保存检查点

    Args:
        run_id: 运行ID
        step: 步骤名称
        data: 要保存的数据

    Returns:
        文件路径
    """
    run_dir = get_data_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    file_path = run_dir / f"{step}.json"

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Checkpoint saved: {file_path}")
    return str(file_path)


@server.tool()
async def load_checkpoint(
    run_id: str,
    step: str
) -> Optional[Dict[str, Any]]:
    """
    加载检查点

    Args:
        run_id: 运行ID
        step: 步骤名称

    Returns:
        检查点数据，不存在返回None
    """
    file_path = get_data_dir() / run_id / f"{step}.json"

    if not file_path.exists():
        logger.warning(f"Checkpoint not found: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Checkpoint loaded: {file_path}")
        return data

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


@server.tool()
async def list_checkpoints(run_id: str) -> list:
    """
    列出运行的所有检查点

    Args:
        run_id: 运行ID

    Returns:
        检查点文件名列表
    """
    run_dir = get_data_dir() / run_id

    if not run_dir.exists():
        return []

    checkpoints = []
    for file in run_dir.glob("*.json"):
        checkpoints.append(file.stem)

    return sorted(checkpoints)


@server.tool()
async def delete_run(run_id: str) -> bool:
    """
    删除运行数据

    Args:
        run_id: 运行ID

    Returns:
        是否成功
    """
    import shutil
    run_dir = get_data_dir() / run_id

    if not run_dir.exists():
        return False

    try:
        shutil.rmtree(run_dir)
        logger.info(f"Deleted run data: {run_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete run: {e}")
        return False


def create_storage_server(config: Dict[str, Any]) -> ServerInstance:
    """创建Storage MCP服务器实例"""
    # 设置数据目录
    global _data_dir
    _data_dir = Path(config.get('data_dir', 'validation_data'))

    return server.create_instance(
        host=config.get('host', 'localhost'),
        port=config.get('port', 8003)
    )
```

---

## 6. UI设计

### 6.1 Agent Streamlit UI

```python
# ui/agent_ui.py
"""
基于Agent的Streamlit UI
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
from pathlib import Path

from agents.orchestrator import OrchestratorAgent
from agents.context_store import RedisContextStore
from agents.config import load_config
from ui.components.progress_monitor import ProgressMonitor
from ui.components.result_viewer import ResultViewer
from ui.components.chat_interface import ChatInterface


# 页面配置
st.set_page_config(
    page_title="小红书商业调研 Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化
@st.cache_resource
def init_agents():
    """初始化Agent系统"""
    config = load_config('config/default.yaml')

    # 创建上下文存储
    context_store = RedisContextStore(
        host=config.get('redis.host', 'localhost'),
        port=config.get('redis.port', 6379)
    )

    # 创建MCP客户端
    mcp_clients = init_mcp_clients(config)

    # 创建编排Agent
    orchestrator = OrchestratorAgent(
        config=config.get('agents.orchestrator', {}),
        context_store=context_store,
        mcp_clients=mcp_clients
    )

    # 注册Subagents
    from agents.subagents.keyword_agent import KeywordAgent
    from agents.subagents.scraper_agent import ScraperAgent
    from agents.subagents.analyzer_agent import AnalyzerAgent
    from agents.subagents.reporter_agent import ReporterAgent

    subagents = {
        'keyword': KeywordAgent(config, context_store, mcp_clients),
        'scraper': ScraperAgent(config, context_store, mcp_clients),
        'analyzer': AnalyzerAgent(config, context_store, mcp_clients),
        'reporter': ReporterAgent(config, context_store, mcp_clients),
    }

    for name, agent in subagents.items():
        orchestrator.register_subagent(name, agent)

    return orchestrator, context_store


def init_mcp_clients(config):
    """初始化MCP客户端"""
    clients = {}

    # XHS MCP客户端
    from mcp_clients.xhs_client import XHSClient
    clients['xhs'] = XHSClient(
        url=config.get('mcp.xhs.url', 'http://localhost:8001')
    )

    # LLM MCP客户端
    from mcp_clients.llm_client import LLMClient
    clients['llm'] = LLMClient(
        url=config.get('mcp.llm.url', 'http://localhost:8002')
    )

    # Storage MCP客户端
    from mcp_clients.storage_client import StorageClient
    clients['storage'] = StorageClient(
        url=config.get('mcp.storage.url', 'http://localhost:8003')
    )

    return clients


def main():
    """主函数"""

    # 侧边栏
    with st.sidebar:
        st.markdown("# 🔍 商业调研 Agent")
        st.markdown("基于AI的小红书市场调研工具")

        st.markdown("---")

        # 历史记录
        st.markdown("## 📚 历史记录")
        # 加载历史运行记录
        # ...

    # 主内容区
    st.markdown("# 小红书商业调研 Agent")
    st.markdown("输入您的业务创意，AI将自动分析小红书数据，生成市场验证报告。")

    # 输入表单
    with st.form("validation_form"):
        col1, col2 = st.columns([3, 1])

        with col1:
            business_idea = st.text_area(
                "业务创意",
                placeholder="例如: 在深圳销售新会陈皮",
                height=100
            )

        with col2:
            st.markdown("### 高级选项")
            analysis_depth = st.selectbox(
                "分析深度",
                ["快速", "标准", "深度"],
                index=1
            )
            target_location = st.text_input("目标地点 (可选)")
            industry = st.text_input("行业 (可选)")

        submitted = st.form_submit_button("🚀 开始分析", type="primary")

    if submitted and business_idea:
        # 显示进度监控
        progress_monitor = ProgressMonitor()
        result_viewer = ResultViewer()

        # 初始化Agents
        orchestrator, context_store = init_agents()

        # 设置进度回调
        orchestrator.set_progress_callback(progress_monitor.on_progress_update)

        # 执行验证
        with st.spinner("正在执行调研分析..."):
            result = asyncio.run(orchestrator.execute(
                task='validate',
                context={
                    'business_idea': business_idea,
                    'user_preferences': {
                        'depth': analysis_depth,
                        'location': target_location,
                        'industry': industry
                    }
                }
            ))

        # 显示结果
        if result.success:
            st.success("✅ 分析完成!")
            result_viewer.display(result.data)

            # AI对话
            st.markdown("---")
            chat_interface = ChatInterface(
                run_id=result.run_id,
                context_store=context_store
            )
            chat_interface.render()
        else:
            st.error(f"❌ 分析失败: {result.error}")


if __name__ == "__main__":
    main()
```

### 6.2 进度监控组件

```python
# ui/components/progress_monitor.py
"""
进度监控组件
"""

import streamlit as st
import time
from typing import Dict, Any


class ProgressMonitor:
    """进度监控器"""

    def __init__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.details_container = st.container()

        # 步骤详情
        self.step_containers = {}

    def on_progress_update(self, update: Dict[str, Any]):
        """进度更新回调"""
        agent = update.get('agent_name')
        step = update.get('step')
        progress = update.get('progress', 0)
        message = update.get('message', '')
        details = update.get('details', {})

        # 更新进度条
        self.progress_bar.progress(progress)

        # 更新状态文本
        self.status_text.text(message)

        # 更新详情
        with self.details_container:
            if step not in self.step_containers:
                self.step_containers[step] = st.container()

            with self.step_containers[step]:
                st.markdown(f"**{agent}** - {step}")
                if details:
                    st.json(details)
```

---

## 7. 部署方案

### 7.1 本地开发环境

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 4. 启动MCP服务器
python scripts/start_mcp_servers.py

# 5. 启动UI
streamlit run ui/agent_ui.py
```

### 7.2 生产环境部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  xhs-mcp:
    build: .
    command: python -m mcp_servers.xhs_server
    ports:
      - "8001:8001"
    environment:
      - TIKHUB_TOKEN=${TIKHUB_TOKEN}
    depends_on:
      - redis

  llm-mcp:
    build: .
    command: python -m mcp_servers.llm_server
    ports:
      - "8002:8002"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL}
    depends_on:
      - redis

  storage-mcp:
    build: .
    command: python -m mcp_servers.storage_server
    ports:
      - "8003:8003"
    volumes:
      - validation_data:/app/validation_data
    depends_on:
      - redis

  streamlit:
    build: .
    command: streamlit run ui/agent_ui.py
    ports:
      - "8501:8501"
    environment:
      - REDIS_HOST=redis
      - XHS_MCP_URL=http://xhs-mcp:8001
      - LLM_MCP_URL=http://llm-mcp:8002
      - STORAGE_MCP_URL=http://storage-mcp:8003
    depends_on:
      - redis
      - xhs-mcp
      - llm-mcp
      - storage-mcp
    volumes:
      - validation_data:/app/validation_data

volumes:
  redis_data:
  validation_data:
```

---

## 8. 测试方案

### 8.1 单元测试

```python
# tests/test_agents/test_keyword_agent.py
"""
关键词Agent单元测试
"""

import pytest
from unittest.mock import Mock, AsyncMock

from agents.subagents.keyword_agent import KeywordAgent
from agents.context_store import ContextStore
from models.agent_models import TaskResult


@pytest.fixture
def mock_context_store():
    """Mock上下文存储"""
    return Mock(spec=ContextStore)


@pytest.fixture
def mock_mcp_clients():
    """Mock MCP客户端"""
    return {
        'llm': Mock(),
        'storage': Mock()
    }


@pytest.fixture
def keyword_agent(mock_context_store, mock_mcp_clients):
    """关键词Agent实例"""
    from agents.config import AgentConfig
    config = AgentConfig()

    return KeywordAgent(
        config=config,
        context_store=mock_context_store,
        mcp_clients=mock_mcp_clients
    )


@pytest.mark.asyncio
async def test_generate_keywords(keyword_agent, mock_mcp_clients):
    """测试生成关键词"""
    # Mock LLM响应
    mock_llm = mock_mcp_clients['llm']
    mock_llm.generate_structured = AsyncMock(return_value=KeywordModel(
        keywords=["关键词1", "关键词2", "关键词3"]
    ))

    # 执行
    result = await keyword_agent.execute(
        task='generate',
        context={'business_idea': '在深圳卖陈皮'},
        count=3
    )

    # 验证
    assert result.success
    assert len(result.data['keywords']) == 3
    mock_llm.generate_structured.assert_called_once()


@pytest.mark.asyncio
async def test_refine_keywords(keyword_agent, mock_mcp_clients):
    """测试优化关键词"""
    mock_llm = mock_mcp_clients['llm']
    mock_llm.generate_structured = AsyncMock(return_value=KeywordRefinement(
        original_keywords=["关键词1"],
        refined_keywords=["优化关键词1"],
        refinement_reason="提高相关性",
        suggested_additions=["建议关键词"]
    ))

    result = await keyword_agent.execute(
        task='refine',
        context={
            'existing_keywords': ['关键词1'],
            'feedback': '关键词不够精准',
            'business_idea': '测试创意'
        }
    )

    assert result.success
    assert 'refined_keywords' in result.data
```

### 8.2 集成测试

```python
# tests/test_integration/test_validation_flow.py
"""
端到端验证流程测试
"""

import pytest
import asyncio

from agents.orchestrator import OrchestratorAgent
from agents.context_store import RedisContextStore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_validation_flow():
    """测试完整的验证流程"""

    # 初始化
    config = load_config('config/test.yaml')
    context_store = RedisContextStore(host='localhost', port=6379)
    mcp_clients = init_test_mcp_clients()

    orchestrator = OrchestratorAgent(config, context_store, mcp_clients)

    # 执行
    result = await orchestrator.execute(
        task='validate',
        context={
            'business_idea': '在深圳销售新会陈皮'
        }
    )

    # 验证
    assert result.success
    assert 'final_report' in result.data
    assert result.data['final_report']['overall_score'] >= 0
```

---

## 9. 总结

本技术规格文档详细描述了小红书商业调研Agent系统的设计方案，包括：

1. **技术选型**: Python + asyncio + Claude Code Agent + MCP协议
2. **系统架构**: 四层架构，从UI到外部服务
3. **核心模块**: BaseAgent、Orchestrator、4个Subagents
4. **数据模型**: Agent通信、进度跟踪、执行计划
5. **MCP服务器**: XHS、LLM、Storage三个独立服务
6. **UI设计**: 基于Streamlit的新界面
7. **部署方案**: Docker Compose一键部署
8. **测试方案**: 单元测试和集成测试

---

*文档审核要点：*
- [ ] 技术选型是否合理
- [ ] 架构设计是否清晰
- [ ] 模块职责是否明确
- [ ] 接口定义是否完整
- [ ] 数据模型是否覆盖全面
- [ ] 部署方案是否可行
- [ ] 测试方案是否充分

---

*版本历史:*
- v1.0 (2025-01-02): 初始版本
