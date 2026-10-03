"""
Agent 系统数据模型

定义 Agent 系统中使用的所有数据模型
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
    agent_name: str = Field(description="Agent 名称")
    step: str = Field(description="当前步骤")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="进度 0-1")
    message: str = Field(description="进度消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    details: Dict[str, Any] = Field(default_factory=dict, description="额外详情")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskResult(BaseModel):
    """任务执行结果"""
    success: bool = Field(description="是否成功")
    data: Optional[Any] = Field(default=None, description="返回数据")
    error: Optional[str] = Field(default=None, description="错误信息")
    agent_name: Optional[str] = Field(default=None, description="Agent 名称")
    run_id: Optional[str] = Field(default=None, description="运行 ID")
    execution_time: Optional[float] = Field(default=None, description="执行时间(秒)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ExecutionStep(BaseModel):
    """执行步骤"""
    step_id: str = Field(description="步骤 ID")
    agent_name: str = Field(description="负责的 Agent 名称")
    task: str = Field(description="任务类型")
    description: str = Field(description="步骤描述")
    params: Dict[str, Any] = Field(default_factory=dict, description="任务参数")
    depends_on: List[str] = Field(default_factory=list, description="依赖的步骤 ID 列表")
    retry_on_failure: bool = Field(default=False, description="失败时是否重试")
    timeout: float = Field(default=300, description="超时时间(秒)")


class ExecutionPlan(BaseModel):
    """执行计划"""
    business_idea: str = Field(description="业务创意")
    steps: List[ExecutionStep] = Field(description="执行步骤列表")
    total_steps: int = Field(description="总步骤数")

    def __init__(self, **data):
        # Handle legacy PlanStep-based data by converting if needed
        if 'steps' in data and data['steps'] and not isinstance(data['steps'][0], ExecutionStep):
            if hasattr(data['steps'][0], 'task'):
                legacy_steps = data['steps']
                new_steps = []
                for i, s in enumerate(legacy_steps):
                    new_steps.append(ExecutionStep(
                        step_id=str(i),
                        agent_name=getattr(s, 'agent', 'unknown'),
                        task=getattr(s, 'task', 'unknown'),
                        description=getattr(s, 'description', ''),
                        params={},
                        depends_on=getattr(s, 'dependencies', []),
                        retry_on_failure=False,
                        timeout=getattr(s, 'timeout', 300)
                    ))
                data['steps'] = new_steps
        super().__init__(**data)


class OrchestratorState(BaseModel):
    """编排器状态"""
    current_stage: str = Field(description="当前阶段")
    run_id: Optional[str] = Field(default=None, description="运行 ID")
    total_steps: int = Field(default=0, description="总步骤数")
    completed_steps: int = Field(default=0, description="已完成步骤数")
    failed_steps: int = Field(default=0, description="失败步骤数")

    # Legacy fields for compatibility
    current_run_id: Optional[str] = Field(default=None, description="当前运行 ID")
    current_plan: Optional[ExecutionPlan] = Field(default=None, description="当前执行计划")
    start_time: Optional[datetime] = Field(default=None, description="开始时间")
    end_time: Optional[datetime] = Field(default=None, description="结束时间")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentMessage(BaseModel):
    """Agent 间通信消息"""
    from_agent: str = Field(description="发送 Agent")
    to_agent: str = Field(description="接收 Agent")
    message_type: str = Field(description="消息类型: request/response/notification")
    content: Dict[str, Any] = Field(description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    correlation_id: Optional[str] = Field(default=None, description="关联 ID")
    reply_to: Optional[str] = Field(default=None, description="回复的消息 ID")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentMetrics(BaseModel):
    """Agent 运行指标"""
    tasks_completed: int = Field(default=0, description="完成任务数")
    tasks_failed: int = Field(default=0, description="失败任务数")
    total_execution_time: float = Field(default=0.0, description="总执行时间")
    avg_execution_time: float = Field(default=0.0, description="平均执行时间")
    last_execution: Optional[datetime] = Field(default=None, description="最后执行时间")
    mcp_calls: int = Field(default=0, description="MCP 调用次数")
    llm_calls: int = Field(default=0, description="LLM 调用次数")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationPlan(BaseModel):
    """验证计划"""
    business_idea: str = Field(description="业务创意")
    target_keywords: List[str] = Field(description="目标关键词")
    search_scope: Dict[str, Any] = Field(default_factory=dict, description="搜索范围")
    analysis_depth: str = Field(default="standard", description="分析深度: basic/standard/deep")
    output_format: str = Field(default="detailed", description="输出格式: summary/standard/detailed")
