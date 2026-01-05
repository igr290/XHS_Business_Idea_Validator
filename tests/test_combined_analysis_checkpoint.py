"""
测试 combined_analysis 失败时的检查点保存行为
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.subagents.analyzer_agent import AnalyzerAgent
from agents.config import AgentConfig, ConfigManager
from agents.context_store import ContextStore
from models.business_models import CombinedAnalysis


@pytest.mark.asyncio
async def test_combined_analysis_saves_checkpoint_on_failure():
    """
    测试：即使 combined_analysis 失败，也应该保存检查点（包含 fallback 数据）
    """
    # 准备测试数据
    context_store = ContextStore()

    # 创建 ConfigManager
    config_manager = ConfigManager()
    config_manager._agent_configs = {
        "analyzer": AgentConfig(
            name="test_analyzer",
            type="analyzer"
        )
    }

    agent = AnalyzerAgent(
        config=config_manager,
        context_store=context_store,
        mcp_clients={}
    )

    # 模拟 save_checkpoint
    checkpoint_data = {}
    saved_checkpoints = []

    async def mock_save_checkpoint(run_id, step, data):
        saved_checkpoints.append((run_id, step, data))
        checkpoint_data[step] = data

    agent.save_checkpoint = mock_save_checkpoint

    # 模拟 LLM 调用失败后的 fallback 结果
    fallback_result = {
        "success": False,
        "analysis": {
            "overall_score": 50,
            "market_validation_summary": "分析失败（重试2次后仍失败）: ValueError",
            "key_pain_points": ["痛点1", "痛点2"],
            "existing_solutions": ["方案1"],
            "market_opportunities": [],
            "recommendations": ["请重新运行分析"],
            "platform_insights": [],
            "metadata": {}
        },
        "error": "LLM validation failed",
        "error_type": "ValueError"
    }

    # 准备上下文
    context = {
        "run_id": "test_run_123",
        "business_idea": "测试业务创意",
        "posts_analyses": {
            "summary": {
                "relevant_count": 5,
                "relevance_rate": 0.5,
                "avg_engagement_score": 7.0,
                "sentiment_distribution": {"positive": 3, "negative": 1, "neutral": 1}
            },
            "relevant_posts": []
        },
        "comments_analyses": {
            "analysis": {
                "insights": [],
                "common_themes": []
            }
        }
    }

    kwargs = {}

    # 模拟 generate_combined_analysis_skill 返回 fallback 结果
    with patch('agents.subagents.analyzer_agent.generate_combined_analysis_skill',
               new=AsyncMock(return_value=fallback_result)):

        # 执行
        result = await agent._combined_analysis(context, kwargs)

        # 验证
        assert result["success"] == False, f"Expected success=False, got {result}"
        assert result["analysis"]["overall_score"] == 50

        # 关键验证：检查点应该被保存
        assert len(saved_checkpoints) == 1, f"Expected 1 checkpoint, got {len(saved_checkpoints)}"
        run_id, step, data = saved_checkpoints[0]

        assert run_id == "test_run_123"
        assert step == "combined_analysis_complete"
        assert "combined_analysis" in data
        assert data["combined_analysis"]["success"] == False
        assert data["combined_analysis"]["analysis"]["overall_score"] == 50


@pytest.mark.asyncio
async def test_combined_analysis_saves_checkpoint_on_success():
    """
    测试：成功时也应该保存检查点（原有行为）
    """
    # 准备测试数据
    context_store = ContextStore()

    # 创建 ConfigManager
    config_manager = ConfigManager()
    config_manager._agent_configs = {
        "analyzer": AgentConfig(
            name="test_analyzer",
            type="analyzer"
        )
    }

    agent = AnalyzerAgent(
        config=config_manager,
        context_store=context_store,
        mcp_clients={}
    )

    # 模拟 save_checkpoint
    saved_checkpoints = []

    async def mock_save_checkpoint(run_id, step, data):
        saved_checkpoints.append((run_id, step, data))

    agent.save_checkpoint = mock_save_checkpoint

    # 模拟成功结果
    success_result = {
        "success": True,
        "analysis": {
            "overall_score": 75,
            "market_validation_summary": "市场验证成功",
            "key_pain_points": ["痛点1"],
            "existing_solutions": ["方案1"],
            "market_opportunities": ["机会1"],
            "recommendations": ["建议1"],
            "platform_insights": [],
            "metadata": {}
        }
    }

    # 准备上下文
    context = {
        "run_id": "test_run_456",
        "business_idea": "测试业务创意",
        "posts_analyses": {
            "summary": {"relevant_count": 5, "relevance_rate": 0.5},
            "relevant_posts": []
        },
        "comments_analyses": {
            "analysis": {"insights": [], "common_themes": []}
        }
    }

    kwargs = {}

    # 模拟 generate_combined_analysis_skill 返回成功结果
    with patch('agents.subagents.analyzer_agent.generate_combined_analysis_skill',
               new=AsyncMock(return_value=success_result)):

        # 执行
        result = await agent._combined_analysis(context, kwargs)

        # 验证
        assert result["success"] == True
        assert result["analysis"]["overall_score"] == 75

        # 关键验证：检查点应该被保存
        assert len(saved_checkpoints) == 1
        run_id, step, data = saved_checkpoints[0]

        assert run_id == "test_run_456"
        assert step == "combined_analysis_complete"
        assert data["combined_analysis"]["success"] == True


if __name__ == "__main__":
    asyncio.run(test_combined_analysis_saves_checkpoint_on_failure())
    print("test_combined_analysis_saves_checkpoint_on_failure passed!")

    asyncio.run(test_combined_analysis_saves_checkpoint_on_success())
    print("test_combined_analysis_saves_checkpoint_on_success passed!")
