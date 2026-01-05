"""
Agent 系统端到端测试

测试完整的业务验证流程
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_orchestrator():
    """测试编排器端到端流程"""
    print("\n" + "="*80)
    print("端到端测试: Orchestrator")
    print("="*80)

    from agents.orchestrator import OrchestratorAgent
    from agents.config import ConfigManager
    from agents.context_store import ContextStore
    from mcp_servers.xhs_server import create_xhs_mcp_server
    from mcp_servers.llm_server import create_llm_mcp_server
    from mcp_servers.storage_server import create_storage_mcp_server

    # 配置
    config = ConfigManager()
    context_store = ContextStore()

    # 获取 API 配置
    xhs_config = config.get_xhs_mcp_config()
    llm_config = config.get_llm_config()

    try:
        # 启动 MCP 服务器
        print("\n🔧 启动 MCP 服务器...")
        xhs_server = await create_xhs_mcp_server(xhs_config.auth_token, request_delay=xhs_config.request_delay)
        llm_server = await create_llm_mcp_server(api_key=llm_config.api_key, base_url=llm_config.base_url)
        storage_server = await create_storage_mcp_server("agent_context/checkpoints")

        mcp_clients = {
            "xhs": xhs_server,
            "llm": llm_server,
            "storage": storage_server
        }
        print("✅ MCP 服务器启动成功")

        # 创建编排器
        print("\n🤖 创建编排器...")
        orchestrator = OrchestratorAgent(
            config=config,
            context_store=context_store,
            mcp_clients=mcp_clients
        )
        await orchestrator.start()
        print("✅ 编排器创建成功")

        # 设置进度回调
        def progress_callback(update):
            bar_length = 30
            filled = int(bar_length * update.progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  [{bar}] {update.progress*100:5.1f}% - {update.message}")

        orchestrator.set_progress_callback(progress_callback)

        # 执行业务验证
        print("\n" + "="*80)
        print("开始业务创意验证")
        print("="*80)

        test_params = {
            "business_idea": "在深圳卖陈皮",
            "keyword_count": 3,
            "pages_per_keyword": 1,  # 减少测试时间
            "comments_per_note": 10,
            "report_format": "html"
        }

        print(f"\n📋 业务创意: {test_params['business_idea']}")
        print(f"   关键词数量: {test_params['keyword_count']}")
        print(f"   每关键词页数: {test_params['pages_per_keyword']}")
        print(f"   每笔记评论数: {test_params['comments_per_note']}")
        print(f"   报告格式: {test_params['report_format']}")

        print("\n" + "-"*80)
        print("执行进度")
        print("-"*80)

        result = await orchestrator.execute(
            task="validate_business_idea",
            context={},
            **test_params
        )

        print("\n" + "-"*80)
        print("执行结果")
        print("-"*80)

        if result.success:
            print(f"\n✅ 验证成功!")
            print(f"   Run ID: {result.run_id}")
            print(f"   执行时间: {result.execution_time:.2f}s")

            # 显示步骤结果
            data = result.data
            state = data.get("state", {})
            step_results = data.get("step_results", {})

            print(f"\n📊 执行统计:")
            print(f"   总步骤: {state.get('total_steps', 0)}")
            print(f"   已完成: {state.get('completed_steps', 0)}")
            print(f"   失败: {state.get('failed_steps', 0)}")

            print(f"\n📝 各步骤结果:")
            for step_id, step_result in step_results.items():
                status = "✅" if step_result.get("success") else "❌"
                exec_time = step_result.get("execution_time", 0)
                print(f"   {status} {step_id}: {exec_time:.2f}s")

            # 显示关键词生成结果
            if "generate_keywords" in step_results:
                kw_result = step_results["generate_keywords"]
                if kw_result.get("success"):
                    keywords = kw_result.get("data", {}).get("keywords", [])
                    print(f"\n🔑 生成关键词: {keywords}")

            # 显示数据抓取结果
            if "scrape_data" in step_results:
                sc_result = step_results["scrape_data"]
                if sc_result.get("success"):
                    sc_data = sc_result.get("data", {})
                    print(f"\n📊 数据抓取:")
                    print(f"   笔记数: {sc_data.get('total_notes', 0)}")
                    print(f"   评论数: {sc_data.get('total_comments', 0)}")

            # 显示分析结果
            if "combined_analysis" in step_results:
                ca_result = step_results["combined_analysis"]
                if ca_result.get("success"):
                    analysis = ca_result.get("data", {}).get("analysis", {})
                    score = analysis.get("overall_score", 0)
                    print(f"\n🎯 综合评分: {score}/100")
                    print(f"   摘要: {analysis.get('market_validation_summary', 'N/A')[:100]}...")

            # 显示报告生成结果
            if "generate_report" in step_results:
                gr_result = step_results["generate_report"]
                if gr_result.get("success"):
                    gr_data = gr_result.get("data", {})
                    saving = gr_result.get("saving", {})
                    if saving.get("success"):
                        print(f"\n📄 报告已保存:")
                        print(f"   路径: {saving.get('path')}")
                        print(f"   大小: {saving.get('size', 0)} bytes")

            print("\n" + "="*80)
            print("🎉 端到端测试通过!")
            print("="*80)
            return True

        else:
            print(f"\n❌ 验证失败: {result.error}")
            print(f"   执行时间: {result.execution_time:.2f}s")
            print("\n" + "="*80)
            print("⚠️  端到端测试失败")
            print("="*80)
            return False

    except Exception as e:
        logger.exception("E2E test failed")
        print(f"\n❌ 测试异常: {e}")
        return False

    finally:
        # 清理资源
        try:
            if 'orchestrator' in locals():
                await orchestrator.stop()
            if 'xhs_server' in locals():
                await xhs_server.stop()
            if 'llm_server' in locals():
                await llm_server.stop()
            if 'storage_server' in locals():
                await storage_server.stop()
            print("\n🧹 资源清理完成")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


async def main():
    """主测试函数"""
    print("="*80)
    print("Agent 系统端到端测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    success = await test_orchestrator()

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
