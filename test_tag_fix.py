"""
快速测试脚本 - 验证标签分析修复
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.orchestrator import OrchestratorAgent
from agents.config import ConfigManager
from agents.context_store import ContextStore
from agents.logging_config import setup_logging
from mcp_servers.xhs_server import create_xhs_mcp_server
from mcp_servers.llm_server import create_llm_mcp_server
from mcp_servers.storage_server import create_storage_mcp_server


async def test_tag_analysis():
    """测试标签分析功能"""
    print("=" * 60)
    print("测试标签分析功能修复")
    print("=" * 60)

    # 初始化配置
    config = ConfigManager()
    log_level = config.get('logging.level', 'INFO')
    setup_logging(log_level)

    # 创建 MCP 服务器
    print("\n[1/4] 创建 MCP 服务器...")
    xhs_server = await create_xhs_mcp_server(config)
    llm_server = await create_llm_mcp_server(config)
    storage_server = await create_storage_mcp_server()  # 不需要 config 参数

    mcp_clients = {
        "xhs": xhs_server,
        "llm": llm_server,
        "storage": storage_server
    }
    print("✓ MCP 服务器创建成功")

    # 创建 ContextStore
    print("\n[2/4] 创建 ContextStore...")
    context_store = ContextStore(ttl_seconds=3600)
    print("✓ ContextStore 创建成功")

    # 创建 OrchestratorAgent
    print("\n[3/4] 创建 OrchestratorAgent...")
    orchestrator = OrchestratorAgent(
        mcp_clients=mcp_clients,
        context_store=context_store,
        config=config
    )
    print("✓ OrchestratorAgent 创建成功")

    # 运行快速验证（1个关键词，1页，5条评论）
    print("\n[4/4] 运行业务创意验证（快速模式）...")
    print("业务创意: '在深圳卖陈皮'")
    print("参数: 1 关键词 × 1 页 × 5 评论\n")

    try:
        result = await orchestrator.execute(
            task="validate_business_idea",
            context={},
            business_idea="在深圳卖陈皮",
            keyword_count=1,
            pages_per_keyword=1,
            comments_per_note=5,
            report_format="html",
            use_user_input_as_keyword=True
        )

        if result.success:
            print("\n" + "=" * 60)
            print("✓ 验证成功完成！")
            print("=" * 60)
            print(f"报告路径: {result.report_path}")
            print(f"总耗时: {result.duration:.2f} 秒")
            return True
        else:
            print("\n" + "=" * 60)
            print("✗ 验证失败")
            print("=" * 60)
            print(f"错误: {result.error}")
            return False

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ 执行过程中发生错误")
        print("=" * 60)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理资源
        print("\n清理资源...")
        await orchestrator.stop()
        await xhs_server.stop()
        await llm_server.stop()
        await storage_server.stop()
        print("✓ 清理完成")


if __name__ == "__main__":
    success = asyncio.run(test_tag_analysis())
    sys.exit(0 if success else 1)
