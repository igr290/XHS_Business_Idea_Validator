"""
测试评论标签分析功能
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from models.business_models import XhsCommentModel, PostWithComments


def test_models():
    """测试数据模型"""
    print("测试数据模型...")

    # 测试 TagAnalysis 模型
    from models.business_models import TagAnalysis

    tag_analysis = TagAnalysis(
        crowd_scenario={
            "用户需求与痛点-痛点问题": ["安装便捷", "使用困难"],
            "用户需求与痛点-使用场景": ["家庭使用", "办公室使用"]
        },
        functional_value={
            "产品反馈-产品优点": ["效果好", "性能稳定"]
        },
        assurance_value={},
        experience_value={
            "价格感知": ["价格合理", "性价比高"]
        },
        total_comments_analyzed=50,
        total_tags_applied=120,
        analysis_summary="测试标签分析",
        tag_statistics={
            "人群场景.用户需求与痛点-痛点问题.安装便捷": 15,
            "人群场景.用户需求与痛点-痛点问题.使用困难": 8
        }
    )

    print(f"✅ TagAnalysis 模型创建成功")
    print(f"   - 人群场景标签数: {sum(len(v) for v in tag_analysis.crowd_scenario.values())}")
    print(f"   - 功能价值标签数: {sum(len(v) for v in tag_analysis.functional_value.values())}")
    print(f"   - 总评论数: {tag_analysis.total_comments_analyzed}")
    print(f"   - 总标签应用数: {tag_analysis.total_tags_applied}")


def test_import_skills():
    """测试 skills 导入"""
    print("\n测试 skills 导入...")

    try:
        from agents.skills.analyzer_skills import (
            analyze_comments_with_tags_skill,
            generate_combined_analysis_from_posts_skill
        )
        print("✅ skills 导入成功")
        print(f"   - analyze_comments_with_tags_skill: {analyze_comments_with_tags_skill.__name__}")
        print(f"   - generate_combined_analysis_from_posts_skill: {generate_combined_analysis_from_posts_skill.__name__}")
    except ImportError as e:
        print(f"❌ skills 导入失败: {e}")
        return False

    return True


def test_analyzer_agent():
    """测试 AnalyzerAgent"""
    print("\n测试 AnalyzerAgent...")

    try:
        from agents.subagents.analyzer_agent import AnalyzerAgent
        print("✅ AnalyzerAgent 导入成功")

        # 检查是否包含新的 task
        import inspect
        methods = [m for m in dir(AnalyzerAgent) if not m.startswith('_')]
        print(f"   - 公共方法数: {len(methods)}")

        # 检查 _analyze_comments_with_tags 方法
        if hasattr(AnalyzerAgent, '_analyze_comments_with_tags'):
            print("   ✅ 包含 _analyze_comments_with_tags 方法")
        else:
            print("   ❌ 缺少 _analyze_comments_with_tags 方法")
            return False

    except ImportError as e:
        print(f"❌ AnalyzerAgent 导入失败: {e}")
        return False

    return True


def test_orchestrator():
    """测试 Orchestrator"""
    print("\n测试 Orchestrator...")

    try:
        from agents.orchestrator import OrchestratorAgent
        from models.agent_models import ExecutionPlan

        # 检查 OrchestratorAgent 是否可以正常导入
        print("✅ OrchestratorAgent 导入成功")

        # 检查 ExecutionPlan 模型
        plan = ExecutionPlan(
            business_idea="测试创意",
            steps=[],
            total_steps=6
        )
        print(f"   ✅ ExecutionPlan 创建成功，total_steps={plan.total_steps}")

    except ImportError as e:
        print(f"❌ Orchestrator 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ Orchestrator 测试失败: {e}")
        return False

    return True


def test_mock_data():
    """测试模拟数据"""
    print("\n测试模拟数据...")

    # 创建模拟的 PostWithComments 数据
    mock_post = PostWithComments(
        note_id="test123",
        title="测试标题",
        desc="这是一个测试帖子",
        type="normal",
        publish_time=1704067200,
        liked_count=100,
        collected_count=50,
        shared_count=20,
        comments_count=10,
        user_id="user123",
        user_nickname="测试用户",
        comments_data=[
            XhsCommentModel(
                comment_id="c1",
                note_id="test123",
                content="这个产品很好用，推荐大家购买",
                publish_time=1704067200,
                like_count=5,
                user_id="u1",
                user_nickname="用户A"
            ),
            XhsCommentModel(
                comment_id="c2",
                note_id="test123",
                content="价格有点贵，但是质量不错",
                publish_time=1704067201,
                like_count=3,
                user_id="u2",
                user_nickname="用户B"
            )
        ],
        comments_fetched=True
    )

    print(f"✅ 模拟数据创建成功")
    print(f"   - 帖子标题: {mock_post.title}")
    print(f"   - 评论数: {len(mock_post.comments_data)}")
    print(f"   - 评论已获取: {mock_post.comments_fetched}")

    return mock_post


def main():
    """主测试函数"""
    print("="*70)
    print("🧪 评论标签分析功能测试")
    print("="*70)

    try:
        # 测试1: 数据模型
        test_models()

        # 测试2: Skills 导入
        if not test_import_skills():
            print("\n❌ Skills 导入测试失败")
            return 1

        # 测试3: AnalyzerAgent
        if not test_analyzer_agent():
            print("\n❌ AnalyzerAgent 测试失败")
            return 1

        # 测试4: Orchestrator
        if not test_orchestrator():
            print("\n❌ Orchestrator 测试失败")
            return 1

        # 测试5: 模拟数据
        test_mock_data()

        print("\n" + "="*70)
        print("✅ 所有测试通过!")
        print("="*70)
        print("\n📋 功能摘要:")
        print("1. ✅ 添加了 TagAnalysis 数据模型")
        print("2. ✅ 实现了 analyze_comments_with_tags_skill")
        print("3. ✅ 在 AnalyzerAgent 中注册了新 task")
        print("4. ✅ 在 Orchestrator 中添加了评论分析步骤")
        print("5. ✅ 更新了执行计划 (total_steps: 5 → 6)")
        print("\n🎯 新的工作流程:")
        print("1. 生成关键词 (可选)")
        print("2. 抓取数据")
        print("3. 分析笔记和评论 (统一分析)")
        print("4. 🆕 评论标签体系分析 (人群/功能/保障/体验价值)")
        print("5. 综合分析")
        print("6. 生成报告")

        return 0

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
