"""
测试 HTML 报告生成功能
"""
import asyncio
import sys
from pathlib import Path

# 添加 agent_system 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.skills.reporter_skills import generate_html_report_skill, save_report_skill


class MockAgent:
    """简单的 Mock Agent"""
    def __init__(self):
        self.name = "test_agent"

    async def use_mcp(self, server_name, method, **kwargs):
        """Mock MCP 调用"""
        return {"success": True, "data": {}}


async def test_html_report_generation():
    """测试 HTML 报告生成"""
    agent = MockAgent()

    # 测试数据
    analysis = {
        'analysis': {
            'overall_score': 75,
            'market_validation_summary': '这是一个测试摘要。市场对该业务创意反应良好，存在一定的机会。',
            'key_pain_points': ['用户不知道在哪里购买优质陈皮', '陈皮价格不透明', '陈皮品质难以辨别'],
            'existing_solutions': ['传统中药材店', '电商平台', '陈皮专卖店'],
            'market_opportunities': ['深圳地区对健康食品需求增长', '陈皮礼品市场潜力大', '年轻人对陈皮养生认知提升'],
            'recommendations': ['建议在深圳核心商圈开设体验店', '开发陈皮相关文创产品', '建立线上社群增强用户粘性'],
            'metadata': {
                'total_posts_analyzed': 50,
                'relevant_count': 30,
                'analysis_date': '2026-01-02'
            }
        }
    }

    # 生成 HTML 报告
    print("🔄 正在生成 HTML 报告...")
    result = await generate_html_report_skill(
        agent, analysis, '在深圳卖陈皮', 'test_run_id_20260102'
    )

    if result['success']:
        print('✅ HTML 报告生成成功')
        print(f"   内容长度: {result['length']} 字符")
        print(f"   格式: {result['report_format']}")

        # 保存报告
        print("\n🔄 正在保存报告...")
        save_result = await save_report_skill(
            agent, result['content'], 'html', 'reports/test_在深圳卖陈皮.html'
        )

        if save_result['success']:
            print('✅ 报告保存成功')
            print(f"   保存路径: {save_result['path']}")
            print(f"   文件大小: {save_result['size']} 字节")
            print("\n📄 请在浏览器中打开报告查看效果")
        else:
            print(f"❌ 报告保存失败: {save_result.get('error')}")
    else:
        print(f"❌ HTML 报告生成失败")


if __name__ == "__main__":
    asyncio.run(test_html_report_generation())
