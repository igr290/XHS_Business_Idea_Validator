"""测试 scraper 返回格式"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.skills.scraper_skills import batch_scrape_skill

class MockAgent:
    def __init__(self):
        self.name = 'test'

    async def use_mcp(self, server, method, **kwargs):
        # 返回模拟数据
        if 'search' in method:
            return {
                'success': True,
                'data': {
                    'notes': [
                        {'note_id': '123', 'title': '测试笔记1'},
                        {'note_id': '456', 'title': '测试笔记2'}
                    ]
                }
            }
        return {'success': True, 'data': {}}

async def test():
    agent = MockAgent()
    result = await batch_scrape_skill(
        agent,
        keywords=['测试'],
        pages_per_keyword=1,
        comments_per_note=0,
        max_notes=5,
        progress_callback=None
    )
    print('Result type:', type(result))
    print('Result keys:', list(result.keys()))
    print('Has notes?', 'notes' in result)
    if 'notes' in result:
        print('Notes count:', len(result['notes']))
    print('Result:', result)

asyncio.run(test())
