"""
XHS API 测试

测试 TikHub API 客户端功能，包括：
- 搜索笔记
- 获取评论
- 批量获取评论
- 429 速率限制处理
- 重试逻辑
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.xhs_server import TikHubXHSClient, XHSMCPServer


# ============================================================================
# Mock 测试数据
# ============================================================================

MOCK_SEARCH_RESPONSE = {
    "data": {
        "data": {
            "items": [
                {
                    "note": {
                        "id": "123456789",
                        "title": "测试笔记标题",
                        "desc": "测试笔记内容",
                        "type": "normal",
                        "time": 1234567890,
                        "liked_count": 100,
                        "collected_count": 50,
                        "shared_count": 20,
                        "comments_count": 30,
                        "user": {
                            "id": "user123",
                            "nickname": "测试用户",
                            "avatar": "https://example.com/avatar.jpg"
                        }
                    }
                }
            ]
        }
    }
}

MOCK_COMMENTS_RESPONSE = {
    "data": {
        "data": {
            "comments": [
                {
                    "id": "comment1",
                    "content": "测试评论内容",
                    "time": 1234567890,
                    "ip_location": "北京",
                    "like_count": 10,
                    "user": {
                        "id": "user456",
                        "nickname": "评论用户"
                    },
                    "parent_comment": {}
                }
            ]
        }
    }
}


# ============================================================================
# TikHubXHSClient 测试
# ============================================================================

class MockResponse:
    """Mock aiohttp 响应"""
    def __init__(self, status, data, headers=None):
        self.status = status
        self._data = data
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status >= 400:
            from aiohttp import ClientResponseError
            raise ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
                message=f"HTTP {self.status}"
            )

    async def json(self):
        return self._data


class MockSession:
    """Mock aiohttp 会话"""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get(self, url, headers=None, params=None):
        """返回模拟响应"""
        response = self.responses[self.call_count] if self.call_count < len(self.responses) else self.responses[-1]
        self.call_count += 1
        return MockResponse(**response)


async def test_search_notes_success():
    """测试搜索笔记成功"""
    print("\n=== 测试: 搜索笔记成功 ===")

    client = TikHubXHSClient("test_token")
    await client.start()

    # Mock 响应
    mock_response = MockResponse(200, MOCK_SEARCH_RESPONSE)

    with patch.object(client._session, 'get', return_value=mock_response):
        result = await client.search_notes("测试关键词")

    assert result is not None
    assert "data" in result
    print("✓ 搜索笔记成功")

    await client.close()


async def test_search_notes_429_retry():
    """测试 429 速率限制重试"""
    print("\n=== 测试: 429 速率限制重试 ===")

    client = TikHubXHSClient("test_token")
    await client.start()

    # Mock 响应序列: 429 -> 成功
    mock_session = MockSession(responses=[
        {"status": 429, "data": {}, "headers": {"Retry-After": "2"}},
        {"status": 200, "data": MOCK_SEARCH_RESPONSE}
    ])

    with patch.object(client._session, 'get', mock_session.get):
        result = await client.search_notes("测试关键词", max_retries=3)

    assert result is not None
    assert "data" in result
    assert mock_session.call_count == 2  # 第一次 429，第二次成功
    print(f"✓ 429 重试成功，共调用 {mock_session.call_count} 次")

    await client.close()


async def test_search_notes_429_exhaust_retries():
    """测试 429 重试次数耗尽"""
    print("\n=== 测试: 429 重试次数耗尽 ===")

    client = TikHubXHSClient("test_token")
    await client.start()

    # Mock 响应: 始终 429
    mock_session = MockSession(responses=[
        {"status": 429, "data": {}, "headers": {"Retry-After": "1"}}
    ])

    with patch.object(client._session, 'get', mock_session.get):
        result = await client.search_notes("测试关键词", max_retries=2)
        # 重试耗尽后，函数会返回 None（循环结束后没有返回值）
        assert result is None, f"应该返回 None，但返回了: {result}"
        print(f"✓ 重试耗尽正确返回 None，共尝试 {mock_session.call_count} 次")

    await client.close()


async def test_get_comments_429_retry():
    """测试获取评论 429 重试"""
    print("\n=== 测试: 获取评论 429 重试 ===")

    client = TikHubXHSClient("test_token")
    await client.start()

    # Mock 响应序列: 429 -> 成功
    mock_session = MockSession(responses=[
        {"status": 429, "data": {}, "headers": {"Retry-After": "3"}},
        {"status": 200, "data": MOCK_COMMENTS_RESPONSE}
    ])

    with patch.object(client._session, 'get', mock_session.get):
        result = await client.get_note_comments("note123", max_retries=3)

    assert result is not None
    assert "data" in result
    assert mock_session.call_count == 2
    print(f"✓ 评论 429 重试成功，共调用 {mock_session.call_count} 次")

    await client.close()


# ============================================================================
# XHSMCPServer 测试
# ============================================================================

async def test_batch_get_comments_serial():
    """测试批量获取评论（串行执行）"""
    print("\n=== 测试: 批量获取评论（串行） ===")

    server = XHSMCPServer("test_token")
    await server.start()

    # 记录调用时间
    call_times = []

    # 创建正确的响应格式（模拟 get_note_comments 的返回值）
    def make_mock_comment_response(note_id):
        comments_list = MOCK_COMMENTS_RESPONSE["data"]["data"]["comments"]
        return {
            "success": True,
            "note_id": note_id,
            "comments": comments_list,
            "total_count": len(comments_list),
            "execution_time": 0.1
        }

    original_get_note_comments = server.get_note_comments

    async def timed_get_comments(note_id, limit=50):
        call_times.append(datetime.now())
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return make_mock_comment_response(note_id)

    # Patch server's get_note_comments method
    server.get_note_comments = timed_get_comments

    # 测试 3 个笔记
    note_ids = ["note1", "note2", "note3"]
    result = await server.batch_get_comments(
        note_ids=note_ids,
        comments_per_note=20,
        delay_between_requests=0.2
    )

    assert result["success"], "批量获取应该成功"
    # MOCK_COMMENTS_RESPONSE 中有 1 条评论，3 个笔记 = 3 条评论
    assert result["total_comments"] == 3, f"预期 3 条评论，实际: {result['total_comments']}"
    assert len(result["results"]) == 3, f"预期 3 个笔记的结果"

    # 验证串行执行（有延迟）
    if len(call_times) >= 2:
        delay_1_2 = (call_times[1] - call_times[0]).total_seconds()
        assert delay_1_2 >= 0.2, f"延迟不足: {delay_1_2}s"
        print(f"✓ 串行执行验证通过，延迟: {delay_1_2:.2f}s")

    # 恢复原方法
    server.get_note_comments = original_get_note_comments
    await server.stop()


# ============================================================================
# 综合测试
# ============================================================================

async def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("XHS API 速率限制测试套件")
    print("=" * 60)

    tests = [
        ("搜索笔记成功", test_search_notes_success),
        ("搜索笔记 429 重试", test_search_notes_429_retry),
        ("搜索笔记重试耗尽", test_search_notes_429_exhaust_retries),
        ("获取评论 429 重试", test_get_comments_429_retry),
        ("批量获取评论串行", test_batch_get_comments_serial),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ 测试失败: {name}")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
