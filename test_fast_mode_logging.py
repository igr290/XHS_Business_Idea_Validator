"""
Test script to verify fast mode logging
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.logging_config import setup_logging
from agents.subagents.keyword_agent import KeywordAgent
from agents.context_store import ContextStore
from agents.config import ConfigManager


async def test_fast_mode_keyword_generation():
    """Test keyword generation in fast mode"""

    print("=" * 70)
    print("Testing Fast Mode Keyword Generation")
    print("=" * 70)

    # Initialize logging
    setup_logging(log_level="DEBUG")

    # Initialize agent
    config = ConfigManager()
    context_store = ContextStore()

    agent = KeywordAgent(
        config=config,
        context_store=context_store,
        mcp_clients={}
    )

    # Test with fast mode enabled
    print("\n1. Testing with use_user_input_as_keyword=True")
    result = await agent.execute(
        task="generate",
        context={
            "business_idea": "测试业务创意描述"
        },
        use_user_input_as_keyword=True
    )

    print(f"   Success: {result.success}")
    print(f"   Keywords: {result.data.get('keywords')}")
    print(f"   Source: {result.data.get('source')}")
    print(f"   Count: {result.data.get('count')}")

    # Verify result
    if result.success:
        keywords = result.data.get("keywords", [])
        if keywords == ["测试业务创意描述"] and result.data.get("source") == "user_input":
            print("   ✅ Fast mode working correctly!")
        else:
            print(f"   ❌ Fast mode failed! Expected ['测试业务创意描述'], got {keywords}")
    else:
        print(f"   ❌ Generation failed: {result.error}")

    # Test with fast mode disabled (LLM generation)
    print("\n2. Testing with use_user_input_as_keyword=False (LLM mode)")
    print("   (This would require LLM server, skipping for now)")

    print("\n" + "=" * 70)
    print("Test complete! Check logs for debug output.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_fast_mode_keyword_generation())
