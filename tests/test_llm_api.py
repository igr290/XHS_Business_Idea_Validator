"""
LLM API 诊断测试

用于检查 OpenAI API 配置是否正确
"""
import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


async def test_env_vars():
    """测试环境变量是否正确加载"""
    print("=" * 60)
    print("步骤 1: 检查环境变量")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    print(f"OPENAI_API_KEY: {'***' + api_key[-4:] if api_key and len(api_key) > 4 else '未设置'}")
    print(f"OPENAI_BASE_URL: {base_url}")

    if not api_key or api_key == "your_openai_api_key_here":
        print("\n❌ 错误: OPENAI_API_KEY 未配置或使用默认值")
        print("   请在 agent_system/.env 文件中设置正确的 OPENAI_API_KEY")
        return False

    if not api_key.startswith("sk-"):
        print("\n⚠️  警告: OPENAI_API_KEY 格式可能不正确")
        print("   OpenAI API Key 通常以 'sk-' 开头")

    print("\n✓ 环境变量检查通过\n")
    return True


async def test_api_connection():
    """测试 API 连接"""
    print("=" * 60)
    print("步骤 2: 测试 API 连接")
    print("=" * 60)

    from mcp_servers.llm_server import create_llm_mcp_server

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

    print(f"配置: model={model_name}, base_url={base_url}")

    try:
        # 创建服务器
        server = await create_llm_mcp_server(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )

        # 测试连接
        print("\n正在发送测试请求...")
        result = await server.test_connection()

        # 关闭服务器
        await server.stop()

        if result.get("success"):
            print(f"\n✅ {result.get('message')}")
            return True
        else:
            print(f"\n❌ 连接失败: {result.get('message')}")
            if result.get("error_type"):
                print(f"   错误类型: {result.get('error_type')}")
            if result.get("error"):
                print(f"   错误详情: {result.get('error')}")
            return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_generation():
    """测试简单的文本生成"""
    print("\n" + "=" * 60)
    print("步骤 3: 测试文本生成")
    print("=" * 60)

    from mcp_servers.llm_server import create_llm_mcp_server

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    try:
        server = await create_llm_mcp_server(
            api_key=api_key,
            base_url=base_url
        )

        print("\n正在生成测试文本...")
        result = await server.generate_text(
            prompt="用一句话介绍你自己",
            max_tokens=50
        )

        await server.stop()

        if result.get("success"):
            print(f"\n✅ 文本生成成功:")
            print(f"   {result.get('text')}")
            print(f"   耗时: {result.get('execution_time', 0):.2f}s")
            return True
        else:
            print(f"\n❌ 生成失败: {result.get('error')}")
            return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("LLM API 诊断工具")
    print("=" * 60)
    print(f"工作目录: {os.getcwd()}")
    print(f".env 路径: {env_path}")
    print(f".env 存在: {env_path.exists()}\n")

    # 步骤 1: 检查环境变量
    env_ok = await test_env_vars()
    if not env_ok:
        return False

    # 步骤 2: 测试 API 连接
    conn_ok = await test_api_connection()
    if not conn_ok:
        return False

    # 步骤 3: 测试文本生成
    gen_ok = await test_simple_generation()

    # 总结
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)
    print(f"环境变量: {'✅ 通过' if env_ok else '❌ 失败'}")
    print(f"API 连接: {'✅ 通过' if conn_ok else '❌ 失败'}")
    print(f"文本生成: {'✅ 通过' if gen_ok else '❌ 失败'}")

    if env_ok and conn_ok and gen_ok:
        print("\n🎉 所有测试通过！API 配置正常。")
        return True
    else:
        print("\n⚠️  部分测试失败，请根据上面的错误信息修复配置。")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
