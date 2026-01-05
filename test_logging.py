"""
简单测试脚本 - 验证日志功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.logging_config import setup_logging, get_logger

# 测试日志功能
def test_logging():
    """测试日志功能"""

    print("=" * 60)
    print("测试日志功能")
    print("=" * 60)

    # 1. 设置日志
    print("\n1. 初始化日志系统...")
    setup_logging(log_level="DEBUG")

    # 2. 获取logger
    print("2. 创建logger...")
    logger = get_logger("test")
    request_logger = __import__('agents.logging_config', fromlist=['RequestLogger']).RequestLogger(logger)

    # 3. 测试不同级别的日志
    print("3. 测试不同级别的日志...\n")
    logger.debug("这是 DEBUG 级别日志")
    logger.info("这是 INFO 级别日志")
    logger.warning("这是 WARNING 级别日志")
    logger.error("这是 ERROR 级别日志")

    # 4. 测试请求日志
    print("\n4. 测试请求日志...\n")
    request_logger.log_request(
        api_name="TEST.API",
        method="POST",
        url="https://api.example.com/test",
        params={"keyword": "测试", "page": 1},
        body={"test": "data"}
    )

    import time
    time.sleep(0.1)

    request_logger.log_response(
        api_name="TEST.API",
        status=200,
        body={"result": "success", "count": 10},
        duration_ms=123.45
    )

    # 5. 测试错误日志
    print("5. 测试错误日志...\n")
    request_logger.log_request(
        api_name="TEST.ERROR",
        method="GET",
        url="https://api.example.com/error"
    )

    time.sleep(0.05)

    request_logger.log_response(
        api_name="TEST.ERROR",
        error="Connection timeout",
        duration_ms=5000
    )

    # 6. 检查日志文件
    print("\n6. 检查日志文件...")
    log_dir = Path(__file__).parent / "logs"
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        print(f"   日志目录: {log_dir.absolute()}")
        print(f"   找到 {len(log_files)} 个日志文件:")
        for f in sorted(log_files):
            size = f.stat().st_size
            print(f"   - {f.name} ({size} 字节)")
    else:
        print(f"   警告: 日志目录不存在: {log_dir.absolute()}")

    print("\n" + "=" * 60)
    print("测试完成! 请检查 agent_system/logs/ 目录")
    print("=" * 60)

if __name__ == "__main__":
    test_logging()
