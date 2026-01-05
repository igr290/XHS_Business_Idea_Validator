"""
测试 TagSystemGeneration 模型
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.business_models import TagSystemGeneration

def test_model():
    """测试模型导入和 schema 生成"""
    print("✓ TagSystemGeneration 模型导入成功")

    # 测试 schema 生成
    schema = TagSystemGeneration.model_json_schema()
    print("✓ Schema 生成成功")

    # 检查字段名
    field_names = list(schema['properties'].keys())
    print(f"✓ 字段名: {field_names}")

    # 预期的字段名
    expected_fields = ['人群场景', '功能价值', '保障价值', '体验价值']
    for field in expected_fields:
        if field in field_names:
            print(f"✓ 字段 '{field}' 存在")
        else:
            print(f"✗ 字段 '{field}' 缺失")
            return False

    # 测试创建实例
    test_data = {
        "人群场景": {
            "用户需求与痛点-痛点问题": ["测试标签1", "测试标签2"]
        },
        "功能价值": {},
        "保障价值": {},
        "体验价值": {}
    }

    instance = TagSystemGeneration(**test_data)
    print("✓ 模型实例创建成功")

    # 测试 model_dump
    dumped = instance.model_dump()
    print("✓ model_dump() 方法工作正常")

    print("\n所有测试通过！")
    return True

if __name__ == "__main__":
    try:
        success = test_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
