# test.py 完整测试代码
import sys
import os

# 打印 Python 版本（确认使用的是嵌入式 3.12.9）
print(f"Python 版本：{sys.version}")
# 打印当前工作目录
print(f"当前工作目录：{os.getcwd()}")
# 打印 sys.path（检查路径是否包含需要的目录）
print("\nPython 搜索路径：")
for i, path in enumerate(sys.path[:8]):  # 只打印前8个，避免过长
    print(f"{i+1}. {path}")

# 尝试导入 ui 模块（验证之前的路径问题是否解决）
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, "src"))  # 加入 src 目录
    from ui.main_ui import PhantomUI
    print("\n✅ ui 模块导入成功！")
except ModuleNotFoundError as e:
    print(f"\n❌ 模块导入失败：{e}")
    print("请检查 ui 目录是否在 src 下，或路径配置是否正确")