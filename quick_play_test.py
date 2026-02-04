#!/usr/bin/env python3
"""
快速流式播放测试脚本

简单易用的流式 TTS 实时播放测试
"""

import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_streaming_play import stream_and_play


def main():
    """快速测试菜单"""

    print("\n" + "="*60)
    print("流式 TTS 实时播放 - 快速测试")
    print("="*60)

    # 默认配置
    url = "http://localhost:13650"

    # 测试文本选项
    test_texts = {
        "1": ("简短测试", "你好！"),
        "2": ("中等长度", "今天的天气确实不错，适合户外活动呢！"),
        "3": ("长文本", "今天是一个美好的日子，阳光明媚，鸟语花香。让我们一起出去走走，感受大自然的美好吧！"),
        "4": ("自定义", None)
    }

    print("\n请选择测试文本：")
    for key, (desc, _) in test_texts.items():
        print(f"  {key}. {desc}")

    choice = input("\n请输入选项 (1-4，默认 2): ").strip() or "2"

    if choice == "4":
        text = input("请输入要转换的文本: ").strip()
        if not text:
            print("错误：文本不能为空")
            return
    elif choice in test_texts:
        text = test_texts[choice][1]
    else:
        print("无效选项，使用默认文本")
        text = test_texts["2"][1]

    # API 类型选择
    print("\n请选择 API 类型：")
    print("  1. OpenAI 兼容 API (默认)")
    print("  2. PhantomVox API")

    api_choice = input("\n请输入选项 (1-2，默认 1): ").strip() or "1"
    api_type = "openai" if api_choice == "1" else "phantomvox"

    # 如果是 PhantomVox，询问说话人
    speaker = "Vivian"
    if api_type == "phantomvox":
        print("\n请选择说话人：")
        print("  1. Vivian (女声，默认)")
        print("  2. Serena (女声)")
        print("  3. Uncle_Fu (男声)")
        print("  4. Dylan (男声)")
        print("  5. Eric (男声)")
        print("  6. Ono_Anna (女声)")

        speaker_choice = input("\n请输入选项 (1-6，默认 1): ").strip() or "1"
        speakers = {
            "1": "Vivian",
            "2": "Serena",
            "3": "Uncle_Fu",
            "4": "Dylan",
            "5": "Eric",
            "6": "Ono_Anna"
        }
        speaker = speakers.get(speaker_choice, "Vivian")

    # 预缓冲大小选择
    print("\n请选择预缓冲大小：")
    print("  1. 低延迟 - 0.5 秒")
    print("  2. 平衡 - 2 秒 (默认)")
    print("  3. 更稳定 - 4 秒")

    buffer_choice = input("\n请输入选项 (1-3，默认 2): ").strip() or "2"
    buffer_sizes = {"1": 12000, "2": 48000, "3": 96000}
    pre_buffer = buffer_sizes.get(buffer_choice, 48000)

    # 显示配置并确认
    print("\n" + "="*60)
    print("配置确认：")
    print(f"  服务器: {url}")
    print(f"  API: {api_type}")
    if api_type == "phantomvox":
        print(f"  说话人: {speaker}")
    print(f"  播放器: sounddevice")
    print(f"  预缓冲: {pre_buffer/24000:.1f} 秒")
    print(f"  文本: {text[:50]}...")
    print("="*60)

    confirm = input("\n确认开始测试？(Y/n): ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消")
        return

    # 开始测试
    print("\n开始测试...")

    success = stream_and_play(
        url=url,
        text=text,
        api_type=api_type,
        speaker=speaker,
        sample_rate=24000,
        pre_buffer_size=pre_buffer
    )

    if success:
        print("\n✓ 测试成功完成！")
    else:
        print("\n✗ 测试失败")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
