"""
关于页面组件 (About View)

展示应用信息、技术栈、参考链接和许可证信息
"""

import flet as ft


class AboutView(ft.Container):
    """关于页面 - 列表式布局展示应用信息"""

    def __init__(self, page: ft.Page, version: str):
        self._page = page
        self.version = version

        # UI 样式配置
        self.BStyle = ft.ButtonStyle(
            text_style=ft.TextStyle(font_family="Microsoft YaHei")
        )

        # 构建UI
        super().__init__(
            content=self._build_ui(),
            expand=True
        )

    def _build_ui(self) -> ft.Control:
        """构建主UI界面"""
        return ft.Column([
            # 标题
            ft.Text("关于 PhantomVox", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

            # 应用信息卡片
            self._build_app_info_card(),

            ft.Container(height=15),

            # 技术栈
            ft.Text("技术栈", size=18, weight=ft.FontWeight.BOLD),
            ft.Container(height=5),
            self._build_tech_stack(),

            ft.Container(height=15),

            # 参考链接
            ft.Text("参考链接", size=18, weight=ft.FontWeight.BOLD),
            ft.Container(height=5),
            self._build_links(),

            ft.Container(height=15),

            # 许可证信息
            ft.Text("许可证", size=18, weight=ft.FontWeight.BOLD),
            ft.Container(height=5),
            self._build_license(),

        ], expand=True, scroll=ft.ScrollMode.AUTO, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)

    def _build_app_info_card(self) -> ft.Card:
        """构建应用信息卡片"""
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.INFO_OUTLINE, size=32, color=ft.Colors.BLUE),
                        ft.Text("PhantomVox", size=20, weight=ft.FontWeight.BOLD),
                    ], spacing=10),
                    ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                    ft.Text("作者: 泠夜Soul", size=14),
                    ft.Text(f"版本: {self.version}", size=14),
                    ft.Text(
                        "基于 Qwen3-TTS 的本地文本转语音应用",
                        size=13,
                        color=ft.Colors.GREY_400
                    ),
                ], spacing=3),
                padding=15,
                border_radius=12
            ),
            elevation=1
        )

    def _build_tech_stack(self) -> ft.Container:
        """构建技术栈列表"""
        tech_items = [
            (ft.Icons.PALETTE, "Flet", "UI 框架", ft.Colors.BLUE),
            (ft.Icons.RECORD_VOICE_OVER, "qwen-tts", "TTS 引擎", ft.Colors.PURPLE),
            (ft.Icons.DOWNLOAD, "modelscope", "模型下载", ft.Colors.ORANGE),
            (ft.Icons.VOLUME_UP, "sounddevice", "音频播放", ft.Colors.GREEN),
            (ft.Icons.LIBRARY_MUSIC, "soundfile", "音频处理", ft.Colors.TEAL),
            (ft.Icons.CALCULATE, "numpy", "数值计算", ft.Colors.CYAN),
        ]

        list_tiles = []
        for icon, name, desc, color in tech_items:
            list_tiles.append(
                ft.ListTile(
                    leading=ft.Icon(icon, color=color),
                    title=ft.Text(name, size=14, font_family="Microsoft YaHei"),
                    subtitle=ft.Text(desc, size=12, color=ft.Colors.GREY_400),
                    content_padding=ft.padding.symmetric(horizontal=10, vertical=0),
                )
            )

        return ft.Container(
            content=ft.Column(list_tiles, spacing=0),
            padding=ft.padding.symmetric(horizontal=5),
            bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.ON_SURFACE),
            border_radius=12,
        )

    def _build_links(self) -> ft.Column:
        """构建参考链接"""
        links = [
            ("PhantomVox GitHub", "https://github.com/LingyeSoul/PhantomVox"),
            ("Qwen3-TTS GitHub", "https://github.com/QwenLM/Qwen3-TTS"),
            ("Qwen3-TTS ModelScope", "https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            ("Qwen3-TTS-streaming GitHub（dffdeeq）", "https://github.com/dffdeeq/Qwen3-TTS-streaming"),
            ("Qwen3-TTS-streaming GitHub（rekuenkdr）", "https://github.com/rekuenkdr/Qwen3-TTS-streaming"),
            ("Flet 官方文档", "https://flet.dev/docs/"),
        ]

        buttons = []
        for text, url in links:
            buttons.append(
                ft.TextButton(
                    text,
                    url=url,
                    icon=ft.Icons.LINK,
                    style=ft.ButtonStyle(
                        text_style=ft.TextStyle(font_family="Microsoft YaHei")
                    )
                )
            )

        return ft.Column(buttons, spacing=5)

    def _build_license(self) -> ft.Card:
        """构建许可证信息卡片"""
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text(
                        "GNU General Public License v3.0",
                        size=14,
                        weight=ft.FontWeight.BOLD
                    ),
                    ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                    ft.Text(
                        "致谢：",
                        size=13,
                        weight=ft.FontWeight.BOLD
                    ),
                    ft.Container(height=3),
                    ft.Text(
                        "• Qwen Team - 提供 Qwen3-TTS 模型",
                        size=12,
                        color=ft.Colors.GREY_400
                    ),
                    ft.Text(
                        "• dffdeeq - 提供 Qwen3-TTS-streaming 实现",
                        size=12,
                        color=ft.Colors.GREY_400
                    ),
                    ft.Text(
                        "• rekuenkdr - 提供改进的 Qwen3-TTS-streaming 实现",
                        size=12,
                        color=ft.Colors.GREY_400
                    ),
                    ft.Text(
                        "• Flet Team - 提供优秀的 Python UI 框架",
                        size=12,
                        color=ft.Colors.GREY_400
                    ),
                ], spacing=3),
                padding=15,
                border_radius=12
            ),
            elevation=1
        )
