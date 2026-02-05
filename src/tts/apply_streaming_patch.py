"""
应用 Qwen3-TTS-streaming 的修改到官方 qwen-tts

基于第三方 Qwen3-TTS-streaming 项目的修改
提取关键代码并作为 monkey patch 应用
https://github.com/rekuenkdr/Qwen3-TTS-streaming
https://github.com/dffdeeq/Qwen3-TTS-streaming
"""

import torch
import numpy as np
from typing import Optional, Generator


# ========================================
# Decoder 优化实现（来自 streaming 项目）
# ========================================

def _decoder_forward_impl(self, codes):
    """Decoder 内部前向传播实现（用于编译）"""
    hidden = self.quantizer.decode(codes)
    hidden = self.pre_conv(hidden).transpose(1, 2)
    hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
    hidden = hidden.permute(0, 2, 1)
    for blocks in self.upsample:
        for block in blocks:
            hidden = block(hidden)
    wav = hidden
    for block in self.decoder:
        wav = block(wav)
    return wav.clamp(min=-1, max=1)


def _decoder_compile_for_streaming(self, mode: str = "reduce-overhead", backend: str = "inductor"):
    """
    对前向传播应用 torch.compile 以实现更快的流式解码。

    Args:
        mode: 编译模式
            - "reduce-overhead"（推荐）：使用内部 CUDA 图，最适合流式
            - "max-autotune"：最大优化，编译时间更长
            - "default"：良好的平衡，无内部 CUDA 图
        backend: 编译后端（推荐 "inductor"）
    """
    if not hasattr(torch, 'compile'):
        print("[Decoder] torch.compile 不可用（需要 PyTorch 2.0+）")
        return self

    print(f"[Decoder] 正在编译前向传播，mode={mode}, backend={backend}...")
    print(f"[Decoder] 注意：mode='reduce-overhead' 自动包含 CUDA 图优化")

    self._compiled_forward = torch.compile(
        self._forward_impl,
        mode=mode,
        fullgraph=False,
        dynamic=False,
        backend=backend,
    )
    self._compile_mode = mode
    print("[Decoder] 编译完成")
    return self


def _decoder_capture_cuda_graph(self, window_size: int = 80, warmup_runs: int = 3):
    """
    为固定窗口大小捕获 CUDA 图。

    CUDA 图通过捕获和重放 GPU 操作来消除 CPU 开销，
    最适合固定 decode_window_frames 的流式场景。

    警告：不要在 torch.compile mode='reduce-overhead' 时使用此方法，
    因为该模式已内部使用 CUDA 图，会冲突。
    仅在 mode='default' 或不使用 torch.compile 时使用。

    Args:
        window_size: 固定的 codec frames 数量（必须匹配 decode_window_frames）
        warmup_runs: 捕获前的预热迭代次数
    """
    if not torch.cuda.is_available():
        print("[Decoder] CUDA 不可用，跳过 CUDA 图捕获")
        return self

    # 检查与 torch.compile reduce-overhead 模式的冲突
    if self._compiled_forward is not None and self._compile_mode == 'reduce-overhead':
        print("[Decoder] 警告：torch.compile mode='reduce-overhead' 已内部使用 CUDA 图。")
        print("[Decoder] 跳过手动 CUDA 图捕获以避免冲突。")
        print("[Decoder] 将使用已编译的前向传播（已优化）。")
        return self

    device = next(self.parameters()).device
    num_quantizers = self.config.num_quantizers

    # 创建静态输入缓冲区
    self._static_input = torch.zeros(
        1, num_quantizers, window_size,
        dtype=torch.long,
        device=device
    )
    self._graph_window_size = window_size

    # 使用非编译的前向传播进行手动 CUDA 图捕获
    forward_fn = self._forward_impl

    # 预热
    print(f"[Decoder] 正在预热 CUDA 图（window_size={window_size}）...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
        for _ in range(warmup_runs):
            _ = forward_fn(self._static_input)
    torch.cuda.current_stream().wait_stream(s)

    # 捕获
    print("[Decoder] 正在捕获 CUDA 图...")
    self._cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self._cuda_graph):
        self._static_output = forward_fn(self._static_input)

    print("[Decoder] CUDA 图捕获成功")
    return self


def _decoder_forward_optimized(self, codes):
    """
    优化的前向传播（如果可用）。
    优先级：
    1. 手动 CUDA 图（如果已捕获且输入匹配大小）
    2. 编译的前向传播（torch.compile，可能包含内部 CUDA 图）
    3. 常规前向传播

    注意：使用 compile_mode="reduce-overhead" 时，编译的前向传播
    已包含 CUDA 图优化。
    """
    B, Q, T = codes.shape

    # 尝试手动 CUDA 图路径（仅当我们捕获了图）
    if (self._cuda_graph is not None
        and B == 1
        and T == self._graph_window_size):
        self._static_input.copy_(codes)
        self._cuda_graph.replay()
        return self._static_output.clone()

    # 使用编译的前向传播（如果可用，包含 reduce-overhead 的 CUDA 图）
    if self._compiled_forward is not None:
        # 标记步骤开始以避免张量覆盖错误
        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_forward(codes)

    # 回退到常规前向传播
    return self._forward_impl(codes)


def _decoder_decode_padded(self, codes: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    使用左填充到固定大小进行解码，以优化 torch.compile。

    当使用 torch.compile 且 dynamic=False 时，模型会对每个新的输入大小重新编译。
    通过将所有输入填充到 target_length，确保单次编译可重用于所有流式解码调用。

    Args:
        codes: 输入张量 [B, Q, T]，其中 T <= target_length
        target_length: 填充到的固定大小（应为 decode_window_frames）

    Returns:
        波形张量，左侧填充样本被修剪
    """
    B, _, T = codes.shape

    if T < target_length:
        # 左侧用零填充
        pad = torch.zeros(B, codes.shape[1], target_length - T, dtype=codes.dtype, device=codes.device)
        codes_padded = torch.cat([pad, codes], dim=-1)
    else:
        codes_padded = codes.contiguous()  # 确保张量格式统一以避免重新编译

    # 运行前向传播（如果可用则使用编译路径）
    wav = self.forward_optimized(codes_padded)

    # 从输出修剪填充
    if T < target_length:
        # 计算对应于填充帧的样本数
        total_samples = wav.shape[-1]
        samples_per_frame = total_samples / target_length
        trim_samples = int((target_length - T) * samples_per_frame)
        wav = wav[..., trim_samples:]

    return wav


# ========================================
# Tokenizer 优化实现
# ========================================

def _tokenizer_enable_streaming_optimizations(
    self,
    decode_window_frames: int = 80,
    use_compile: bool = True,
    use_cuda_graphs: bool = False,
    compile_mode: str = "reduce-overhead",
):
    """
    为流式解码启用优化。

    此方法对 decoder 应用 torch.compile 以实现更快的流式生成。

    重要：compile_mode="reduce-overhead"（默认）已包含 CUDA 图优化
    内部。您不需要在设置此模式时同时设置 use_cuda_graphs=True。
    手动 CUDA 图仅在 compile_mode="default" 时有用。

    Args:
        decode_window_frames: 流式解码的窗口大小（用于手动 CUDA 图）
        use_compile: 对 decoder 应用 torch.compile（推荐）
        use_cuda_graphs: 捕获手动 CUDA 图（仅与 compile_mode="default" 结合使用）
        compile_mode: torch.compile 的模式
            - "reduce-overhead"（推荐）：自动包含 CUDA 图
            - "max-autotune"：最大优化
            - "default"：基础编译，可与手动 CUDA 图结合

    Returns:
        self 以支持方法链式调用

    Example:
        # 推荐：仅使用 torch.compile with reduce-overhead
        model.speech_tokenizer.model.enable_streaming_optimizations(
            use_compile=True,
            compile_mode="reduce-overhead",
        )
    """
    print(f"[Tokenizer] 正在启用流式优化...")
    print(f"  use_compile={use_compile}, compile_mode={compile_mode}")
    print(f"  use_cuda_graphs={use_cuda_graphs} (手动)")

    if use_compile:
        self.decoder.compile_for_streaming(mode=compile_mode)

    # 仅在明确请求且未使用 reduce-overhead 时捕获手动 CUDA 图
    if use_cuda_graphs:
        if compile_mode == "reduce-overhead":
            print(f"[Tokenizer] 注意：compile_mode='reduce-overhead' 已包含 CUDA 图")
            print(f"[Tokenizer] 跳过手动 CUDA 图捕获（不需要）")
        else:
            self.decoder.capture_cuda_graph(window_size=decode_window_frames)

    return self


def _tokenizer_decode_streaming(
    self,
    audio_codes: torch.Tensor,
    use_optimized: bool = True,
    pad_to_size: Optional[int] = None,
) -> torch.Tensor:
    """
    针对流式优化的音频代码解码（单个窗口）。

    与常规 decode() 不同，此方法：
    - 不使用分块解码（假设小窗口）
    - 如果可用则使用 CUDA 图
    - 返回原始张量（不是列表）
    - 可选择填充到固定大小以优化 torch.compile

    Args:
        audio_codes: [B, T, num_quantizers] 形状的 codec 索引张量
        use_optimized: 如果为 True，在可用时使用 CUDA 图路径
        pad_to_size: 如果指定，填充输入到此大小（帧数）以实现一致的
                    torch.compile 行为。应匹配流式的 decode_window_frames。

    Returns:
        波形张量 [B, samples]
    """
    # 转置为 [B, num_quantizers, T] 以供 decoder 使用
    codes = audio_codes.transpose(1, 2)

    if use_optimized:
        if pad_to_size is not None:
            wav = self.decoder.decode_padded(codes, pad_to_size)
        else:
            wav = self.decoder.forward_optimized(codes)
    else:
        wav = self.decoder(codes)

    return wav.squeeze(1)


# ========================================
# 辅助函数（来自 streaming 项目）
# ========================================

def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Apply top-k and top-p (nucleus) filtering to logits."""
    if top_k > 0:
        topk = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        min_keep = topk.values[..., -1, None]
        logits = torch.where(logits < min_keep, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        inv_idx = torch.argsort(sorted_idx, dim=-1)
        logits = torch.gather(sorted_logits, dim=-1, index=inv_idx)
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    suppress_tokens: Optional[list[int]] = None,
) -> torch.Tensor:
    """Sample next token from logits with temperature, top-k, top-p and token suppression."""
    if suppress_tokens is not None and len(suppress_tokens) > 0:
        logits = logits.clone()
        logits[..., suppress_tokens] = float("-inf")

    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _crossfade(prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
    """Crossfade between end of previous chunk and start of new chunk using Hann window."""
    n = min(len(prev_tail), len(new_head))
    if n <= 0:
        return new_head
    t = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    fade_in = 0.5 * (1 - np.cos(np.pi * t))
    fade_out = 1 - fade_in
    return prev_tail[:n] * fade_out + new_head[:n] * fade_in


# Default blend samples for boundary blending
# ~21ms at 24kHz, matches RMS check window for better coverage
# Lower values may cause clicks, set to 0 to disable
DEFAULT_BLEND_SAMPLES = 512


def _add_ref_code_context(
    window_codes: torch.Tensor,
    ref_code_context: Optional[torch.Tensor],
    ref_code_frames: int,
    decode_window_frames: int,
) -> tuple[torch.Tensor, int]:
    """Add ref_code as context prefix when window doesn't fill decode_window_frames."""
    if ref_code_context is None or window_codes.shape[0] >= decode_window_frames:
        return window_codes, 0

    available_space = decode_window_frames - window_codes.shape[0]
    ref_prefix_frames = min(available_space, ref_code_frames)

    if ref_prefix_frames > 0:
        ref_prefix = ref_code_context[-ref_prefix_frames:]
        return torch.cat([ref_prefix, window_codes], dim=0), ref_prefix_frames

    return window_codes, 0


# ========================================
# Monkey patch 应用函数
# ========================================

def apply_streaming_patch_to_qwen_tts():
    """
    将 streaming 修改应用到 qwen_tts 库

    这会向以下类添加方法：
    1. Qwen3TTSForConditionalGeneration:
       - _build_talker_inputs 辅助方法
       - stream_generate_pcm 核心流式方法
       - enable_streaming_optimizations 优化方法
    2. Qwen3TTSTokenizerV2Decoder:
       - compile_for_streaming 编译优化
       - capture_cuda_graph CUDA 图捕获
       - forward_optimized 优化的前向传播
       - decode_padded 固定大小解码
    3. Qwen3TTSTokenizerV2Model:
       - enable_streaming_optimizations 启用流式优化
       - decode_streaming 流式解码
    """
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

    print("[StreamingPatch] 正在应用 streaming 修改到 qwen-tts...")

    # ========== Patch Qwen3TTSForConditionalGeneration ==========
    # 添加 _build_talker_inputs 辅助方法
    Qwen3TTSForConditionalGeneration._build_talker_inputs = _build_talker_inputs_impl

    # 添加 stream_generate_pcm 方法
    Qwen3TTSForConditionalGeneration.stream_generate_pcm = _stream_generate_pcm_impl

    # 添加 enable_streaming_optimizations 方法（如果还没有）
    if not hasattr(Qwen3TTSForConditionalGeneration, 'enable_streaming_optimizations'):
        Qwen3TTSForConditionalGeneration.enable_streaming_optimizations = _enable_streaming_optimizations_impl

    # ========== Patch Qwen3TTSTokenizerV2Decoder ==========
    # 初始化优化状态
    if not hasattr(Qwen3TTSTokenizerV2Decoder, '_compiled_forward'):
        Qwen3TTSTokenizerV2Decoder._compiled_forward = None
        Qwen3TTSTokenizerV2Decoder._compile_mode = None
        Qwen3TTSTokenizerV2Decoder._cuda_graph = None
        Qwen3TTSTokenizerV2Decoder._static_input = None
        Qwen3TTSTokenizerV2Decoder._static_output = None
        Qwen3TTSTokenizerV2Decoder._graph_window_size = None

    # 添加 decoder 优化方法
    Qwen3TTSTokenizerV2Decoder.compile_for_streaming = _decoder_compile_for_streaming
    Qwen3TTSTokenizerV2Decoder.capture_cuda_graph = _decoder_capture_cuda_graph
    Qwen3TTSTokenizerV2Decoder.forward_optimized = _decoder_forward_optimized
    Qwen3TTSTokenizerV2Decoder.decode_padded = _decoder_decode_padded
    Qwen3TTSTokenizerV2Decoder._forward_impl = _decoder_forward_impl

    # ========== Patch Qwen3TTSTokenizerV2Model ==========
    # 添加 tokenizer 层面的优化方法
    Qwen3TTSTokenizerV2Model.enable_streaming_optimizations = _tokenizer_enable_streaming_optimizations
    Qwen3TTSTokenizerV2Model.decode_streaming = _tokenizer_decode_streaming

    print("[StreamingPatch] ✓ 修改应用成功（包含 decoder 优化）")
    return Qwen3TTSForConditionalGeneration


# ========================================
# _build_talker_inputs 实现
# ========================================

def _build_talker_inputs_impl(
    self,
    input_ids: list[torch.Tensor],
    instruct_ids: Optional[list[torch.Tensor]],
    ref_ids: Optional[list[torch.Tensor]],
    voice_clone_prompt: Optional[list[dict]],
    languages: list[str],
    speakers: Optional[list[str]],
    non_streaming_mode: bool = False,
):
    """
    Build talker input embeddings, attention mask, trailing text hiddens and tts_pad_embed.

    基于 dffdeeq/Qwen3-TTS-streaming 项目的实现
    """
    # 辅助函数：获取 codec embedding（兼容两种版本）
    def get_codec_embedding(token_ids):
        if hasattr(self.talker, 'codec_embedding'):
            return self.talker.codec_embedding(token_ids)
        else:
            return self.talker.get_input_embeddings()(token_ids)

    talker_input_embeds = [[] for _ in range(len(input_ids))]

    voice_clone_spk_embeds = None
    if voice_clone_prompt is not None:
        voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

    if instruct_ids is not None:
        for index, instruct_id in enumerate(instruct_ids):
            if instruct_id is not None:
                talker_input_embeds[index].append(self.talker.text_projection(
                    self.talker.get_text_embeddings()(instruct_id)))

    trailing_text_hiddens = []
    if speakers is None:
        speakers = [None] * len(input_ids)

    for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
        if voice_clone_spk_embeds is None:
            if speaker == "" or speaker is None:
                speaker_embed = None
            else:
                if speaker.lower() not in self.config.talker_config.spk_id:
                    raise NotImplementedError(f"Speaker {speaker} not implemented")
                else:
                    spk_id = self.config.talker_config.spk_id[speaker.lower()]
                    speaker_embed = get_codec_embedding(
                        torch.tensor(
                            spk_id,
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
        else:
            if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                speaker_embed = voice_clone_spk_embeds[index]
            else:
                speaker_embed = None

        assert language is not None

        if language.lower() == "auto":
            language_id = None
        else:
            if language.lower() not in self.config.talker_config.codec_language_id:
                raise NotImplementedError(f"Language {language} not implemented")
            else:
                language_id = self.config.talker_config.codec_language_id[language.lower()]

        if (language.lower() in ["chinese", "auto"] and
                speaker != "" and speaker is not None and
                self.config.talker_config.spk_is_dialect[speaker.lower()] != False):
            dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
            language_id = self.config.talker_config.codec_language_id[dialect]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(
                torch.tensor(
                    [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        if language_id is None:
            codec_prefill_list = [[
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                self.config.talker_config.codec_think_id,
                self.config.talker_config.codec_think_bos_id,
                language_id,
                self.config.talker_config.codec_think_eos_id,
            ]]

        codec_input_embedding_0 = get_codec_embedding(
            torch.tensor(
                codec_prefill_list,
                device=self.talker.device,
                dtype=input_id.dtype,
            )
        )
        codec_input_embedding_1 = get_codec_embedding(
            torch.tensor(
                [[
                    self.config.talker_config.codec_pad_id,
                    self.config.talker_config.codec_bos_id,
                ]],
                device=self.talker.device,
                dtype=input_id.dtype,
            )
        )

        if speaker_embed is None:
            codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
        else:
            codec_input_embedding = torch.cat([codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1], dim=1)

        _talker_input_embed_role = self.talker.text_projection(
            self.talker.get_text_embeddings()(input_id[:, :3])
        )

        _talker_input_embed = torch.cat(
            (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed),
            dim=1
        ) + codec_input_embedding[:, :-1]

        talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

        if voice_clone_prompt is not None and voice_clone_prompt["ref_code"] is not None and voice_clone_prompt["icl_mode"][index]:
            icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_ids[index][:, 3:-2],
                ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        else:
            talker_input_embed = torch.cat([
                talker_input_embed,
                self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_embedding[:, -1:]
            ], dim=1)

            if non_streaming_mode:
                talker_input_embed = talker_input_embed[:, :-1]
                talker_input_embed = torch.cat([
                    talker_input_embed,
                    torch.cat(
                        (self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:-5])), tts_eos_embed),
                        dim=1
                    ) + get_codec_embedding(
                        torch.tensor(
                            [[self.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    ),
                    tts_pad_embed + get_codec_embedding(
                        torch.tensor(
                            [[self.config.talker_config.codec_pad_id, self.config.talker_config.codec_bos_id]],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
                ], dim=1)
                trailing_text_hidden = tts_pad_embed
            else:
                trailing_text_hidden = torch.cat(
                    (self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 4:-5])), tts_eos_embed),
                    dim=1
                )

        talker_input_embeds[index].append(talker_input_embed)
        trailing_text_hiddens.append(trailing_text_hidden)

    for index, talker_input_embed in enumerate(talker_input_embeds):
        talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

    original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
    sequences = [t.squeeze(0) for t in talker_input_embeds]
    sequences_reversed = [t.flip(dims=[0]) for t in sequences]
    padded_reversed = torch.nn.utils.rnn.pad_sequence(
        sequences_reversed,
        batch_first=True,
        padding_value=0.0
    )
    talker_input_embeds = padded_reversed.flip(dims=[1])
    batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
    indices = torch.arange(max_len).expand(batch_size, -1)
    num_pads = max_len - original_lengths
    talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)

    pad_embedding_vector = tts_pad_embed.squeeze()
    sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
    trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
    padded_hiddens = torch.nn.utils.rnn.pad_sequence(
        sequences_to_pad,
        batch_first=True,
        padding_value=0.0
    )
    arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
        len(trailing_text_original_lengths), -1
    )
    lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
    padding_mask = arange_tensor >= lengths_tensor
    padded_hiddens[padding_mask] = pad_embedding_vector
    trailing_text_hiddens = padded_hiddens

    return talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed


# ========================================
# stream_generate_pcm 实现（核心流式方法）
# ========================================

def _stream_generate_pcm_impl(
    self,
    input_ids: list[torch.Tensor],
    instruct_ids: Optional[list[torch.Tensor]] = None,
    ref_ids: Optional[list[torch.Tensor]] = None,
    voice_clone_prompt: Optional[list[dict]] = None,
    languages: Optional[list[str]] = None,
    speakers: Optional[list[str]] = None,
    non_streaming_mode: bool = False,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 0.9,
    subtalker_dosample: bool = True,
    subtalker_top_k: int = 50,
    subtalker_top_p: float = 1.0,
    subtalker_temperature: float = 0.9,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    overlap_samples: int = 0,
    max_frames: int = 10000,
    use_optimized_decode: bool = True,
    # Two-phase streaming: aggressive first chunk
    first_chunk_emit_every: int = 0,  # 0 = disabled, use emit_every_frames throughout
    first_chunk_decode_window: int = 48,
    first_chunk_frames: int = 48,  # Switch to stable after this many frames
    speed_factor: float = 1.0,
    pitch_factor: float = 1.0,
    **kwargs  # 接受其他未知参数
) -> Generator[tuple[np.ndarray, int], None, None]:
    """
    Stream audio generation, yielding PCM chunks as they are generated.

    基于 dffdeeq/Qwen3-TTS-streaming 项目的实现

    Args:
        first_chunk_emit_every: Emit interval for first chunk phase (0 = disabled, use emit_every_frames)
        first_chunk_decode_window: Decode window size for first chunk phase
        first_chunk_frames: Switch to stable settings after this many frames
    """
    # 注意：speed_factor 和 pitch_factor 参数目前未使用
    # 流式生成基于原始 token 序列，不支持变速/变调
    # 如果需要这些功能，请使用非流式的 generate_* 方法
    # Build talker inputs
    talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed = \
        self._build_talker_inputs(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
        )

    # Multiple EOS tokens that can terminate generation
    eos_ids = {
        self.config.talker_config.codec_eos_token_id,  # Primary codec EOS
        2150,    # Codec EOS (model-specific)
        2157,    # Secondary codec token
        151670,  # TTS special token
        self.config.tts_eos_token_id,   # 151673
        self.config.im_end_token_id,    # 151645
        151643,  # TTS special token
    }

    # Build suppress_tokens list (exclude all EOS tokens)
    vocab_size = self.config.talker_config.vocab_size
    suppress_tokens = [
        i for i in range(vocab_size - 1024, vocab_size)
        if i not in eos_ids
    ]

    torch.compiler.cudagraph_mark_step_begin()

    # Prefill: single forward pass to initialize KV cache
    out = self.talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=talker_attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
        subtalker_dosample=subtalker_dosample,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
        subtalker_temperature=subtalker_temperature,
    )

    past_key_values = out.past_key_values
    past_hidden = out.past_hidden
    generation_step = out.generation_step

    # Sample first token from prefill logits
    last_logits = out.logits[:, -1, :]
    if do_sample:
        token = _sample_next_token(last_logits, temperature, top_k, top_p, suppress_tokens)
    else:
        token = torch.argmax(last_logits, dim=-1)

    # Extract ref_code for decoder context (if in ICL mode)
    ref_code_context: Optional[torch.Tensor] = None
    ref_code_frames: int = 0
    if voice_clone_prompt is not None:
        ref_code_list = voice_clone_prompt.get("ref_code", None)
        icl_mode_list = voice_clone_prompt.get("icl_mode", None)
        if ref_code_list is not None and icl_mode_list is not None:
            if ref_code_list[0] is not None and icl_mode_list[0]:
                ref_code_context = ref_code_list[0].to(self.talker.device)
                ref_code_frames = ref_code_context.shape[0]

    # Decode loop
    codes_buffer: list[torch.Tensor] = []
    decoded_tail: Optional[np.ndarray] = None
    frames_since_emit = 0
    total_frames_emitted = 0

    for step_idx in range(max_frames):
        torch.compiler.cudagraph_mark_step_begin()

        # Single-step forward
        step_out = self.talker.forward(
            input_ids=token.unsqueeze(1),
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,
            past_key_values=past_key_values,
            past_hidden=past_hidden,
            generation_step=generation_step,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )

        # Update state for next iteration
        past_key_values = step_out.past_key_values
        past_hidden = step_out.past_hidden
        generation_step = step_out.generation_step

        # Get codec_ids from hidden_states tuple: (layer_outputs, codec_ids)
        codec_ids = step_out.hidden_states[1]  # [B, num_code_groups]

        # Check for EOS in first codebook
        # Check against all EOS tokens
        if codec_ids[0, 0].item() in eos_ids:
            break

        # Keep on GPU to avoid CPU<->GPU transfers during decode
        codes_buffer.append(codec_ids[0].detach())

        # Sample next token for first codebook
        step_logits = step_out.logits[:, -1, :]
        if do_sample:
            token = _sample_next_token(step_logits, temperature, top_k, top_p, suppress_tokens)
        else:
            token = torch.argmax(step_logits, dim=-1)

        frames_since_emit += 1

        # Two-phase streaming: determine current phase settings
        total_frames_generated = len(codes_buffer)
        if first_chunk_emit_every > 0 and total_frames_generated < first_chunk_frames:
            # Phase 1: Aggressive settings for first chunk (lower latency)
            current_emit_every = first_chunk_emit_every
            current_decode_window = first_chunk_decode_window
            current_use_optimized = False  # Non-optimized allows flexible window size
        else:
            # Phase 2: Stable settings (better quality)
            current_emit_every = emit_every_frames
            current_decode_window = decode_window_frames
            current_use_optimized = use_optimized_decode

        if frames_since_emit < current_emit_every:
            continue
        frames_since_emit = 0

        # Decode window of codec frames to PCM
        start = max(0, len(codes_buffer) - current_decode_window)
        window_codes = torch.stack(codes_buffer[start:], dim=0)  # [T, num_code_groups]

        # Add ref_code as context prefix for stable decoder context from the start
        window, _ = _add_ref_code_context(
            window_codes, ref_code_context, ref_code_frames, current_decode_window
        )

        # Use optimized decode path when available
        # Pass pad_to_size to ensure fixed tensor size for torch.compile
        if current_use_optimized and hasattr(self.speech_tokenizer, 'decode_streaming'):
            wavs, sr = self.speech_tokenizer.decode_streaming(
                window.to(self.talker.device),
                use_optimized=True,
                pad_to_size=current_decode_window,
            )
        else:
            wavs, sr = self.speech_tokenizer.decode([{"audio_codes": window.to(self.talker.device)}])

        wav = wavs[0].astype(np.float32)

        # Extract only new samples (tail of decoded window)
        # Use fixed upsample rate to avoid floating-point drift
        samples_per_frame = self.speech_tokenizer.get_decode_upsample_rate()
        step_samples = samples_per_frame * current_emit_every
        chunk = wav[-step_samples:] if step_samples > 0 else wav

        # Always blend boundaries to prevent clicks from sliding window re-decode artifacts
        blend_samples = overlap_samples
        if decoded_tail is not None:
            ov = min(blend_samples, len(decoded_tail), len(chunk))
            if ov > 0:
                head = _crossfade(decoded_tail[-ov:], chunk[:ov])
                chunk = np.concatenate([head, chunk[ov:]], axis=0)

        # Apply Hann fade-in to very first chunk to avoid pop at audio start
        if decoded_tail is None:
            fade_len = min(blend_samples, len(chunk))
            if fade_len > 0:
                t = np.arange(fade_len, dtype=np.float32) / max(fade_len - 1, 1)
                fade_in = 0.5 * (1 - np.cos(np.pi * t))
                chunk[:fade_len] *= fade_in

        # Save FULL chunk for next crossfade reference
        decoded_tail = chunk.copy()

        # Trim END of chunk - this region will be replaced by next chunk's crossfade
        # Don't trim if chunk would become too small
        if len(chunk) > blend_samples * 2:
            chunk = chunk[:-blend_samples]

        total_frames_emitted = len(codes_buffer)  # Mark these frames as emitted
        yield chunk, sr

    # Flush: decode only remaining frames that haven't been emitted yet
    remaining_frames = len(codes_buffer) - total_frames_emitted
    if remaining_frames > 0:
        # Decode a window that includes some context for quality
        context_frames = min(total_frames_emitted, decode_window_frames - remaining_frames)
        start_idx = total_frames_emitted - context_frames
        window_codes = torch.stack(codes_buffer[start_idx:], dim=0)

        # Add ref_code as context prefix for stable decoder context
        window, flush_ref_prefix_frames = _add_ref_code_context(
            window_codes, ref_code_context, ref_code_frames, decode_window_frames
        )

        wavs, sr = self.speech_tokenizer.decode([{"audio_codes": window.to(self.talker.device)}])
        wav = wavs[0].astype(np.float32)

        # Extract only the new samples (skip ref_code and context portions)
        skip_frames = flush_ref_prefix_frames + context_frames
        if skip_frames > 0:
            samples_per_frame = len(wav) / window.shape[0]
            skip_samples = int(skip_frames * samples_per_frame)
            wav = wav[skip_samples:]

        # Always blend flush boundary
        blend_samples = overlap_samples
        if decoded_tail is not None and len(wav) > 0:
            ov = min(blend_samples, len(decoded_tail), len(wav))
            if ov > 0:
                head = _crossfade(decoded_tail[-ov:], wav[:ov])
                wav = np.concatenate([head, wav[ov:]], axis=0)

        # Apply fade-out at very end of audio to avoid pop on completion
        if len(wav) > blend_samples:
            fade_len = min(blend_samples, len(wav))
            t = np.arange(fade_len, dtype=np.float32) / max(fade_len - 1, 1)
            fade_out = 0.5 * (1 + np.cos(np.pi * t))  # Hann fade-out
            wav[-fade_len:] *= fade_out

        yield wav, sr


# ========================================
# enable_streaming_optimizations 实现
# ========================================

def _enable_streaming_optimizations_impl(
    self,
    decode_window_frames: int = 80,
    use_compile: bool = True,
    use_cuda_graphs: bool = False,
    compile_mode: str = "reduce-overhead",
    use_fast_codebook: bool = False,
    compile_codebook_predictor: bool = True,
):
    """
    Enable torch.compile and CUDA graphs optimizations for streaming decode.

    基于 dffdeeq/Qwen3-TTS-streaming 项目的实现
    """
    if hasattr(self.speech_tokenizer, 'model'):
        self.speech_tokenizer.model.enable_streaming_optimizations(
            decode_window_frames=decode_window_frames,
            use_compile=use_compile,
            use_cuda_graphs=use_cuda_graphs,
            compile_mode=compile_mode,
        )

    return self


# ========================================
# 导出接口
# ========================================

__all__ = [
    'apply_streaming_patch_to_qwen_tts',
    'DEFAULT_BLEND_SAMPLES',
]
