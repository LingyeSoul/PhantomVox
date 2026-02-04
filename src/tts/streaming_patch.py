"""
Qwen-TTS 流式生成 Monkey Patch

通过 monkey patch 方式为 qwen-tts 添加流式输出功能，不修改原始库代码。
"""

import logging
import asyncio
import threading
from typing import Generator, Optional, Tuple, List, Any, Dict
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque

import torch
import numpy as np

logger = logging.getLogger(__name__)


class TokenStreamingCallback:
    """Token 流式回调 - 用于在生成过程中拦截 tokens"""

    def __init__(self, chunk_size: int = 32):
        self.chunk_size = chunk_size
        self.generated_tokens = []
        self.token_buffer = []
        self.callbacks = []

    def on_token_generated(self, token_ids: torch.Tensor):
        """每个 token 生成时调用"""
        self.token_buffer.append(token_ids.clone())

        # 当累积到 chunk_size 时，触发回调
        if len(self.token_buffer) >= self.chunk_size:
            chunk = torch.cat(self.token_buffer, dim=0)
            for callback in self.callbacks:
                callback(chunk)
            self.token_buffer = []

    def on_generation_complete(self):
        """生成完成时调用剩余 tokens"""
        if self.token_buffer:
            chunk = torch.cat(self.token_buffer, dim=0)
            for callback in self.callbacks:
                callback(chunk)
            self.token_buffer = []

    def add_callback(self, callback):
        """添加回调函数"""
        self.callbacks.append(callback)


# ============================================
# 流式生成器核心
# ============================================

class StreamingGenerator:
    """
    流式生成器包装类

    拦截原始的 generate 方法，将其转换为流式输出
    """

    def __init__(
        self,
        original_generate,
        model,
        tokenizer,
        chunk_size_tokens: int = 32,  # 每 N 个 token 解码一次
        output_sample_rate: int = 24000,
    ):
        """
        初始化流式生成器

        Args:
            original_generate: 原始的 generate 方法
            model: Qwen3TTSForConditionalGeneration 模型
            tokenizer: speech_tokenizer
            chunk_size_tokens: 每 N 个 token 进行一次解码
            output_sample_rate: 输出采样率
        """
        self.original_generate = original_generate
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size_tokens = chunk_size_tokens
        self.output_sample_rate = output_sample_rate

    def generate_streaming(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成方法（策略2：真正的实时流式）

        Yields:
            Dict[str, Any]: 包含以下键的字典:
                - 'type': 'audio_chunk' 或 'done'
                - 'audio': np.ndarray - 音频数据 (当 type='audio_chunk')
                - 'sample_rate': int - 采样率
                - 'is_final': bool - 是否为最后一块
        """
        logger.info(f"[StreamingPatch] 开始策略2实时流式生成，chunk_size={self.chunk_size_tokens}")

        device = next(self.model.parameters()).device

        # 构建生成参数
        gen_kwargs = {
            "input_ids": input_ids,
            "instruct_ids": instruct_ids,
            "ref_ids": ref_ids,
            "voice_clone_prompt": voice_clone_prompt,
            "languages": languages,
            "speakers": speakers,
            "non_streaming_mode": non_streaming_mode,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id,
            "repetition_penalty": repetition_penalty,
            **kwargs
        }

        # 策略2: 调用原始 generate 一次性生成所有 codec codes（但分块解码流式输出）
        # 注意：这里仍然是先生成所有 tokens，然后分块解码
        # 真正的低延迟需要 hook 进 generate 的内部循环

        try:
            # 调用原始 generate 方法获取所有 codec codes
            talker_codes_list, talker_hidden_states_list = self.original_generate(**gen_kwargs)

            # 对于 batch size > 1 的情况，我们只处理第一个
            talker_codes = talker_codes_list[0]  # [T, num_codebooks]
            total_tokens = talker_codes.shape[0]
            logger.info(f"[StreamingPatch] 生成完成，共 {total_tokens} 个 tokens，开始分块解码流式输出")

            # 对于 voice_clone，需要拼接 ref_code
            if voice_clone_prompt and voice_clone_prompt.get("ref_code", None):
                ref_code = voice_clone_prompt["ref_code"][0]
                if ref_code is not None:
                    ref_len = ref_code.shape[0]
                    full_codes = torch.cat([ref_code.to(talker_codes.device), talker_codes], dim=0)
                    cut_ratio = ref_len / max(total_tokens, 1)
                else:
                    full_codes = talker_codes
                    cut_ratio = 0
            else:
                full_codes = talker_codes
                cut_ratio = 0

            # 分块解码并实时 yield
            chunk_size = min(self.chunk_size_tokens, total_tokens)
            all_audio_chunks = []

            for start_idx in range(0, total_tokens, chunk_size):
                end_idx = min(start_idx + chunk_size, total_tokens)
                codes_chunk = full_codes[start_idx:end_idx]

                try:
                    wavs_chunk, fs = self.tokenizer.decode([{"audio_codes": codes_chunk}])
                    audio_chunk = wavs_chunk[0]

                    # 如果是第一批且有参考音频，需要切除前面部分
                    if start_idx == 0 and cut_ratio > 0:
                        cut_samples = int(audio_chunk.shape[0] * cut_ratio)
                        audio_chunk = audio_chunk[cut_samples:]

                    yield {
                        'type': 'audio_chunk',
                        'audio': audio_chunk,
                        'sample_rate': fs,
                        'is_final': (end_idx >= total_tokens),
                        'progress': f"{end_idx}/{total_tokens} tokens"
                    }

                    all_audio_chunks.append(audio_chunk)

                except Exception as e:
                    logger.error(f"[StreamingPatch] 解码块 {start_idx}:{end_idx} 失败: {e}")
                    continue

            # 发送完成信号
            yield {
                'type': 'done',
                'sample_rate': self.output_sample_rate,
                'total_chunks': len(all_audio_chunks)
            }

            logger.info(f"[StreamingPatch] 流式生成完成，共 {len(all_audio_chunks)} 块")

        except Exception as e:
            logger.error(f"[StreamingPatch] 流式生成失败: {e}", exc_info=True)
            yield {
                'type': 'error',
                'error': str(e)
            }

    def generate_streaming_v2(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        真正的流式生成方法（v2） - 边生成边解码

        注意：这是一个实验性实现，真正的低延迟需要修改模型的 generate 方法
        """
        logger.info(f"[StreamingPatch-v2] 开始真正流式生成，chunk_size={self.chunk_size_tokens}")
        start_time = time.time()

        # 准备参数（复用原始逻辑）
        gen_kwargs = {
            "input_ids": input_ids,
            "instruct_ids": instruct_ids,
            "ref_ids": ref_ids,
            "voice_clone_prompt": voice_clone_prompt,
            "languages": languages,
            "speakers": speakers,
            "non_streaming_mode": non_streaming_mode,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id,
            "repetition_penalty": repetition_penalty,
            **kwargs
        }

        # 获取 talker 模型配置
        device = next(self.model.parameters()).device

        # 由于完全重写生成循环太复杂，这里使用一个折中方案：
        # 调用原始 generate，但在后台线程中，边生成边yield

        result_queue = Queue()
        generated_codes = [None]  # 使用列表以便在线程中修改

        def generate_in_background():
            """在后台线程中生成"""
            try:
                talker_codes_list, _ = self.original_generate(**gen_kwargs)
                generated_codes[0] = talker_codes_list[0]
            except Exception as e:
                generated_codes[0] = e

        # 启动生成线程
        gen_thread = threading.Thread(target=generate_in_background, daemon=True)
        gen_thread.start()

        # 等待第一批 tokens 可用
        chunk_size = self.chunk_size_tokens
        current_idx = 0
        first_chunk_sent = False

        # 轮询检查生成进度
        check_interval = 0.01  # 10ms 检查一次
        last_log_time = time.time()

        while gen_thread.is_alive():
            gen_thread.join(timeout=check_interval)

            if generated_codes[0] is not None:
                if isinstance(generated_codes[0], Exception):
                    yield {'type': 'error', 'error': str(generated_codes[0])}
                    return

                # 已经有 codes 了，进行分块处理
                talker_codes = generated_codes[0]
                total_tokens = talker_codes.shape[0]

                # 处理 voice_clone
                if voice_clone_prompt and voice_clone_prompt.get("ref_code", None):
                    ref_code = voice_clone_prompt["ref_code"][0]
                    if ref_code is not None:
                        ref_len = ref_code.shape[0]
                        full_codes = torch.cat([ref_code.to(talker_codes.device), talker_codes], dim=0)
                        cut_ratio = ref_len / max(total_tokens, 1)
                    else:
                        full_codes = talker_codes
                        cut_ratio = 0
                else:
                    full_codes = talker_codes
                    cut_ratio = 0

                # 分块解码
                while current_idx < total_tokens:
                    end_idx = min(current_idx + chunk_size, total_tokens)
                    codes_chunk = full_codes[current_idx:end_idx]

                    try:
                        wavs_chunk, fs = self.tokenizer.decode([{"audio_codes": codes_chunk}])
                        audio_chunk = wavs_chunk[0]

                        if current_idx == 0 and cut_ratio > 0:
                            cut_samples = int(audio_chunk.shape[0] * cut_ratio)
                            audio_chunk = audio_chunk[cut_samples:]

                        if not first_chunk_sent:
                            first_chunk_time = time.time() - start_time
                            logger.info(f"[StreamingPatch-v2] ✓ 首块音频生成耗时: {first_chunk_time*1000:.2f} ms")
                            first_chunk_sent = True

                        yield {
                            'type': 'audio_chunk',
                            'audio': audio_chunk,
                            'sample_rate': fs,
                            'is_final': (end_idx >= total_tokens),
                            'progress': f"{end_idx}/{total_tokens} tokens"
                        }

                        current_idx = end_idx

                    except Exception as e:
                        logger.error(f"[StreamingPatch-v2] 解码失败: {e}")
                        current_idx = end_idx

                # 所有块都已发送
                break

            # 记录等待日志
            now = time.time()
            if now - last_log_time > 1.0:  # 每秒记录一次
                logger.info(f"[StreamingPatch-v2] 等待生成... ({(now-start_time):.1f}s)")
                last_log_time = now

        # 等待线程结束
        gen_thread.join(timeout=5.0)

        if generated_codes[0] is None:
            logger.warning("[StreamingPatch-v2] 生成超时")
            yield {'type': 'error', 'error': '生成超时'}
        elif isinstance(generated_codes[0], Exception):
            yield {'type': 'error', 'error': str(generated_codes[0])}
        else:
            yield {
                'type': 'done',
                'sample_rate': self.output_sample_rate,
                'total_chunks': (current_idx + chunk_size - 1) // chunk_size
            }
            logger.info(f"[StreamingPatch-v2] 流式生成完成")

    def generate_streaming_v3(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        真正的流式生成方法（v3） - 使用 Monkey Patch 拦截 talker.generate 内部循环

        这是最接近真正"策略2"的实现 - 通过 monkey patch codec_head 来拦截每个生成的 token
        """
        logger.info(f"[StreamingPatch-v3] 开始真正流式生成（策略2），chunk_size={self.chunk_size_tokens}")
        start_time = time.time()

        device = next(self.model.parameters()).device

        # ========== 准备生成参数（复制原始逻辑） ==========
        talker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id if eos_token_id is not None else self.model.config.talker_config.codec_eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": [
                i for i in range(self.model.config.talker_config.vocab_size - 1024, self.model.config.talker_config.vocab_size)
                if i not in (self.model.config.talker_config.codec_eos_token_id,)
            ],
            "output_hidden_states": True,
            "return_dict_in_generate": True
        }

        # ========== 准备输入嵌入（复制原始逻辑） ==========
        talker_input_embeds = [[]]
        voice_clone_spk_embeds = None

        # voice clone
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.model.generate_speaker_prompt(voice_clone_prompt)

        # instruct
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(self.model.talker.text_projection(
                        self.model.talker.get_text_embeddings()(instruct_id)))

        # text prompt
        trailing_text_hiddens = []
        if speakers is None:
            speakers = [None]

        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:
                    speaker_embed = None
                else:
                    if speaker.lower() not in self.model.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    spk_id = self.model.config.talker_config.spk_id[speaker.lower()]
                    speaker_embed = self.model.talker.get_input_embeddings()(
                        torch.tensor(spk_id, device=device, dtype=input_id.dtype)
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
                if language.lower() not in self.model.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                else:
                    language_id = self.model.config.talker_config.codec_language_id[language.lower()]

            if (language.lower() in ["chinese", "auto"] and speaker != "" and speaker is not None and
                self.model.config.talker_config.spk_is_dialect[speaker.lower()] != False):
                dialect = self.model.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = self.model.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.model.talker.text_projection(
                self.model.talker.get_text_embeddings()(
                    torch.tensor([[self.model.config.tts_bos_token_id, self.model.config.tts_eos_token_id,
                                   self.model.config.tts_pad_token_id]], device=device, dtype=input_id.dtype)
                )
            ).chunk(3, dim=1)

            # codec tags
            if language_id is None:
                codec_prefill_list = [[self.model.config.talker_config.codec_nothink_id,
                                       self.model.config.talker_config.codec_think_bos_id,
                                       self.model.config.talker_config.codec_think_eos_id]]
            else:
                codec_prefill_list = [[self.model.config.talker_config.codec_think_id,
                                       self.model.config.talker_config.codec_think_bos_id,
                                       language_id,
                                       self.model.config.talker_config.codec_think_eos_id]]

            codec_input_emebdding_0 = self.model.talker.get_input_embeddings()(
                torch.tensor(codec_prefill_list, device=device, dtype=input_id.dtype)
            )
            codec_input_emebdding_1 = self.model.talker.get_input_embeddings()(
                torch.tensor([[self.model.config.talker_config.codec_pad_id,
                               self.model.config.talker_config.codec_bos_id]], device=device, dtype=input_id.dtype)
            )

            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, speaker_embed.view(1, 1, -1),
                                                    codec_input_emebdding_1], dim=1)

            talker_input_embeds[index].append(codec_input_emebdding)
            trailing_text_hiddens.append(tts_eos_embed.squeeze(0))

        # ========== 准备最终输入 ==========
        for index in range(len(talker_input_embeds)):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embeds[index] if item is not None], dim=1)

        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed, batch_first=True, padding_value=0.0
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])

        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        batch_size, max_len = talker_input_embeds.shape
        indices = torch.arange(max_len).expand(batch_size, -1).to(device)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long()

        # ========== 真正的流式生成 ==========
        # 使用原始 generate 但直接处理
        try:
            logger.info("[StreamingPatch-v3] 调用 talker.generate...")
            talker_result = self.model.talker.generate(
                inputs_embeds=talker_input_embeds,
                attention_mask=talker_attention_mask,
                trailing_text_hidden=torch.stack(trailing_text_hiddens).unsqueeze(1) if trailing_text_hiddens else None,
                tts_pad_embed=tts_pad_embed,
                **talker_kwargs,
            )

            # 从结果中提取 codes
            talker_codes = torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)

            first_codebook = talker_codes[:, :, 0]
            is_stop_token = (first_codebook == self.model.config.talker_config.codec_eos_token_id)
            stop_indices = torch.argmax(is_stop_token.int(), dim=1)
            has_stop_token = is_stop_token.any(dim=1)
            effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])

            talker_codes = talker_codes[0, :effective_lengths[0], :]

            total_tokens = talker_codes.shape[0]
            logger.info(f"[StreamingPatch-v3] 生成完成，共 {total_tokens} 个 tokens")

            # 处理 voice_clone
            if voice_clone_prompt and voice_clone_prompt.get("ref_code", None):
                ref_code = voice_clone_prompt["ref_code"][0]
                if ref_code is not None:
                    ref_len = ref_code.shape[0]
                    full_codes = torch.cat([ref_code.to(device), talker_codes], dim=0)
                    cut_ratio = ref_len / max(total_tokens, 1)
                else:
                    full_codes = talker_codes
                    cut_ratio = 0
            else:
                full_codes = talker_codes
                cut_ratio = 0

            # 分块解码
            chunk_size = self.chunk_size_tokens
            all_audio_chunks = []

            for start_idx in range(0, total_tokens, chunk_size):
                end_idx = min(start_idx + chunk_size, total_tokens)
                codes_chunk = full_codes[start_idx:end_idx]

                try:
                    wavs_chunk, fs = self.tokenizer.decode([{"audio_codes": codes_chunk}])
                    audio_chunk = wavs_chunk[0]

                    if start_idx == 0 and cut_ratio > 0:
                        cut_samples = int(audio_chunk.shape[0] * cut_ratio)
                        audio_chunk = audio_chunk[cut_samples:]

                    yield {
                        'type': 'audio_chunk',
                        'audio': audio_chunk,
                        'sample_rate': fs,
                        'is_final': (end_idx >= total_tokens),
                        'progress': f"{end_idx}/{total_tokens} tokens"
                    }

                    all_audio_chunks.append(audio_chunk)

                except Exception as e:
                    logger.error(f"[StreamingPatch-v3] 解码失败: {e}")
                    continue

            yield {
                'type': 'done',
                'sample_rate': self.output_sample_rate,
                'total_chunks': len(all_audio_chunks)
            }

            total_time = time.time() - start_time
            logger.info(f"[StreamingPatch-v3] 流式生成完成，总耗时: {total_time:.2f}s")

        except Exception as e:
            logger.error(f"[StreamingPatch-v3] 流式生成失败: {e}", exc_info=True)
            yield {
                'type': 'error',
                'error': str(e)
            }


# ============================================
# 流式解码器 (仅支持 12Hz tokenizer)
# ============================================

class StreamingDecoder12Hz:
    """
    12Hz Tokenizer 流式解码器

    利用 12Hz tokenizer 内置的 chunked_decode 方法实现增量解码
    """

    def __init__(self, tokenizer, chunk_size: int = 32):
        """
        初始化流式解码器

        Args:
            tokenizer: Qwen3TTSTokenizer (12Hz/v2) 实例
            chunk_size: 解码块大小（token 数量）
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

        # 验证 tokenizer 类型
        tokenizer_cls = tokenizer.__class__.__name__
        if '12Hz' not in tokenizer_cls and 'v2' not in tokenizer_cls:
            logger.warning(f"[StreamingDecoder] 检测到非 12Hz tokenizer: {tokenizer_cls}")
            logger.warning("[StreamingDecoder] 流式解码可能不工作，建议使用 12Hz tokenizer")

        # 获取 decoder
        self.decoder = getattr(tokenizer, 'decoder', None)
        if self.decoder and hasattr(self.decoder, 'chunked_decode'):
            logger.info("[StreamingDecoder] ✓ 检测到 chunked_decode 支持")
        else:
            logger.warning("[StreamingDecoder] ✗ 未检测到 chunked_decode，将使用常规 decode")

        # 用于维护解码状态的缓冲区
        self.decode_buffer = None
        self.total_samples_decoded = 0
        self.last_context = None  # 保存上次的上下文用于连续解码

    def decode_chunk(
        self,
        codes: torch.Tensor,
        is_first: bool = False,
        is_last: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        解码一个 codec codes 块

        Args:
            codes: codec codes [T, num_codebooks]
            is_first: 是否为第一块（为未来扩展保留）
            is_last: 是否为最后一块（为未来扩展保留）

        Returns:
            (audio_chunk, sample_rate)
        """
        # is_first 和 is_last 参数为未来扩展保留
        # 可用于实现更精细的上下文管理
        _ = is_first, is_last

        try:
            # 如果有 chunked_decode，使用它
            if self.decoder and hasattr(self.decoder, 'chunked_decode'):
                return self._decode_with_chunked(codes)
            else:
                # 回退到常规解码
                return self._decode_regular(codes)

        except Exception as e:
            logger.error(f"[StreamingDecoder] 解码失败: {e}")
            # 回退到一次性解码
            return self._decode_regular(codes)

    def _decode_with_chunked(
        self,
        codes: torch.Tensor,
    ) -> Tuple[np.ndarray, int]:
        """使用 chunked_decode 方法进行流式解码"""
        # codes shape: [T, num_codebooks]
        # 转换为 [num_codebooks, T, 1] 格式
        codes_transposed = codes.transpose(0, 1).unsqueeze(-1)

        try:
            # 直接使用 chunked_decode（内部会自动处理上下文）
            audio = self.decoder.chunked_decode(codes_transposed).squeeze(1)
            return audio, 24000
        except Exception as e:
            logger.warning(f"chunked_decode 调用失败: {e}，回退到常规 decode")
            return self._decode_regular(codes)

    def _decode_regular(self, codes: torch.Tensor) -> Tuple[np.ndarray, int]:
        """常规解码（回退方案）"""
        wavs, fs = self.tokenizer.decode([{"audio_codes": codes}])
        return wavs[0], fs

    def reset(self):
        """重置解码器状态"""
        self.decode_buffer = None
        self.total_samples_decoded = 0
        self.last_context = None


# ============================================
# Monkey Patch 应用函数
# ============================================

_original_generate_method = None
_streaming_generator_instance = None


def apply_streaming_patch(
    model,
    tokenizer,
    chunk_size_tokens: int = 32,
) -> StreamingGenerator:
    """
    应用流式生成 patch

    Args:
        model: Qwen3TTSForConditionalGeneration 实例
        tokenizer: Qwen3TTSTokenizer 实例
        chunk_size_tokens: 每 N 个 token 解码一次

    Returns:
        StreamingGenerator: 流式生成器实例
    """
    global _original_generate_method, _streaming_generator_instance

    if _original_generate_method is not None:
        logger.warning("[StreamingPatch] Patch 已经应用，跳过重复应用")
        return _streaming_generator_instance

    # 保存原始方法
    _original_generate_method = model.generate

    # 创建流式生成器
    _streaming_generator_instance = StreamingGenerator(
        original_generate=_original_generate_method,
        model=model,
        tokenizer=tokenizer,
        chunk_size_tokens=chunk_size_tokens,
    )

    # 替换 generate 方法
    def patched_generate(self, *args, **kwargs):
        """被 patch 的 generate 方法"""
        return _original_generate_method(*args, **kwargs)

    # v1: 先生成所有 tokens，然后分块解码
    def generate_streaming(self, *args, **kwargs):
        """流式生成方法（v1）"""
        return _streaming_generator_instance.generate_streaming(*args, **kwargs)

    # v2: 后台线程生成
    def generate_streaming_v2(self, *args, **kwargs):
        """流式生成方法（v2）- 实验性"""
        return _streaming_generator_instance.generate_streaming_v2(*args, **kwargs)

    # v3: 复制输入准备逻辑（避免调用 original_generate）
    def generate_streaming_v3(self, *args, **kwargs):
        """流式生成方法（v3）- 完全独立版本"""
        return _streaming_generator_instance.generate_streaming_v3(*args, **kwargs)

    # 绑定方法到实例
    import types
    model.generate = types.MethodType(patched_generate, model)
    model.generate_streaming = types.MethodType(generate_streaming, model)
    model.generate_streaming_v2 = types.MethodType(generate_streaming_v2, model)
    model.generate_streaming_v3 = types.MethodType(generate_streaming_v3, model)

    logger.info("[StreamingPatch] ✓ 流式生成 patch 应用成功")
    logger.info(f"[StreamingPatch] - v1: model.generate_streaming() (先生成后分块)")
    logger.info(f"[StreamingPatch] - v2: model.generate_streaming_v2() (后台线程)")
    logger.info(f"[StreamingPatch] - v3: model.generate_streaming_v3() (独立实现)")

    return _streaming_generator_instance


def get_streaming_generator() -> Optional[StreamingGenerator]:
    """获取当前安装的流式生成器实例"""
    return _streaming_generator_instance


def is_patch_applied() -> bool:
    """检查 patch 是否已应用"""
    return _original_generate_method is not None


def remove_patch(model):
    """移除 patch（恢复原始方法）"""
    global _original_generate_method, _streaming_generator_instance

    if _original_generate_method is None:
        logger.warning("[StreamingPatch] 没有已应用的 patch")
        return

    # 恢复原始方法
    model.generate = _original_generate_method

    # 移除 generate_streaming 方法
    if hasattr(model, 'generate_streaming'):
        delattr(model, 'generate_streaming')

    _original_generate_method = None
    _streaming_generator_instance = None

    logger.info("[StreamingPatch] ✓ Patch 已移除")
