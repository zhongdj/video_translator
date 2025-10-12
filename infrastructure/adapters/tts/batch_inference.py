"""
infrastructure/adapters/tts/batch_inference.py

IndexTTS2 批处理推理模块
将这些函数注入到 IndexTTS2 实例中以支持批处理
"""

import torch
import torchaudio
import time
import os
from pathlib import Path
from typing import List, Optional


def batch_infer_same_speaker2(
        self,
        texts: List[str],
        spk_audio_prompt: str,
        output_paths: Optional[List[str]] = None,
        emo_audio_prompt: Optional[str] = None,
        emo_alpha: float = 1.0,
        interval_silence: int = 200,
        verbose: bool = False,
        max_text_tokens_per_segment: int = 120,
        **generation_kwargs
) -> List[Optional[str]]:
    """
    批量推理相同说话人的多个文本

    Args:
        texts: 文本列表
        spk_audio_prompt: 说话人音频路径
        output_paths: 输出路径列表 (None 则返回音频数组)
        emo_audio_prompt: 情感音频路径
        emo_alpha: 情感混合系数
        interval_silence: 段落间静音时长(ms)
        verbose: 详细输出
        max_text_tokens_per_segment: 每段最大token数
        **generation_kwargs: 其他生成参数

    Returns:
        输出路径列表或 (sampling_rate, wav_data) 元组列表
    """

    print(f">> 批量推理: {len(texts)} 个文本，相同说话人")
    start_time = time.perf_counter()

    batch_size = len(texts)
    if output_paths is None:
        output_paths = [None] * batch_size
    if emo_audio_prompt is None:
        emo_audio_prompt = spk_audio_prompt
        emo_alpha = 1.0

    # ============================================
    # 步骤1: 准备共享条件 (只计算一次)
    # ============================================
    self._set_gr_progress(0, "准备说话人条件...")

    if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
        _prepare_speaker_condition(self, spk_audio_prompt, verbose)

    if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
        _prepare_emotion_condition(self, emo_audio_prompt, verbose)

    spk_cond_emb = self.cache_spk_cond
    emo_cond_emb = self.cache_emo_cond
    style = self.cache_s2mel_style
    prompt_condition = self.cache_s2mel_prompt
    ref_mel = self.cache_mel

    # ============================================
    # 步骤2: 批量文本分段
    # ============================================
    self._set_gr_progress(0.1, "处理文本...")

    all_segments = []
    for text in texts:
        tokens = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(tokens, max_text_tokens_per_segment)
        all_segments.append(segments)

    max_segments = max(len(s) for s in all_segments)
    if verbose:
        print(f"最大段落数: {max_segments}")

    # ============================================
    # 步骤3: 逐 segment 批量生成
    # ============================================
    all_wavs = [[] for _ in range(batch_size)]

    # 提取生成参数
    do_sample = generation_kwargs.pop("do_sample", True)
    top_p = generation_kwargs.pop("top_p", 0.8)
    top_k = generation_kwargs.pop("top_k", 30)
    temperature = generation_kwargs.pop("temperature", 0.8)
    length_penalty = generation_kwargs.pop("length_penalty", 0.0)
    num_beams = generation_kwargs.pop("num_beams", 3)
    repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
    max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)

    for seg_idx in range(max_segments):
        # 收集当前 segment 的所有文本
        batch_texts = []
        batch_indices = []

        for i, segments in enumerate(all_segments):
            if seg_idx < len(segments):
                batch_texts.append(segments[seg_idx])
                batch_indices.append(i)

        if not batch_texts:
            continue

        # 转换为 token ids 并 padding
        batch_token_ids = [
            self.tokenizer.convert_tokens_to_ids(text)
            for text in batch_texts
        ]
        max_len = max(len(t) for t in batch_token_ids)
        text_tokens_batch = torch.stack([
            torch.nn.functional.pad(
                torch.tensor(t, dtype=torch.int32, device=self.device),
                (0, max_len - len(t)),
                value=0
            )
            for t in batch_token_ids
        ])

        current_batch_size = len(batch_texts)

        if verbose:
            print(f"段落 {seg_idx + 1}/{max_segments}: batch_size={current_batch_size}, max_len={max_len}")

        # ============================================
        # 🔥 关键: 批量调用 GPT
        # ============================================
        with torch.no_grad():
            with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
                # 扩展条件到 batch size
                spk_cond_batch = spk_cond_emb.expand(current_batch_size, -1, -1)
                emo_cond_batch = emo_cond_emb.expand(current_batch_size, -1, -1)

                # 合并情感向量
                emovec = self.gpt.merge_emovec(
                    spk_cond_batch,
                    emo_cond_batch,
                    torch.tensor([spk_cond_emb.shape[-1]] * current_batch_size, device=self.device),
                    torch.tensor([emo_cond_emb.shape[-1]] * current_batch_size, device=self.device),
                    alpha=emo_alpha
                )

                # 批量 GPT 推理
                codes_batch, latent_batch = self.gpt.inference_speech(
                    spk_cond_batch,
                    text_tokens_batch,
                    emo_cond_batch,
                    cond_lengths=torch.tensor([spk_cond_emb.shape[-1]] * current_batch_size, device=self.device),
                    emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]] * current_batch_size, device=self.device),
                    emo_vec=emovec,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=1,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_generate_length=max_mel_tokens,
                    **generation_kwargs
                )

        # ============================================
        # 步骤4: 逐个转换为 wav
        # ============================================
        for i, batch_idx in enumerate(batch_indices):
            wav = _single_codes_to_wav(
                self,
                codes_batch[i:i + 1],
                latent_batch[i:i + 1],
                text_tokens_batch[i:i + 1],
                emovec[i:i + 1],
                style,
                prompt_condition,
                ref_mel
            )
            all_wavs[batch_idx].append(wav.cpu())

        self._set_gr_progress(
            0.2 + 0.7 * (seg_idx + 1) / max_segments,
            f"生成段落 {seg_idx + 1}/{max_segments}..."
        )

    # ============================================
    # 步骤5: 保存结果
    # ============================================
    self._set_gr_progress(0.9, "保存音频...")

    results = []
    sampling_rate = 22050

    for i, wavs in enumerate(all_wavs):
        if not wavs:
            results.append(None)
            continue

        wavs = self.insert_interval_silence(wavs, sampling_rate, interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu()

        if output_paths[i]:
            output_path = Path(output_paths[i])
            if output_path.parent:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), wav.type(torch.int16), sampling_rate)
            results.append(str(output_path))
            if verbose:
                print(f">> 保存到: {output_path}")
        else:
            results.append((sampling_rate, wav.type(torch.int16).numpy().T))

    total_time = time.perf_counter() - start_time
    print(f">> 批量推理完成: {total_time:.2f}秒 (每个文本平均 {total_time / batch_size:.2f}秒)")

    return results


def _prepare_speaker_condition(self, spk_audio_prompt: str, verbose: bool):
    """准备说话人条件"""
    if verbose:
        print(f">> 加载说话人音频: {spk_audio_prompt}")

    audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

    inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(self.device)
    attention_mask = inputs["attention_mask"].to(self.device)

    spk_cond_emb = self.get_emb(input_features, attention_mask)

    _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
    ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
    ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

    feat = torchaudio.compliance.kaldi.fbank(
        audio_16k.to(ref_mel.device),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    style = self.campplus_model(feat.unsqueeze(0))

    prompt_condition = self.s2mel.models['length_regulator'](
        S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
    )[0]

    # 缓存
    self.cache_spk_cond = spk_cond_emb
    self.cache_s2mel_style = style
    self.cache_s2mel_prompt = prompt_condition
    self.cache_spk_audio_prompt = spk_audio_prompt
    self.cache_mel = ref_mel

    if verbose:
        print(f">> 说话人 embedding 准备完成: {spk_cond_emb.shape}")


def _prepare_emotion_condition(self, emo_audio_prompt: str, verbose: bool):
    """准备情感条件"""
    if verbose:
        print(f">> 加载情感音频: {emo_audio_prompt}")

    emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
    emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
    emo_input_features = emo_inputs["input_features"].to(self.device)
    emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
    emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

    # 缓存
    self.cache_emo_cond = emo_cond_emb
    self.cache_emo_audio_prompt = emo_audio_prompt

    if verbose:
        print(f">> 情感 embedding 准备完成: {emo_cond_emb.shape}")


def _single_codes_to_wav(
        self,
        codes: torch.Tensor,
        speech_conditioning_latent: torch.Tensor,
        text_tokens: torch.Tensor,
        emovec: torch.Tensor,
        style: torch.Tensor,
        prompt_condition: torch.Tensor,
        ref_mel: torch.Tensor
) -> torch.Tensor:
    """将单个 codes 转换为音频波形"""

    # 处理 codes 长度
    code_len = codes.shape[-1]
    if self.stop_mel_token in codes[0]:
        stop_idx = (codes[0] == self.stop_mel_token).nonzero(as_tuple=False)
        if len(stop_idx) > 0:
            code_len = stop_idx[0].item()
            codes = codes[:, :code_len]

    code_lens = torch.LongTensor([code_len]).to(self.device)

    # GPT forward pass
    use_speed = torch.zeros(1).to(self.device).long()

    with torch.amp.autocast(self.device, enabled=self.dtype is not None, dtype=self.dtype):
        latent = self.gpt(
            speech_conditioning_latent,
            text_tokens,
            torch.tensor([text_tokens.shape[-1]], device=self.device),
            codes,
            code_lens,
            emovec,
            cond_mel_lengths=torch.tensor([emovec.shape[1]], device=self.device),
            emo_cond_mel_lengths=torch.tensor([emovec.shape[1]], device=self.device),
            emo_vec=emovec,
            use_speed=use_speed,
        )

    # S2Mel 转换
    diffusion_steps = 25
    inference_cfg_rate = 0.7

    latent = self.s2mel.models['gpt_layer'](latent)
    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
    S_infer = S_infer.transpose(1, 2) + latent
    target_lengths = (code_lens * 1.72).long()

    cond = self.s2mel.models['length_regulator'](
        S_infer, ylens=target_lengths, n_quantizers=3, f0=None
    )[0]

    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    vc_target = self.s2mel.models['cfm'].inference(
        cat_condition,
        torch.LongTensor([cat_condition.size(1)]).to(cond.device),
        ref_mel, style, None, diffusion_steps,
        inference_cfg_rate=inference_cfg_rate
    )

    vc_target = vc_target[:, :, ref_mel.size(-1):]

    # Vocoder 生成波形
    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
    wav = wav.squeeze(1)
    wav = torch.clamp(32767 * wav, -32767.0, 32767.0)

    return wav