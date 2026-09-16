"""WeSpeaker ONNX inference with its reference Kaldi feature settings.

Feature recipe adapted from wespeaker/bin/infer_onnx.py:
Copyright (c) 2022, Shuai Wang (wsstriving@gmail.com).
Licensed under the Apache License, Version 2.0:
http://www.apache.org/licenses/LICENSE-2.0
Distributed on an AS IS basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
Source: https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/infer_onnx.py
"""
import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi


class SpeakerEncoder:
    """Input: mono float samples in [-1, 1] at 16 kHz, after speech detection.

This encoder returns an embedding, not an authorization decision. Owner
calibration and handling mixed/uncertain speakers belong to the next stage.
"""
    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])

    def encode(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or len(samples) < 16000:
            raise ValueError("Нужна как минимум секунда монофонической речи при 16 кГц")
        if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.01:
            raise ValueError("Некорректные значения аудиосэмплов")
        if np.sqrt(np.mean(samples * samples)) < 1e-5:
            raise ValueError("Тишина не подходит для проверки голоса")
        waveform = torch.from_numpy(samples.copy()).unsqueeze(0) * (1 << 15)
        with torch.inference_mode():
            features = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25,
                                   frame_shift=10, dither=0.0, sample_frequency=16000,
                                   window_type="hamming", use_energy=False)
            features -= torch.mean(features, dim=0)
        vector = self.session.run(["embs"], {"feats": features.unsqueeze(0).numpy()})[0].reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm < 1e-8:
            raise ValueError("Модель вернула некорректный голосовой вектор")
        return vector / norm
