"""
audio_tools.py —— 音频特性 TXT 导出/还原、DFT 主频提取与幅频响应绘图

1. export_audio_txt        : 原始采样 + 主频特征 -> 可逆 TXT（AI 训练/验证集）
2. restore_txt_to_wav      : TXT -> WAV 逆向还原
3. compute_main_frequencies: 零填充 DFT + 对数抛物线插值（精度优于 0.1 Hz）
4. plot_amplitude_response : 幅频响应绘图（线性 + dB）
5. load_audio_file / load_audio_any : 外界音频加载接口（WAV 始终可用；
   若安装 soundfile 可读 mp3/flac/ogg 等）
"""

import io
import os
import re
import json
import csv
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.io import wavfile
from scipy.fft import dct
from scipy.signal import detrend, find_peaks, medfilt, resample_poly, windows

plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except Exception:
    HAVE_SOUNDFILE = False


# ---------------- 输出目录与命名 ----------------

def build_run_path(outdir, freq, group, timestamp, suffix="", ext=".wav"):
    """在 OUTPUT 根目录下创建 “特征频率--测试组别--记录时间” 子文件夹，
    返回 (文件完整路径, 运行文件夹路径)。"""

    def _clean(s, fallback):
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
        s = re.sub(r"_+", "_", s).strip("._")
        return s or fallback

    freq_s = _clean(freq, "auto")
    try:
        float(freq_s)
        if not freq_s.lower().endswith("hz"):
            freq_s = f"{freq_s}Hz"
    except ValueError:
        pass
    group_s = _clean(group, "A1")
    ts_s = _clean(timestamp, datetime.now().strftime("%Y%m%d_%H%M%S"))
    stem = f"{freq_s}--{group_s}--{ts_s}"
    run_dir = os.path.join(str(outdir), stem)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, stem + suffix + ext), run_dir


# ---------------- 音频读取 ----------------

def load_audio_raw(file_path):
    """读取 WAV，返回 (fs, 原始样本, dtype, 声道数)。"""
    try:
        fs, data = wavfile.read(file_path)
    except Exception as e:
        raise ValueError(f"读取音频文件失败: {file_path}\n错误信息: {e}")
    n_channels = data.shape[1] if data.ndim > 1 else 1
    return fs, data, data.dtype, n_channels


def load_audio(file_path, normalise=False):
    """读取 WAV 并转为单声道 float64；normalise=True 时峰值归一化到 1。"""
    fs, data, dtype, _ = load_audio_raw(file_path)
    if dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    elif dtype == np.uint8:
        audio = (data.astype(np.float64) - 128) / 128.0
    elif dtype in (np.float32, np.float64):
        audio = np.clip(data.astype(np.float64), -1.0, 1.0)
    else:
        raise TypeError(f"不支持的数据类型: {dtype}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if normalise:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    return fs, audio


def load_audio_any(file_path):
    """加载任意音频为原始样本数组，返回 (fs, raw, dtype, n_channels)。
    WAV 保留原始 dtype（可精确还原）；非 WAV 需要 soundfile。"""
    if str(file_path).lower().endswith(".wav"):
        return load_audio_raw(file_path)
    if HAVE_SOUNDFILE:
        data, fs = sf.read(file_path, dtype="float64", always_2d=False)
        n_ch = data.shape[1] if data.ndim > 1 else 1
        return int(fs), data, np.dtype("float64"), n_ch
    raise ValueError(
        "非 WAV 文件需要安装 soundfile: pip install soundfile"
    )


def load_audio_file(file_path, normalise=False):
    """通用音频加载接口：返回 (fs, 单声道 float64)。
    WAV 直接读取；mp3/flac/ogg 等需 soundfile。"""
    if str(file_path).lower().endswith(".wav"):
        return load_audio(file_path, normalise=normalise)
    if HAVE_SOUNDFILE:
        data, fs = sf.read(file_path, dtype="float64", always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if normalise:
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
        return int(fs), data
    raise ValueError(
        "非 WAV 文件需要安装 soundfile: pip install soundfile"
    )


def save_audio_wav(file_path, fs, signal):
    """把 float 信号裁剪到 [-1,1] 后存为 int16 WAV，返回 int16 数组。"""
    signal = np.clip(np.asarray(signal), -1.0, 1.0)
    sig_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(file_path, fs, sig_int16)
    print(f"已保存: {file_path}")
    return sig_int16


# ---------------- 可逆 TXT ----------------

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(fs, n_fft, n_mels, fmin, fmax):
    """构造梅尔三角滤波器组，返回 (n_mels, n_fft//2+1)。"""
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.clip(np.floor((n_fft + 1) * hz_points / fs).astype(int),
                   0, n_fft // 2)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bins[m - 1], bins[m], bins[m + 1]
        if ctr > lo:
            fbank[m - 1, lo:ctr] = np.arange(ctr - lo) / max(ctr - lo, 1)
        if hi > ctr:
            fbank[m - 1, ctr:hi] = 1.0 - np.arange(hi - ctr) / max(hi - ctr, 1)
    return fbank


def compute_log_mel_spectrogram(signal, fs, n_fft=1024, hop=256, n_mels=128,
                                fmin=50.0, fmax=None, top_db=80.0,
                                fixed_time_frames=128, target_sr=None):
    """计算固定尺寸的 log-mel 谱图，返回 (float32 数组, 参数字典)。

    时间轴统一重采样到 fixed_time_frames 帧，保证 CNN 输入尺寸一致；
    幅值取 10*log10(power)，并裁剪到 [-top_db, 0] dB。
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    if target_sr is not None:
        target_sr = int(target_sr)
        if target_sr <= 0:
            raise ValueError("target_sr 必须大于 0")
        if fs != target_sr:
            signal = resample_poly(signal, target_sr, int(fs))
            fs = target_sr
    if fmax is None:
        fmax = fs / 2.0

    win = windows.hann(n_fft, sym=False)
    fbank = _mel_filterbank(fs, n_fft, n_mels, fmin, fmax)
    frames = []
    for start in range(0, len(signal) - n_fft + 1, hop):
        seg = signal[start:start + n_fft] * win
        power = np.abs(np.fft.rfft(seg, n=n_fft)) ** 2
        frames.append(np.dot(fbank, power))

    if frames:
        mel = np.asarray(frames, dtype=np.float64)          # (T, n_mels)
        mel_db = 10.0 * np.log10(mel + 1e-12)
        if mel_db.shape[0] != fixed_time_frames:
            old_t = np.linspace(0.0, 1.0, mel_db.shape[0])
            new_t = np.linspace(0.0, 1.0, fixed_time_frames)
            mel_db = np.array(
                [np.interp(new_t, old_t, mel_db[:, j])
                 for j in range(mel_db.shape[1])]).T
        mel_db = np.maximum(mel_db, -top_db)
    else:
        mel_db = np.full((fixed_time_frames, n_mels), -top_db)

    params = {
        "sample_rate": int(fs),
        "n_fft": int(n_fft),
        "hop": int(hop),
        "n_mels": int(n_mels),
        "fmin": float(fmin),
        "fmax": float(fmax),
        "top_db": float(top_db),
        "fixed_time_frames": int(fixed_time_frames),
        "target_sr": int(target_sr) if target_sr is not None else None,
        "shape": list(mel_db.shape),
    }
    return mel_db.astype(np.float32), params


def save_spectrogram(signal, fs, base_path, n_fft=1024, hop=256, n_mels=128,
                     fmin=50.0, fmax=None, top_db=80.0,
                     fixed_time_frames=128, title=None):
    """保存 log-mel 谱图为 .npy（CNN 输入）、.png（预览）和 .json（参数）。

    base_path 为不带扩展名的输出基底，例如 OUTPUT/<stem>。
    返回 (npy_path, png_path, meta_path, params)。
    """
    mel, params = compute_log_mel_spectrogram(
        signal, fs, n_fft=n_fft, hop=hop, n_mels=n_mels,
        fmin=fmin, fmax=fmax, top_db=top_db,
        fixed_time_frames=fixed_time_frames)
    npy_path = base_path + "_melspectrogram.npy"
    png_path = base_path + "_melspectrogram.png"
    meta_path = base_path + "_melspectrogram_meta.json"
    np.save(npy_path, mel)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    fmax_view = params["fmax"]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, mel.shape[0], fmin, fmax_view])
    ax.set_xlabel("时间帧 / Time frames")
    ax.set_ylabel("频率 (Hz) / Frequency (Hz)")
    ax.set_title(f"{title or os.path.basename(base_path)} — "
                 "log-mel 谱图 / Spectrogram")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"log-mel 谱图已保存: {npy_path}")
    print(f"谱图预览已保存: {png_path}")
    return npy_path, png_path, meta_path, params


def _save_spectrogram_png(mel, params, png_path, title=None):
    """把 log-mel 谱图数组渲染成 PNG 预览。"""
    fmin = float(params["fmin"])
    fmax = float(params["fmax"])
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, mel.shape[0], fmin, fmax])
    ax.set_xlabel("时间帧 / Time frames")
    ax.set_ylabel("频率 (Hz) / Frequency (Hz)")
    ax.set_title(f"{title or os.path.basename(png_path)} — "
                 "log-mel 谱图 / Spectrogram")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def iter_audio_files(input_dir, extensions=(".wav", ".flac", ".ogg", ".mp3")):
    """递归列出目录下的音频文件，返回排序后的绝对路径列表。"""
    exts = {e.lower() if e.startswith(".") else "." + e.lower()
            for e in extensions}
    result = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in exts:
                result.append(os.path.join(root, fn))
    return result


def _label_from_path(path, input_dir):
    """用音频文件所在子目录名作为类别标签。"""
    rel = os.path.relpath(path, input_dir)
    parts = rel.split(os.sep)
    return parts[-2] if len(parts) >= 2 else "unlabeled"


def _safe_filename_token(value, fallback):
    """把用户输入整理成纯英文、可安全用于文件名的 token。"""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or fallback


def batch_generate_spectrograms(input_dir, output_dir, name_prefix,
                                extensions=(".wav", ".flac", ".ogg", ".mp3"),
                                label_from_subdir=True, preview_png=True,
                                split=None, max_files=None,
                                source_tag=None, fixed_label=None,
                                dataset_name=None, file_list=None,
                                n_fft=1024, hop=256, n_mels=128,
                                fmin=50.0, fmax=None, top_db=80.0,
                                fixed_time_frames=128, target_sr=None):
    """批量生成 log-mel 谱图（其余存 .npy，仅第一段生成 PNG 预览）。

    source_tag:  数据来源标签，写入文件名、manifest.csv 与 params.json。
    fixed_label: 固定类别标签；为 None 时按子目录名自动推断。
    file_list:   指定音频文件列表；为 None 时递归扫描 input_dir。
    target_sr:   统一重采样到该采样率；None 表示保留原始采样率。
    """
    if file_list is not None:
        files = [os.path.abspath(str(p)) for p in file_list]
    else:
        files = iter_audio_files(input_dir, extensions)
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    os.makedirs(output_dir, exist_ok=True)

    name_token = _safe_filename_token(name_prefix, "spec")
    source_token = _safe_filename_token(source_tag, "source")
    fixed_label_token = _safe_filename_token(fixed_label, "")
    n = len(files)
    train_set = None
    if split is not None and 0 < split < 1 and n > 1:
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        n_train = int(round(n * split))
        train_set = set(int(i) for i in perm[:n_train])

    manifest = []
    first_params = None
    preview_saved = False
    for idx, path in enumerate(files):
        if fixed_label_token:
            label = fixed_label_token
        else:
            label = (_label_from_path(path, input_dir)
                     if label_from_subdir else "unlabeled")
        label = _safe_filename_token(label, "unlabeled")
        try:
            fs, signal = load_audio_file(path, normalise=False)
        except Exception as e:
            print(f"[跳过/Skip] {path}: {e}")
            continue
        mel, params = compute_log_mel_spectrogram(
            signal, fs, n_fft=n_fft, hop=hop, n_mels=n_mels,
            fmin=fmin, fmax=fmax, top_db=top_db,
            fixed_time_frames=fixed_time_frames, target_sr=target_sr)
        if first_params is None:
            first_params = params
        stem = f"{name_token}__{source_token}__{label}__{idx:06d}"
        npy_path = os.path.join(output_dir, stem + "_melspectrogram.npy")
        np.save(npy_path, mel)
        png_path = ""
        if preview_png and not preview_saved:
            png_path = os.path.join(output_dir, stem + "_melspectrogram.png")
            _save_spectrogram_png(mel, params, png_path, title=stem)
            preview_saved = True
        split_val = ""
        if train_set is not None:
            split_val = "train" if idx in train_set else "val"
        manifest.append({
            "index": idx,
            "source": source_token,
            "dataset_name": _safe_filename_token(dataset_name, name_token),
            "source_file": path,
            "label": label,
            "split": split_val,
            "npy_path": npy_path,
            "png_path": png_path,
        })
        print(f"[{idx + 1}/{n}] {stem} | source={source_token} | "
              f"label={label} | {split_val}")

    csv_path = os.path.join(output_dir, "manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "source", "dataset_name", "source_file",
                         "label", "split", "npy_path", "png_path"])
        for m in manifest:
            writer.writerow([m["index"], m["source"], m["dataset_name"],
                             m["source_file"], m["label"], m["split"],
                             m["npy_path"], m["png_path"]])

    label_counts = {}
    for m in manifest:
        label_counts[m["label"]] = label_counts.get(m["label"], 0) + 1
    dataset_info = {
        "dataset_name": _safe_filename_token(dataset_name, name_token),
        "source": source_token,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "label_mode": ("fixed" if fixed_label_token
                       else ("subdirectory" if label_from_subdir
                             else "unlabeled")),
        "fixed_label": fixed_label_token or None,
        "total_found": n,
        "total_processed": len(manifest),
        "class_distribution": label_counts,
        "spectrogram_params": first_params or {},
    }
    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    return manifest, csv_path, params_path


def _raw_to_float(raw):
    raw = np.asarray(raw)
    dtype = raw.dtype
    if dtype == np.int16:
        return raw.astype(np.float64) / 32768.0
    if dtype == np.int32:
        return raw.astype(np.float64) / 2147483648.0
    if dtype == np.uint8:
        return (raw.astype(np.float64) - 128.0) / 128.0
    if dtype in (np.float32, np.float64):
        return np.clip(raw.astype(np.float64), -1.0, 1.0)
    raise TypeError(f"不支持的数据类型: {dtype}")


def extract_frame_features(signal, fs, frame_ms=25.0, hop_ms=10.0,
                           n_mfcc=13, n_mels=26, f0_frame_ms=40.0):
    """提取短时帧特征，返回 (列名列表, 行列表)。

    每帧特征：frame_index, time_s, rms, zcr, spectral_centroid,
    spectral_rolloff, spectral_flatness, energy, f0_hz, mfcc_1..n。
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    frame_len = int(round(fs * frame_ms / 1000.0))
    hop = int(round(fs * hop_ms / 1000.0))
    nfft = 1
    while nfft < frame_len:
        nfft <<= 1
    win = windows.hann(frame_len, sym=False)

    # 梅尔滤波器组
    mel_points = np.linspace(_hz_to_mel(0.0), _hz_to_mel(fs / 2.0), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points / fs).astype(int)
    fbank = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bins[m - 1], bins[m], bins[m + 1]
        if ctr > lo:
            fbank[m - 1, lo:ctr] = np.arange(ctr - lo) / (ctr - lo)
        if hi > ctr:
            fbank[m - 1, ctr:hi] = 1.0 - np.arange(hi - ctr) / (hi - ctr)

    # 基频曲线（帧中心时间对齐）
    f0_t, f0_v = estimate_f0_curve(signal, fs, f0_min=50.0, f0_max=500.0,
                                   frame_ms=f0_frame_ms, hop_ms=hop_ms)

    cols = (["frame_index", "time_s", "rms", "zcr", "spectral_centroid",
             "spectral_rolloff", "spectral_flatness", "energy", "f0_hz"]
            + [f"mfcc_{i}" for i in range(1, n_mfcc + 1)])
    rows = []
    idx = 0
    for start in range(0, len(signal) - frame_len + 1, hop):
        frame = signal[start:start + frame_len] * win
        t_center = (start + frame_len / 2.0) / fs
        rms = float(np.sqrt(np.mean(frame ** 2)))
        energy = float(np.sum(frame ** 2))
        zcr = (float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
               if len(frame) > 1 else 0.0)

        spec = np.fft.rfft(frame, n=nfft)
        power = np.abs(spec) ** 2
        freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
        total_power = float(power.sum())
        centroid = (float(np.sum(freq * power) / total_power)
                    if total_power > 0 else 0.0)
        rolloff = 0.0
        if total_power > 0:
            cum = np.cumsum(power)
            rolloff = float(freq[int(np.searchsorted(cum, 0.85 * total_power))])
        eps = 1e-12
        flatness = float(np.exp(np.mean(np.log(power + eps))) /
                         (np.mean(power) + eps))

        f0 = float(np.interp(t_center, f0_t, f0_v))
        mel_energy = np.dot(fbank, power)
        mel_energy = np.log(mel_energy + eps)
        mfcc = dct(mel_energy, type=2, norm="ortho")[:n_mfcc]

        row = ([idx, f"{t_center:.6f}", f"{rms:.6g}", f"{zcr:.6g}",
                f"{centroid:.3f}", f"{rolloff:.3f}", f"{flatness:.6g}",
                f"{energy:.6g}", f"{f0:.3f}"]
               + [f"{c:.6g}" for c in mfcc])
        rows.append(row)
        idx += 1
    return cols, rows


def export_audio_txt(txt_path, source_name, fs, raw, main_freqs=None,
                     include_frame_features=True, frame_ms=25.0,
                     hop_ms=10.0, n_mfcc=13):
    """导出可逆 TXT：头部元数据 + 主频特征 + 逐帧特征 + 全部原始采样。"""
    raw = np.asarray(raw)
    n = raw.shape[0]
    n_ch = 1 if raw.ndim == 1 else raw.shape[1]
    dtype = raw.dtype
    lines = [
        "# UAV Audio Characteristics TXT v1",
        f"# created={datetime.now().isoformat(timespec='seconds')}",
        f"source_file={os.path.basename(source_name)}",
        f"sample_rate={fs}",
        f"n_channels={n_ch}",
        f"n_samples={n}",
        f"sample_dtype={dtype.name}",
        f"duration={n / fs:.6f}",
    ]
    if main_freqs is not None and len(main_freqs):
        lines.append(
            "main_frequencies=" + ",".join(f"{f:.6f}" for f in np.asarray(main_freqs)[:, 0])
        )
    lines.append("# --- MAIN_FREQUENCIES ---")
    lines.append("# freq_hz,magnitude,phase_rad")
    if main_freqs is not None:
        for f, a, p in np.asarray(main_freqs):
            lines.append(f"{float(f):.9f},{float(a):.9e},{float(p):.9f}")
    if include_frame_features:
        cols, feat_rows = extract_frame_features(
            _raw_to_float(raw), fs, frame_ms=frame_ms, hop_ms=hop_ms,
            n_mfcc=n_mfcc)
        lines.append("frame_feature_columns=" + ",".join(cols))
        lines.append(f"n_frames={len(feat_rows)}")
        lines.append("# --- FRAME_FEATURES ---")
        lines.append("# " + ",".join(cols))
        for row in feat_rows:
            lines.append(",".join(str(v) for v in row))
    lines.append("# --- SAMPLES ---")
    flat = raw.reshape(n, -1)
    if np.issubdtype(dtype, np.integer):
        rows = (",".join(str(int(v)) for v in row) for row in flat)
    elif dtype == np.float32:
        rows = (",".join(f"{v:.9g}" for v in row) for row in flat)
    else:
        rows = (",".join(f"{v:.17g}" for v in row) for row in flat)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.write("\n".join(rows) + "\n")
    print(f"可逆 TXT 已保存: {txt_path}")


def restore_txt_to_wav(txt_path, wav_path):
    """从可逆 TXT 还原 WAV，返回 (fs, 样本数组, 头部信息)。"""
    header = {}
    sample_lines = []
    section = None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# --- MAIN_FREQUENCIES"):
                    section = "main"
                elif line.startswith("# --- SAMPLES"):
                    section = "samples"
                elif line.startswith("# ---"):
                    section = None
                continue
            if section == "samples":
                sample_lines.append(line)
            elif section != "main" and "=" in line:
                key, value = line.split("=", 1)
                header[key.strip()] = value.strip()
    for key in ("sample_rate", "n_samples", "sample_dtype"):
        if key not in header:
            raise ValueError(f"TXT 头部缺少字段: {key}")
    fs = int(header["sample_rate"])
    n_ch = int(header.get("n_channels", "1"))
    dtype = np.dtype(header["sample_dtype"])
    if not sample_lines:
        raise ValueError("TXT 中没有 SAMPLES 数据段")
    text = "\n".join(sample_lines)
    if n_ch == 1:
        arr = np.array(sample_lines, dtype=dtype)
    else:
        arr = np.loadtxt(io.StringIO(text), delimiter=",", dtype=dtype)
    expected = int(header["n_samples"])
    if arr.shape[0] != expected:
        raise ValueError(f"样本数量不匹配: 头部 {expected}，实际 {arr.shape[0]}")
    wavfile.write(wav_path, fs, arr)
    print(f"已从 TXT 还原 WAV: {wav_path} ({fs} Hz, {arr.shape[0]} 样本)")
    return fs, arr, header


# ---------------- DFT 主频与绘图 ----------------

def _padded_spectrum(signal, fs, nfft=None, window="hann"):
    """零填充 DFT。默认 nfft 保证频率间隔 <= 0.1 Hz。返回 (freq, mag, phase)。"""
    signal = np.asarray(signal, dtype=np.float64)
    # 去除直流偏置与线性漂移，避免次声峰压过真实信号峰
    signal = detrend(signal, type="linear")
    n = len(signal)
    if n < 2:
        raise ValueError("信号长度过短，无法进行 DFT")
    if nfft is None:
        target = max(n, int(np.ceil(fs / 0.1)))
        nfft = 1
        while nfft < target:
            nfft <<= 1
    elif nfft < n:
        print(f"警告: nfft={nfft} 小于信号长度 {n}，已自动改为 {n}")
        nfft = n
    if fs / nfft > 0.1:
        print(f"警告: nfft={nfft} 时频率间隔 {fs / nfft:.4f} Hz > 0.1 Hz")
    if nfft > 2 ** 24:
        nfft = 2 ** 24
        print(f"警告: nfft 封顶为 2^24，频率间隔 {fs / nfft:.4f} Hz")
    if window and window != "none":
        win = windows.get_window(window, n, fftbins=True)
        win = win / win.mean()          # 相干增益补偿，保证幅值正确
        sig = signal * win
    else:
        sig = signal
    spec = np.fft.rfft(sig, n=nfft)
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = np.abs(spec) / n
    mag[1:] *= 2.0                      # 单侧幅值
    if nfft % 2 == 0:
        mag[-1] /= 2.0                  # Nyquist 不翻倍
    return freq, mag, np.angle(spec), nfft


def compute_main_frequencies(signal, fs, n_main=10, min_ratio=0.05,
                             min_hz=20.0, max_hz=None, nfft=None, window="hann"):
    """返回 (K,3)：主频(Hz)、线性幅值、相位(rad)，按幅值降序。"""
    freq, mag, phase, nfft = _padded_spectrum(signal, fs, nfft=nfft, window=window)
    if max_hz is None:
        max_hz = fs / 2.0
    band = (freq >= min_hz) & (freq <= max_hz)
    mag_band = mag.copy()
    mag_band[~band] = 0.0
    peak_max = np.max(mag_band)
    if peak_max <= 0:
        return np.zeros((0, 3))
    n = len(signal)
    lobe = max(1, int(2.0 * nfft / n))   # 主瓣半宽（插值点），排除旁瓣
    peaks, _ = find_peaks(mag_band, height=peak_max * min_ratio, distance=lobe)
    peaks = peaks[np.argsort(mag[peaks])[::-1]][:n_main]
    step = freq[1] - freq[0]
    results = []
    for i in peaks:
        if 1 <= i <= len(mag) - 2:
            y = np.log(np.maximum(mag[i - 1:i + 2], 1e-300))
            denom = y[0] - 2.0 * y[1] + y[2]
            delta = (
                float(np.clip(0.5 * (y[0] - y[2]) / denom, -0.5, 0.5))
                if abs(denom) > 1e-300
                else 0.0
            )
            amp = mag[i] - 0.25 * (mag[i - 1] - mag[i + 1]) * delta
        else:
            delta = 0.0
            amp = mag[i]
        results.append((freq[i] + delta * step, amp, phase[i]))
    return np.array(results)


def estimate_f0_summary(signal, fs, f0_min=50.0, f0_max=500.0,
                        frame_ms=40.0, hop_ms=10.0):
    """用 YIN(FFT 自相关) 估计整段基频，返回统计摘要。"""
    _, f0 = estimate_f0_curve(signal, fs, f0_min=f0_min, f0_max=f0_max,
                              frame_ms=frame_ms, hop_ms=hop_ms)
    v = f0[f0 > 0]
    return {
        "median": float(np.median(v)) if len(v) else float("nan"),
        "mean": float(np.mean(v)) if len(v) else float("nan"),
        "min": float(v.min()) if len(v) else float("nan"),
        "max": float(v.max()) if len(v) else float("nan"),
        "voiced_frames": int(len(v)),
        "total_frames": int(len(f0)),
    }


def classify_peaks(main_freqs, f0):
    """把 DFT 主频峰归类为基频 / 谐波 / 共振峰，返回标签列表。"""
    labels = []
    for f, a, p in np.asarray(main_freqs):
        if not np.isfinite(f0) or f0 <= 0:
            labels.append("peak")
            continue
        ratio = f / f0
        n = int(round(ratio))
        if n >= 1 and abs(ratio - n) <= 0.25:
            labels.append("F0" if n == 1 else f"harmonic_{n}")
        else:
            labels.append("formant")
    return labels


def estimate_f0_curve(signal, fs, f0_min=50.0, f0_max=500.0,
                      frame_ms=40.0, hop_ms=10.0,
                      threshold=0.15, max_cmndf=0.6, median_k=5,
                      window="hann"):
    """基于 FFT 自相关 + YIN 归一化差分（CMNDF）的基频曲线估计。

    原理（仍属 FFT 谱分析范畴）：
      1. 每帧加窗，用 FFT 快速计算自相关 r(tau) = IFFT(|FFT(x)|^2)；
      2. 差分函数 d(tau) = 2*(1 - r(tau)/r(0))，再做 YIN 累积均值归一化
         CMNDF(tau) = d(tau) / (mean(d(1..tau)))；
      3. 在 [fs/f0_max, fs/f0_min] 滞后区间取第一个低于 threshold 的
         局部极小（否则取全局极小），并做抛物线插值细化；
      4. 对浊音段做中值滤波平滑。

    返回 (times, f0_hz)，times 为帧中心时刻；无语音/不可靠帧 F0 记为 0。
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    frame_len = int(round(fs * frame_ms / 1000.0))
    hop = int(round(fs * hop_ms / 1000.0))
    if frame_len < 2 or hop < 1:
        raise ValueError("frame_ms/hop_ms 过小")
    nfft = 1
    while nfft < 2 * frame_len:
        nfft <<= 1
    win = windows.get_window(window, frame_len, fftbins=True)
    tau_lo = int(np.floor(fs / f0_max))
    tau_hi = int(np.ceil(fs / f0_min))
    tau_hi = min(tau_hi, frame_len)
    if tau_hi <= tau_lo:
        raise ValueError("f0_min/f0_max 范围过窄")
    taus = np.arange(tau_lo, tau_hi + 1)

    times = []
    f0s = []
    for start in range(0, n - frame_len + 1, hop):
        frame = signal[start:start + frame_len]
        frame = (frame - frame.mean()) * win   # 去除帧内直流后再自相关
        rms = float(np.sqrt(np.mean(frame ** 2)))
        times.append((start + frame_len / 2.0) / fs)
        if rms < 1e-4:
            f0s.append(0.0)
            continue
        # FFT 快速自相关
        spec = np.fft.rfft(frame, n=nfft)
        r = np.fft.irfft(np.abs(spec) ** 2, n=nfft)[:frame_len + 1]
        r0 = r[0]
        if r0 <= 0:
            f0s.append(0.0)
            continue
        r = r / r0
        d = 2.0 * (1.0 - r)
        cum = np.cumsum(d[1:])
        cmndf = d[taus] / (cum[taus - 1] / taus)

        # 第一个低于 threshold 的局部极小；否则全局极小
        best_i = None
        best_val = 1.0
        for i in range(1, len(cmndf) - 1):
            if cmndf[i] < cmndf[i - 1] and cmndf[i] <= cmndf[i + 1]:
                if cmndf[i] < threshold:
                    best_i = i
                    best_val = cmndf[i]
                    break
        if best_i is None:
            best_i = int(np.argmin(cmndf))
            best_val = cmndf[best_i]
        if best_val > max_cmndf:
            f0s.append(0.0)
            continue

        # 抛物线插值细化
        delta = 0.0
        if 1 <= best_i <= len(cmndf) - 2:
            y = cmndf[best_i - 1:best_i + 2]
            denom = y[0] - 2.0 * y[1] + y[2]
            if abs(denom) > 1e-12:
                delta = float(np.clip(0.5 * (y[0] - y[2]) / denom,
                                      -0.5, 0.5))
        tau_f = taus[best_i] + delta
        f0 = fs / tau_f
        f0s.append(f0 if f0_min <= f0 <= f0_max else 0.0)

    f0s = np.asarray(f0s, dtype=np.float64)
    if median_k >= 3:
        voiced = f0s > 0
        if voiced.sum() > median_k:
            f0s[voiced] = medfilt(f0s[voiced], kernel_size=median_k)
    return np.array(times), f0s


def plot_amplitude_response(signal, fs, main_freqs=None, out_png=None,
                            xmin=None, xmax=None,
                            ylin_min=None, ylin_max=None,
                            ydb_min=None, ydb_max=None,
                            title="幅频响应", window="hann",
                            logx=False, annotate_peaks=True,
                            languages=("ch", "en")):
    """绘制幅频响应：线性幅值与 dB 幅值分别输出，且分中英文（ch/en）存放。

    每张图都带有图例、坐标轴标签、网格、自适应量程与防重叠主频标注；
    out_png 为输出基底路径，实际保存到 out_png 所在目录的 ch/ 与 en/ 子目录。

    xmin/xmax  : 横轴（频率 Hz）范围，None 表示自适应
    ylin_min/max : 线性幅值图纵轴范围，None 表示自动（默认从 0 开始）
    ydb_min/max  : dB 幅值图纵轴范围，None 表示自动
    logx       : 是否使用对数频率轴
    annotate_peaks : 是否在主频位置标注数值
    languages  : 输出语言版本，默认 ("ch", "en")
    """
    return _plot_amplitude_split(signal, fs, main_freqs, out_png,
                          xmin, xmax, ylin_min, ylin_max,
                          ydb_min, ydb_max, title, window,
                          logx, annotate_peaks, languages)
    freq, mag, _, _ = _padded_spectrum(signal, fs, window=window)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 线性幅值子图
    axes[0].plot(freq, mag, lw=1.0, color="#1f77b4")
    axes[0].fill_between(freq, mag, alpha=0.14, color="#1f77b4")
    axes[0].set_ylabel("幅值 (线性) / Magnitude (linear)")
    axes[0].set_title(f"{title} — 线性幅频响应 / Linear", loc="left")

    # dB 幅值子图
    axes[1].plot(freq, mag_db, lw=1.0, color="#d62728")
    axes[1].fill_between(freq, mag_db, alpha=0.10, color="#d62728")
    axes[1].axhline(0.0, color="black", lw=0.6, alpha=0.5)
    axes[1].set_ylabel("幅值 (dB) / Magnitude (dB)")
    axes[1].set_xlabel("频率 (Hz) / Frequency (Hz)")
    axes[1].set_title(f"{title} — dB 幅频响应 / dB", loc="left")

    for ax in axes:
        ax.grid(True, which="major", linestyle="--", alpha=0.45)
        ax.grid(True, which="minor", linestyle=":", alpha=0.20)
        ax.minorticks_on()
    if main_freqs is not None and len(main_freqs) and annotate_peaks:
        for f, a, _ in np.asarray(main_freqs):
            if xmin is not None and f < xmin:
                continue
            if xmax is not None and f > xmax:
                continue
            db = 20.0 * np.log10(max(float(a), 1e-12))
            axes[0].axvline(f, color="#d62728", lw=0.7, alpha=0.35,
                            linestyle=":")
            axes[0].plot(f, a, "o", color="#d62728", ms=6, zorder=5)
            axes[1].plot(f, db, "o", color="#1f77b4", ms=6, zorder=5)
            axes[0].annotate(f"{f:.2f} Hz", xy=(f, a), xytext=(5, 7),
                             textcoords="offset points", fontsize=8,
                             color="#333333")
            axes[1].annotate(f"{f:.2f} Hz", xy=(f, db), xytext=(5, 7),
                             textcoords="offset points", fontsize=8,
                             color="#333333")
    x_lo = xmin if xmin is not None else 0.0
    x_hi = xmax if xmax is not None else fs / 2.0
    axes[0].set_xlim(x_lo, x_hi)
    axes[1].set_xlim(x_lo, x_hi)
    if logx:
        pos = freq > 0
        auto_lo = float(freq[pos].min())
        x_lo = max(xmin, auto_lo) if xmin is not None else auto_lo
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        fmt = FuncFormatter(lambda v, _pos: f"{v:g}")
        for ax in axes:
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_formatter(fmt)
        axes[0].set_xlim(x_lo, x_hi)
        axes[1].set_xlim(x_lo, x_hi)
    if ylin_min is not None or ylin_max is not None:
        axes[0].set_ylim(ylin_min, ylin_max)
    if ydb_min is not None or ydb_max is not None:
        axes[1].set_ylim(ydb_min, ydb_max)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"幅频响应图已保存: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def _suffixed_path(path, tag):
    if not path:
        return None
    stem, ext = os.path.splitext(path)
    return stem + tag + ext


def _apply_log_axis(ax, freq, x_lo, x_hi):
    pos = freq > 0
    auto_lo = float(freq[pos].min())
    lo = max(x_lo, auto_lo) if x_lo is not None else auto_lo
    ax.set_xscale("log")
    fmt = FuncFormatter(lambda v, _pos: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(fmt)
    ax.set_xlim(lo, x_hi)


LANG_TEXTS = {
    "ch": {
        "spec_lin": "幅值谱",
        "spec_db": "幅值谱 (dB)",
        "peaks": "主频",
        "xlabel": "频率 (Hz)",
        "ylin": "幅值 (线性)",
        "ydb": "幅值 (dB)",
        "title_lin": "线性幅频响应",
        "title_db": "dB 幅频响应",
    },
    "en": {
        "spec_lin": "Magnitude Spectrum",
        "spec_db": "Magnitude Spectrum (dB)",
        "peaks": "Main Peaks",
        "xlabel": "Frequency (Hz)",
        "ylin": "Magnitude (linear)",
        "ydb": "Magnitude (dB)",
        "title_lin": "Linear Amplitude Response",
        "title_db": "dB Amplitude Response",
    },
}


def _lang_path(path, lang):
    """把输出路径放入 ch/en 子文件夹（自动创建）。"""
    if not path:
        return None
    d = os.path.dirname(path)
    sub = os.path.join(d, lang)
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, os.path.basename(path))


def _rects_overlap(a, b, pad=5):
    return not (a.x1 + pad < b.x0 or b.x1 + pad < a.x0 or
                a.y1 + pad < b.y0 or b.y1 + pad < a.y0)


def _annotate_peaks(ax, peaks, color="#333333", max_labels=8):
    """标注主频数值；自动尝试多个偏移方向，避免文字框重叠。"""
    if peaks is None or len(peaks) == 0:
        return
    if len(peaks) > max_labels:
        order = np.argsort(peaks[:, 1])[::-1][:max_labels]
        peaks = peaks[order]
    peaks = peaks[np.argsort(peaks[:, 0])]
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    offsets = [(6, 8), (6, -16), (-80, 8), (6, 22),
               (-80, -16), (-150, 8), (6, 36), (-220, 8)]
    for f, a, _ in peaks:
        text = f"{f:.2f} Hz" if f < 1000 else f"{f:.1f} Hz"
        ann = None
        for dx, dy in offsets:
            ann = ax.annotate(text, xy=(f, a), xytext=(dx, dy),
                              textcoords="offset points", fontsize=8,
                              color=color)
            bb = ann.get_window_extent(renderer)
            if not any(_rects_overlap(bb, p) for p in placed):
                placed.append(bb)
                break
            ann.remove()
            ann = None
        if ann is None:
            ann = ax.annotate(text, xy=(f, a), xytext=(6, 8),
                              textcoords="offset points", fontsize=8,
                              color=color)
            placed.append(ann.get_window_extent(renderer))


def _plot_amplitude_linear(freq, mag, peaks, x_lo, x_hi, ylim, logx,
                           title, out_png, lang):
    """单独绘制线性幅频响应图：全图统一蓝色，带图例与合适量程。"""
    color = "#1f77b4"
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(freq, mag, lw=1.0, color=color, label=lang["spec_lin"])
    ax.fill_between(freq, mag, alpha=0.14, color=color)
    if peaks is not None and len(peaks):
        ax.plot(peaks[:, 0], peaks[:, 1], "o", color=color, ms=6,
                zorder=5, label=lang["peaks"])
    ax.set_xlabel(lang["xlabel"])
    ax.set_ylabel(lang["ylin"])
    ax.set_title(f"{title} — {lang['title_lin']}", loc="left")
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", alpha=0.20)
    ax.minorticks_on()
    ax.set_xlim(x_lo, x_hi)
    if logx:
        _apply_log_axis(ax, freq, x_lo, x_hi)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0.0, top=float(np.max(mag)) * 1.05)
    ax.legend(loc="best", framealpha=0.9)
    if peaks is not None and len(peaks):
        _annotate_peaks(ax, peaks)
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"线性幅频响应图已保存: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def _plot_amplitude_db(freq, mag_db, peaks, x_lo, x_hi, ylim, logx,
                       title, out_png, lang):
    """单独绘制 dB 幅频响应图：全图统一红色，带图例与合适量程。"""
    color = "#d62728"
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(freq, mag_db, lw=1.0, color=color, label=lang["spec_db"])
    ax.fill_between(freq, mag_db, alpha=0.10, color=color)
    if peaks is not None and len(peaks):
        db_peaks = np.column_stack([
            peaks[:, 0],
            20.0 * np.log10(np.maximum(peaks[:, 1], 1e-12)),
            np.zeros(len(peaks))])
        ax.plot(db_peaks[:, 0], db_peaks[:, 1], "o", color=color, ms=6,
                zorder=5, label=lang["peaks"])
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel(lang["xlabel"])
    ax.set_ylabel(lang["ydb"])
    ax.set_title(f"{title} — {lang['title_db']}", loc="left")
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", alpha=0.20)
    ax.minorticks_on()
    ax.set_xlim(x_lo, x_hi)
    if logx:
        _apply_log_axis(ax, freq, x_lo, x_hi)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=float(np.min(mag_db)) - 10.0, top=0.0)
    ax.legend(loc="best", framealpha=0.9)
    if peaks is not None and len(peaks):
        _annotate_peaks(ax, db_peaks)
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"dB 幅频响应图已保存: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def _plot_amplitude_split(signal, fs, main_freqs, out_png,
                          xmin, xmax, ylin_min, ylin_max,
                          ydb_min, ydb_max, title, window,
                          logx, annotate_peaks, languages):
    """把线性与 dB 幅频响应分别保存为 ch/en 下的独立图片。

    未手动指定坐标时自动选取合适范围：
      X 起点 = max(20 Hz, 0.2 × 最高主频)；
      X 终点 = min(fs/2, max(1000 Hz, 最高主频 × 1.5))；
      线性 Y 从 0 到峰值 1.05 倍；dB Y 从谱底-10 dB 到 0 dB。
    """
    freq, mag, _, _ = _padded_spectrum(signal, fs, window=window)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    peaks = None
    if main_freqs is not None and len(main_freqs):
        peaks = np.asarray(main_freqs)

    top_freq = float(peaks[0][0]) if peaks is not None and len(peaks) else max(fs / 4.0, 20.0)
    x_lo = xmin if xmin is not None else max(20.0, 0.2 * top_freq)
    if xmax is not None:
        x_hi = xmax
    else:
        peak_max_f = (float(peaks[:, 0].max())
                      if peaks is not None and len(peaks) else fs / 2.0)
        x_hi = min(fs / 2.0, max(1000.0, peak_max_f * 1.5))

    if peaks is not None and len(peaks):
        mask = (peaks[:, 0] >= x_lo) & (peaks[:, 0] <= x_hi)
        peaks = peaks[mask]
    if not annotate_peaks:
        peaks = None

    ylin = (ylin_min, ylin_max) if (ylin_min is not None
                                    or ylin_max is not None) else None
    ydb = (ydb_min, ydb_max) if (ydb_min is not None
                                 or ydb_max is not None) else None

    for lang_name in languages:
        lang = LANG_TEXTS[lang_name]
        linear_path = _lang_path(_suffixed_path(out_png, "_linear"), lang_name)
        db_path = _lang_path(_suffixed_path(out_png, "_db"), lang_name)
        _plot_amplitude_linear(freq, mag, peaks, x_lo, x_hi, ylin, logx,
                               title, linear_path, lang)
        _plot_amplitude_db(freq, mag_db, peaks, x_lo, x_hi, ydb, logx,
                           title, db_path, lang)
    print(f"幅频响应图已保存（ch/en）: {os.path.dirname(out_png)}")
