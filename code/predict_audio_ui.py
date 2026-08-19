"""Simple UI for testing the trained CNN with a local audio file."""

from __future__ import annotations

import argparse
import threading
import traceback
import sys
from pathlib import Path
from tkinter import END, BOTH, DISABLED, NORMAL, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from audio_tools import compute_log_mel_spectrogram, load_audio_file
from train_cnn_spectrogram import SpectrogramCNN, SpectrogramCNNV2, load_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SUBMISSION_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL = SUBMISSION_ROOT / "model" / "best_model.pt"
DEFAULT_CLASS_NAMES = ["normal_speech", "scream", "water_splash", "other_non_scream"]

SPECTROGRAM_CONFIG = {
    "n_fft": 1024,
    "hop": 256,
    "n_mels": 128,
    "fmin": 50.0,
    "fmax": 8000.0,
    "top_db": 80.0,
    "fixed_time_frames": 128,
    "target_sr": 16000,
}


class SpectrogramEnsemble(torch.nn.Module):
    def __init__(
        self,
        models: list[torch.nn.Module],
        means: list[float],
        stds: list[float],
        fusion_weight_b: float,
    ) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.means = [float(m) for m in means]
        self.stds = [float(s) for s in stds]
        self.fusion_weight_b = float(fusion_weight_b)

    def _norm(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - self.means[idx]) / max(self.stds[idx], 1e-6)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        pa = torch.softmax(self.models[0](self._norm(x_raw, 0)), dim=1)
        pb = torch.softmax(self.models[1](self._norm(x_raw, 1)), dim=1)
        wb = self.fusion_weight_b
        return (1.0 - wb) * pa + wb * pb


def seconds_to_mmss(value: float) -> str:
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def format_probs(class_names: list[str], probs: np.ndarray) -> str:
    lines = []
    for name, prob in zip(class_names, probs):
        lines.append(f"  {name}: {prob:.4f}")
    return "\n".join(lines)


class AudioCnnPredictor:
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = load_checkpoint(self.model_path, self.device)
        self.class_names = list(checkpoint.get("class_names") or DEFAULT_CLASS_NAMES)
        self.ensemble = bool(checkpoint.get("ensemble", False))

        if self.ensemble:
            models, means, stds = [], [], []
            for entry in checkpoint.get("models", []):
                model_config = entry.get("model_config", {}) or {}
                model = SpectrogramCNNV2(
                    num_classes=len(self.class_names),
                    dropout=float(model_config.get("dropout", 0.5)),
                    width=int(model_config.get("width", 48)),
                    dual_pool=bool(model_config.get("dual_pool", True)),
                ).to(self.device)
                model.load_state_dict(entry["model_state"])
                model.eval()
                models.append(model)
                means.append(float(entry.get("train_mean", -14.0)))
                stds.append(float(entry.get("train_std", 22.0)))
            self.model = SpectrogramEnsemble(
                models, means, stds, float(checkpoint.get("fusion_weight_b", 0.5))
            ).to(self.device)
            self.model.eval()
            self.mean = None
            self.std = None
            return

        self.mean = float(checkpoint.get("train_mean", -14.238170409173696))
        self.std = float(checkpoint.get("train_std", 21.23290543312406))
        if self.std <= 0:
            self.std = 1.0

        model_class = str(checkpoint.get("model_class", "SpectrogramCNN"))
        model_config = checkpoint.get("model_config", {}) or {}
        if model_class == "SpectrogramCNNV2":
            self.model = SpectrogramCNNV2(
                num_classes=len(self.class_names),
                dropout=float(model_config.get("dropout", 0.25)),
                width=int(model_config.get("width", 32)),
                dual_pool=bool(model_config.get("dual_pool", True)),
            ).to(self.device)
        else:
            self.model = SpectrogramCNN(
                num_classes=len(self.class_names),
                dropout=float(model_config.get("dropout", 0.25)),
            ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict_array(self, signal: np.ndarray, fs: int) -> tuple[str, np.ndarray]:
        mel, _ = compute_log_mel_spectrogram(signal, fs, **SPECTROGRAM_CONFIG)
        with torch.no_grad():
            if self.ensemble:
                x = torch.from_numpy(mel.astype(np.float32)[None, None, :, :]).to(self.device)
                probs = self.model(x).cpu().numpy()[0]
            else:
                arr = ((mel.astype(np.float32) - self.mean) / self.std).astype(np.float32)
                x = torch.from_numpy(arr[None, None, :, :]).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_index = int(np.argmax(probs))
        return self.class_names[pred_index], probs

    def predict_file_whole(self, audio_path: Path) -> dict[str, object]:
        fs, signal = load_audio_file(str(audio_path), normalise=False)
        signal = np.asarray(signal, dtype=np.float64)
        if signal.size == 0:
            raise ValueError("Audio file is empty.")
        duration = float(signal.size / fs)
        label, probs = self.predict_array(signal, fs)
        return {
            "path": Path(audio_path),
            "sample_rate": int(fs),
            "duration": duration,
            "prediction": label,
            "probs": probs,
        }

    def predict_file_windows(
        self,
        audio_path: Path,
        window_seconds: float = 5.0,
        hop_seconds: float = 2.5,
    ) -> dict[str, object]:
        fs, signal = load_audio_file(str(audio_path), normalise=False)
        signal = np.asarray(signal, dtype=np.float64)
        if signal.size == 0:
            raise ValueError("Audio file is empty.")

        duration = float(signal.size / fs)
        window_samples = max(int(round(window_seconds * fs)), 1)
        hop_samples = max(int(round(hop_seconds * fs)), 1)

        starts = []
        if signal.size <= window_samples:
            starts = [0]
        else:
            start = 0
            while start + window_samples <= signal.size:
                starts.append(start)
                start += hop_samples
            last_start = signal.size - window_samples
            if starts[-1] != last_start:
                starts.append(last_start)

        windows = []
        for start in starts:
            end = min(start + window_samples, signal.size)
            clip = signal[start:end]
            label, probs = self.predict_array(clip, fs)
            windows.append(
                {
                    "start": float(start / fs),
                    "end": float(end / fs),
                    "prediction": label,
                    "probs": probs,
                }
            )

        return {
            "path": Path(audio_path),
            "sample_rate": int(fs),
            "duration": duration,
            "window_seconds": float(window_seconds),
            "hop_seconds": float(hop_seconds),
            "windows": windows,
        }


def format_whole_result(predictor: AudioCnnPredictor, result: dict[str, object]) -> str:
    return "\n".join(
        [
            "整段音频测试结果",
            "",
            f"模型文件：{predictor.model_path}",
            f"音频文件：{result['path']}",
            f"采样率：{result['sample_rate']} Hz",
            f"时长：{float(result['duration']):.2f} 秒",
            "",
            f"预测类别：{result['prediction']}",
            "",
            "四类概率：",
            format_probs(predictor.class_names, result["probs"]),
            "",
            "说明：当前模型是片段级 CNN。长音频建议同时使用“5秒窗口测试”。",
        ]
    )


def format_window_result(predictor: AudioCnnPredictor, result: dict[str, object]) -> str:
    lines = [
        "5秒窗口测试结果",
        "",
        f"模型文件：{predictor.model_path}",
        f"音频文件：{result['path']}",
        f"采样率：{result['sample_rate']} Hz",
        f"时长：{float(result['duration']):.2f} 秒",
        f"窗口长度：{result['window_seconds']} 秒",
        f"窗口步长：{result['hop_seconds']} 秒",
        "",
    ]

    counts = {name: 0 for name in predictor.class_names}
    max_probs = {name: 0.0 for name in predictor.class_names}

    for index, item in enumerate(result["windows"], start=1):
        pred = item["prediction"]
        probs = item["probs"]
        counts[pred] += 1
        for name, prob in zip(predictor.class_names, probs):
            max_probs[name] = max(max_probs[name], float(prob))

        lines.extend(
            [
                f"窗口 {index}: {seconds_to_mmss(item['start'])} - {seconds_to_mmss(item['end'])}",
                f"预测类别：{pred}",
                "四类概率：",
                format_probs(predictor.class_names, probs),
                "",
            ]
        )

    lines.append("窗口统计：")
    for name in predictor.class_names:
        lines.append(f"  {name}: {counts[name]} 个窗口，最高概率 {max_probs[name]:.4f}")
    lines.extend(
        [
            "",
            "说明：该结果只表示每个窗口的声音类别，不等于 need_rescue 时序判定。",
        ]
    )
    return "\n".join(lines)


class AudioTestUi:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("CNN 音频测试")
        self.root.geometry("900x680")
        self.root.minsize(760, 560)

        self.audio_path_var = StringVar()
        self.model_path_var = StringVar(value=str(DEFAULT_MODEL))
        self.status_var = StringVar(value="请选择音频文件。")
        self.predictor: AudioCnnPredictor | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        main = ttk.Frame(self.root)
        main.pack(fill=BOTH, expand=True, padx=12, pady=12)

        ttk.Label(main, text="音频文件").grid(row=0, column=0, sticky="w", **pad)
        audio_entry = ttk.Entry(main, textvariable=self.audio_path_var)
        audio_entry.grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(main, text="选择音频", command=self.choose_audio).grid(row=0, column=2, sticky="ew", **pad)

        ttk.Label(main, text="模型文件").grid(row=1, column=0, sticky="w", **pad)
        model_entry = ttk.Entry(main, textvariable=self.model_path_var)
        model_entry.grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(main, text="选择模型", command=self.choose_model).grid(row=1, column=2, sticky="ew", **pad)

        button_row = ttk.Frame(main)
        button_row.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=8)
        ttk.Button(button_row, text="整段测试", command=lambda: self.run_prediction("whole")).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="5秒窗口测试", command=lambda: self.run_prediction("windows")).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="清空结果", command=self.clear_output).pack(side="left")

        ttk.Label(main, textvariable=self.status_var).grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        self.output = tk_text = __import__("tkinter").Text(main, wrap="word", height=24)
        tk_text.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=10, pady=8)

        scrollbar = ttk.Scrollbar(main, orient="vertical", command=tk_text.yview)
        scrollbar.grid(row=4, column=3, sticky="ns", pady=8)
        tk_text.configure(yscrollcommand=scrollbar.set)

        main.columnconfigure(1, weight=1)
        main.rowconfigure(4, weight=1)

    def choose_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[
                ("Audio files", "*.wav *.flac *.ogg *.mp3"),
                ("WAV files", "*.wav"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.audio_path_var.set(path)
            self.status_var.set("音频文件已选择。")

    def choose_model(self) -> None:
        path = filedialog.askopenfilename(
            title="选择模型文件",
            initialdir=str(DEFAULT_MODEL.parent) if DEFAULT_MODEL.parent.exists() else str(SCRIPT_DIR),
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)
            self.predictor = None
            self.status_var.set("模型文件已选择。")

    def clear_output(self) -> None:
        self.output.configure(state=NORMAL)
        self.output.delete("1.0", END)
        self.output.configure(state=NORMAL)
        self.status_var.set("结果已清空。")

    def append_output(self, text: str) -> None:
        self.output.configure(state=NORMAL)
        self.output.delete("1.0", END)
        self.output.insert(END, text)
        self.output.configure(state=NORMAL)

    def get_predictor(self, model_path: Path) -> AudioCnnPredictor:
        if self.predictor is None or self.predictor.model_path != model_path.resolve():
            self.predictor = AudioCnnPredictor(model_path)
        return self.predictor

    def run_prediction(self, mode: str) -> None:
        audio_path = Path(self.audio_path_var.get()).expanduser()
        if not audio_path.exists():
            messagebox.showerror("错误", "请先选择有效的音频文件。")
            return
        model_path = Path(self.model_path_var.get()).expanduser()
        self.status_var.set("正在加载模型...")

        def worker() -> None:
            try:
                predictor = self.get_predictor(model_path)
                self.root.after(0, lambda: self.status_var.set("正在测试音频..."))
                if mode == "windows":
                    result = predictor.predict_file_windows(audio_path)
                    text = format_window_result(predictor, result)
                else:
                    result = predictor.predict_file_whole(audio_path)
                    text = format_whole_result(predictor, result)
                self.root.after(0, lambda: self.append_output(text))
                self.root.after(0, lambda: self.status_var.set("测试完成。"))
            except Exception as exc:
                detail = traceback.format_exc()
                self.root.after(0, lambda: self.status_var.set("测试失败。"))
                self.root.after(0, lambda: messagebox.showerror("测试失败", f"{exc}\n\n{detail}"))

        threading.Thread(target=worker, daemon=True).start()

    def run(self) -> None:
        self.root.mainloop()


def run_cli(args: argparse.Namespace) -> None:
    predictor = AudioCnnPredictor(Path(args.model))
    audio_path = Path(args.file)
    if args.windows:
        result = predictor.predict_file_windows(
            audio_path,
            window_seconds=float(args.window_seconds),
            hop_seconds=float(args.hop_seconds),
        )
        print(format_window_result(predictor, result))
    else:
        result = predictor.predict_file_whole(audio_path)
        print(format_whole_result(predictor, result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the trained CNN with an audio file.")
    parser.add_argument("--file", type=str, default=None, help="Audio file to test. If omitted, the UI starts.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to best_model.pt.")
    parser.add_argument("--windows", action="store_true", help="Run 5-second sliding-window prediction.")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--hop-seconds", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file:
        run_cli(args)
    else:
        AudioTestUi().run()


if __name__ == "__main__":
    main()
