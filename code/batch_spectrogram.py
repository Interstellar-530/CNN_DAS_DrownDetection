"""
batch_spectrogram.py —— 批量生成 log-mel 谱图，用于 CNN 训练/验证数据准备。

示例:
  python batch_spectrogram.py -i dataset -o spec_output -n mydata --split 0.8
"""

import argparse

from audio_tools import batch_generate_spectrograms


def main():
    parser = argparse.ArgumentParser(
        description="批量生成 log-mel 谱图（.npy + 可选 .png + manifest.csv）")
    parser.add_argument("--input", "-i", required=True,
                        help="输入音频目录（可递归）")
    parser.add_argument("--output", "-o", required=True,
                        help="输出大文件夹")
    parser.add_argument("--name", "-n", required=True,
                        help="自定义名称前缀")
    parser.add_argument("--source", "-s", default=None,
                        help="数据来源标签，写入文件名与配置文件")
    parser.add_argument("--label", "-l", default=None,
                        help="固定类别标签（覆盖子目录自动标签）")
    parser.add_argument("--dataset", default=None,
                        help="数据集名称（用于 params.json 与 manifest）")
    parser.add_argument("--extensions", default="wav,flac,ogg,mp3",
                        help="逗号分隔的扩展名")
    parser.add_argument("--no-label", action="store_true",
                        help="不从子目录名推断类别标签")
    parser.add_argument("--no-png", action="store_true",
                        help="不生成任何预览 PNG（默认仅第一段生成一张）")
    parser.add_argument("--split", type=float, default=None,
                        help="训练集比例(0~1)，例如 0.8")
    parser.add_argument("--max-files", type=int, default=0,
                        help="最多处理多少个文件（0=全部）")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--fmin", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--top-db", type=float, default=80.0)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--target-sr", type=int, default=16000,
                        help="统一重采样采样率；0 表示保留原始采样率")
    args = parser.parse_args()

    extensions = tuple(e.strip() for e in args.extensions.split(",")
                       if e.strip())
    manifest, csv_path, params_path = batch_generate_spectrograms(
        args.input, args.output, args.name,
        extensions=extensions,
        label_from_subdir=not args.no_label,
        preview_png=not args.no_png,
        split=args.split,
        max_files=args.max_files or None,
        source_tag=args.source,
        fixed_label=args.label,
        dataset_name=args.dataset,
        n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels,
        fmin=args.fmin, fmax=args.fmax, top_db=args.top_db,
        fixed_time_frames=args.frames,
        target_sr=args.target_sr or None)
    print(f"完成 / Done: 共 {len(manifest)} 个谱图")
    print(f"清单 / Manifest: {csv_path}")
    print(f"参数 / Params: {params_path}")


if __name__ == "__main__":
    main()
