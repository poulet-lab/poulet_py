import argparse
from pathlib import Path

from poulet_py.hardware.camera import BaslerCamera


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Preview and record from up to two Basler cameras simultaneously, "
            "saving videos, timestamps, metadata, and diagnostics."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "basler_dualcam_test",
        help="Directory where recording artifacts are saved.",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="Recording duration in seconds."
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Acquisition frame rate.")
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=2,
        help="Maximum number of Basler cameras to use.",
    )
    parser.add_argument(
        "--video-format",
        choices=("mp4", "avi"),
        default="mp4",
        help="Output video container.",
    )
    parser.add_argument(
        "--preview-key",
        default="e",
        help="Key used to stop preview/recording early.",
    )
    parser.add_argument("--window-width", type=int, default=None, help="Optional preview width.")
    parser.add_argument("--window-height", type=int, default=None, help="Optional preview height.")
    return parser


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera = BaslerCamera(max_cameras=args.max_cameras)
    diagnostics_paths = camera.recording(
        data_save_folder=str(args.output_dir),
        cage_id="test-cage",
        n_mouse=0,
        condition="dualcam-preview-record-test",
        mouse_ids=[],
        duration_s=args.duration,
        buffer_s=0,
        total_rec=1,
        fps=args.fps,
        video_format=args.video_format,
        show_preview=True,
        preview_key=args.preview_key,
        window_width=args.window_width,
        window_height=args.window_height,
    )

    print("Recording finished.")
    print(f"Output directory: {args.output_dir.resolve()}")
    if diagnostics_paths:
        print("Diagnostics files:")
        for path in diagnostics_paths:
            print(f" - {path}")


if __name__ == "__main__":
    main()
