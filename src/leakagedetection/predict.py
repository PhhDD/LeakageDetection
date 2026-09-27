"""Command-line interface for gate-leakage detection."""

import argparse
from pathlib import Path

from .inference import predict_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect gate leakage in stability-diagram images."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing raw PNG images.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for prediction CSV files.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    weights_path = project_root / "models" / (
        "resnet18_gate_leakage_augmented.pth"
    )

    config_path = project_root / "models" / "config.json"

    predict_directory(
        input_dir=args.input_dir,
        weights_path=weights_path,
        config_path=config_path,
        output_path=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()