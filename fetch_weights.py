"""Fetch the exact public checkpoints used in the toolbox comparison."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    from huggingface_hub import snapshot_download

    snapshot_download(
        "Auwomo/Awomo-0.5I-Robodojo",
        revision="bb15096482e0e60a505e80953635d819802d9916",
        allow_patterns=["model.pt", "awomo_config.json", "dataset_stats.json", "SHA256SUMS"],
        local_dir=args.output / "awomo",
        token=False,
    )
    snapshot_download(
        "Qwen/Qwen3-VL-4B-Instruct",
        revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        local_dir=args.output / "qwen3-vl-4b",
        token=False,
    )
    snapshot_download(
        "black-forest-labs/FLUX.2-klein-base-4B",
        revision="a3b4f4849157f664bdbc776fd7453c2783562f4d",
        allow_patterns=["vae/*"],
        local_dir=args.output / "flux2",
        token=False,
    )


if __name__ == "__main__":
    main()
