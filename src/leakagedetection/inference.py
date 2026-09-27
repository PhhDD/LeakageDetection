"""Inference utilities for gate-leakage detection."""

from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from .model import load_model
from .preprocessing import preprocess_image, get_eval_transform, load_crop

class ImageDataset(Dataset):
    """Dataset for inference on raw images."""

    def __init__(
        self,
        files: list[Path],
        crop: tuple[int, int, int, int],
    ):
        self.files = files
        self.crop = crop
        self.transform = get_eval_transform()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        image = preprocess_image(path, self.crop)
        image = self.transform(image)

        return image, path.name


def predict_directory(
    input_dir: str | Path,
    weights_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    """
    Run the inference on all png images in a directory.

    Returns a DataFrame containing filename, leakage probability,
    and binanry prediction (0 for no leakage, 1 for leakage).
    """

    input_dir = Path(input_dir)
    weights_path = Path(weights_path)
    config_path = Path(config_path)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r") as f:
        config = json.load(f)

    threshold = float(config.get("threshold"))
    crop = load_crop(config)
    files = sorted(input_dir.glob("*.png"))
    if not files:
        raise ValueError(f"No PNG images found in {input_dir}")
    
    dataset = ImageDataset(files, crop)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(weights_path, device)
    torch_device = torch.device(device)

    results = []

    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(torch_device)
            logits = model(images).squeeze(1)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            for filename, prob in zip(filenames, probabilities):
                probability = float(prob)
                results.append({
                    "filename": filename,
                    "prob_leakage": probability,
                    "prediction": int(probability >= threshold),
                })

    df_results = pd.DataFrame(results)

    prediction_path = output_path / "predictions.csv"
    keep_path = output_path / "keep.csv"
    flagged_path = output_path / "flagged.csv"

    df_results.to_csv(prediction_path, index=False)

    df_results[
        df_results["prediction"] == 0
    ][["filename", "prob_leakage"]].to_csv(keep_path, index=False)

    df_results[
        df_results["prediction"] == 1
    ][["filename", "prob_leakage"]].to_csv(flagged_path, index=False)

    print(f"Processed:          {len(df_results)}")
    print(f"Keep:               {(df_results['prediction'] == 0).sum()}")
    print(f"Potential leakage:  {(df_results['prediction'] == 1).sum()}")
    print(f"Threshold:          {threshold:.2f}")
    print()
    print(f"Predictions:        {prediction_path}")
    print(f"Keep list:          {keep_path}")
    print(f"Flagged list:       {flagged_path}")

    return df_results