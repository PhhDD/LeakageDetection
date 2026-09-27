"""Image preprocessing tools."""

from pathlib import Path
from PIL import Image
from torchvision import transforms


def load_crop(config: dict) -> tuple[int, int, int, int]:
    """Extract crop coordinates from the model config."""

    crop = config["crop"]

    return (crop["left"], crop["top"], crop["right"], crop["bottom"])


def crop_image(
        image: Image.Image,
        crop: tuple[int, int, int, int],
) -> Image.Image:
    """Crop the image to the specified coordinates."""

    return image.crop(crop)


def get_eval_transform() -> transforms.Compose:
    """
    Get the evaluation transforms for the model.

    Grayscale images resized and converted to three channels
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def preprocess_image(
        path: str | Path,
        crop: tuple[int, int, int, int],
) -> Image.Image:
    """Load and preprocess an image for model evaluation."""

    image = Image.open(path).convert("L")
    return crop_image(image, crop)