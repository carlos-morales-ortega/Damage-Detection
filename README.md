# Damage Build Detector

Damage Build Detector is a PyTorch-based notebook for training a lightweight U-Net model to detect building damage from paired pre-disaster and post-disaster images.

## Project overview

- Loads pairs of pre-disaster and post-disaster satellite or aerial images.
- Builds binary masks from JSON annotations for building footprints.
- Computes a damage mask that marks destroyed buildings and new buildings.
- Trains a lightweight U-Net segmentation model on the post-disaster images.
- Validates the model and visualizes predictions with color overlays.

## Notebook workflow

The notebook executes the following main steps:
1. Mounts Google Drive (optional) and sets the data path.
2. Defines a custom `DisasterDataset` to load images and JSON labels.
3. Converts labeled building polygons into segmentation masks.
4. Defines a lightweight U-Net model (`UNetLite`).
5. Loads train/test datasets from `data/train` and `data/test`.
6. Optionally runs a fast test mode with a small sample subset.
7. Trains the model, validates it, and saves the best weights.
8. Plots training/validation loss and displays test predictions.

## Data structure

The notebook expects a folder named `data` with this structure:

```
data/
  train/
    images/
    labels/
  test/
    images/
    labels/
```

- `images/` should contain paired images named like `XXXX_pre_disaster.png` and `XXXX_post_disaster.png`.
- `labels/` should contain corresponding JSON annotation files named like `XXXX_pre_disaster.json` and `XXXX_post_disaster.json`.
- The dataset loader also supports files prefixed with `copia de ` if copied from Google Drive.

## Requirements

- Python 3.8+ (recommended)
- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow
- tqdm

## How to run

1. Open `Damage_Detection.ipynb`.
2. Set the `BASE_PATH` variable to your dataset folder, for example:
   `/content/drive/MyDrive/data_pruebas`
3. If using Google Colab, mount Google Drive when prompted.
4. Run the notebook cells in order.

## Notes

- The notebook uses a simplified `UNetLite` architecture with 3 output classes.
- The loss function applies higher weights to damaged classes to prioritize damage detection.
- The script saves the best model weights to `/content/drive/MyDrive/best_damage_model.pth`.
- Visualization overlays show building masks and predicted damage areas.
