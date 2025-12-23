from pathlib import Path

from ultralytics import YOLO


def train_yolo_visdrone(
    data_yaml: str,
    pretrained_weights: str,
    output_dir: str,
    epochs: int,
) -> None:
    data_path = Path(data_yaml).expanduser()
    weights_path = Path(pretrained_weights).expanduser()
    save_dir = Path(output_dir).expanduser()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_path),
        epochs=epochs,
        project=str(save_dir),
        name="yolo_visdrone",
        verbose=True,
    )
