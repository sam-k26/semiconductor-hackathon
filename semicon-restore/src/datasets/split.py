import json
from pathlib import Path

from torch.utils.data import Subset

from src.datasets.kla_dataset import KLADataset


def load_split_ids(split_path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(split_path.read_text())
    return data["train_ids"], data["val_ids"]


def subset_by_ids(dataset: KLADataset, ids: list[str]) -> Subset:
    id_to_index = {
        image_id: i
        for i, (image_id, _, _) in enumerate(dataset.samples)
    }
    indices = [id_to_index[image_id] for image_id in ids]
    return Subset(dataset, indices)
