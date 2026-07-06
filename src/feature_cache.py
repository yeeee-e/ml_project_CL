from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from .datasets import ensure_dir


BackboneName = Literal["resnet18", "vit_b_16", "clip_vit_b_16"]
DatasetName = Literal["cifar10", "cifar100", "tinyimagenet"]
CLIP_VIT_B16_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
)


class ClipImageEncoder(nn.Module):
    def __init__(self, model: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _clip_vit_b16_path() -> Path:
    if os.environ.get("CLIP_VIT_B16_PATH"):
        return Path(os.environ["CLIP_VIT_B16_PATH"]).expanduser()
    local_path = Path("models") / "ViT-B-16.pt"
    if local_path.exists():
        return local_path
    return Path.home() / ".cache" / "clip" / "ViT-B-16.pt"


def _build_clip_vit_b16(device: torch.device) -> tuple[nn.Module, transforms.Compose, int]:
    model_path = _clip_vit_b16_path()
    if not model_path.exists():
        raise FileNotFoundError(
            "CLIP ViT-B/16 weights were not found. Download ViT-B-16.pt from "
            f"{CLIP_VIT_B16_URL} and place it at models/ViT-B-16.pt, "
            f"{model_path}, or set CLIP_VIT_B16_PATH."
        )
    model = torch.jit.load(str(model_path), map_location=device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return ClipImageEncoder(model).eval().to(device), transform, 512


def _build_backbone(name: BackboneName, device: torch.device) -> tuple[nn.Module, transforms.Compose, int]:
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        transform = weights.transforms()
    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        feature_dim = model.heads.head.in_features
        model.heads = nn.Identity()
        transform = weights.transforms()
    elif name == "clip_vit_b_16":
        return _build_clip_vit_b16(device)
    else:
        raise ValueError(f"unsupported backbone: {name}")
    model.eval().to(device)
    return model, transform, feature_dim


@torch.inference_mode()
def _extract_split(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for images, targets in tqdm(loader, desc="extract", leave=False):
        images = images.to(device, non_blocking=True)
        out = model(images).detach().cpu().float().numpy()
        features.append(out)
        labels.append(targets.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0).astype(np.int64)


class TinyImageNetValDataset(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], transform) -> None:
        self.root = root
        self.transform = transform
        val_root = _resolve_tiny_root(root) / "val"
        anno_path = val_root / "val_annotations.txt"
        self.samples: list[tuple[Path, int]] = []
        with anno_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, wnid = parts[0], parts[1]
                self.samples.append((val_root / "images" / image_name, class_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _resolve_tiny_root(data_dir: Path) -> Path:
    direct = data_dir / "tiny-imagenet-200"
    nested = data_dir / "tiny-imagenet-200" / "tiny-imagenet-200"
    if (direct / "train").exists() and (direct / "val").exists():
        return direct
    if (nested / "train").exists() and (nested / "val").exists():
        return nested
    return direct


def _build_dataset(dataset_name: DatasetName, data_dir: Path, transform):
    if dataset_name == "cifar10":
        _extract_tar_if_needed(data_dir / "cifar-10-python.tar.gz", data_dir, data_dir / "cifar-10-batches-py")
        train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        _extract_tar_if_needed(data_dir / "cifar-100-python.tar.gz", data_dir, data_dir / "cifar-100-python")
        train_set = datasets.CIFAR100(root=str(data_dir), train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root=str(data_dir), train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == "tinyimagenet":
        tiny_root = _resolve_tiny_root(data_dir)
        train_root = tiny_root / "train"
        if not train_root.exists():
            raise FileNotFoundError(
                "Tiny-ImageNet not found. Expected data/tiny-imagenet-200/train "
                "or data/tiny-imagenet-200/tiny-imagenet-200/train. "
                "Extract tiny-imagenet-200.zip under the configured data_dir."
            )
        train_set = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_set = TinyImageNetValDataset(data_dir, train_set.class_to_idx, transform)
        num_classes = 200
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    return train_set, test_set, num_classes


def _extract_tar_if_needed(archive_path: Path, extract_dir: Path, expected_dir: Path) -> None:
    if expected_dir.exists():
        return
    if not archive_path.exists():
        return
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)


def build_or_load_features(
    data_dir: str | Path,
    cache_dir: str | Path,
    dataset_name: DatasetName = "cifar100",
    backbone: BackboneName = "resnet18",
    batch_size: int = 256,
    num_workers: int = 4,
    force: bool = False,
) -> Path:
    data_dir = ensure_dir(data_dir)
    cache_dir = ensure_dir(cache_dir)
    cache_path = cache_dir / f"{dataset_name}_{backbone}_features.npz"
    meta_path = cache_dir / f"{dataset_name}_{backbone}_features.meta.json"
    if cache_path.exists() and not force:
        return cache_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, feature_dim = _build_backbone(backbone, device)
    train_set, test_set, num_classes = _build_dataset(dataset_name, data_dir, transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    train_features, train_labels = _extract_split(model, train_loader, device)
    test_features, test_labels = _extract_split(model, test_loader, device)
    np.savez_compressed(
        cache_path,
        train_features=train_features.astype(np.float32),
        train_labels=train_labels,
        test_features=test_features.astype(np.float32),
        test_labels=test_labels,
    )
    meta = {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "backbone": backbone,
        "feature_dim": feature_dim,
        "train_samples": int(len(train_labels)),
        "test_samples": int(len(test_labels)),
        "device": str(device),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cache_path
