import os
import random
import subprocess
import sys
import zipfile
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split


def _setup_kaggle_credentials():
    """Carga el token nuevo de Kaggle (KAGGLE_API_TOKEN, formato 'KGAT_...') desde
    Colab Secrets si está disponible.

    En Colab: Secrets (icono de llave en el panel izquierdo) -> secret llamado
    KAGGLE_KEY con el token (Kaggle -> Settings -> API -> Create New Token).
    Si no estamos en Colab, asume que ya está seteada como variable de entorno
    o que existe ~/.kaggle/kaggle.json (flujo estándar de la lib kaggle).
    """
    try:
        from google.colab import userdata
    except ImportError:
        return

    if "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_KEY")


def download_datasets(base_dir):
    """Descarga LOL-v1 (Kaggle) y LOL-v2-real (HuggingFace) en base_dir/raw.

    Pensada para correrse una vez por sesión de Colab (el runtime arranca
    limpio cada vez), por eso no chequea si ya existen los datos.
    """
    raw_dir = os.path.join(base_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    _setup_kaggle_credentials()
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("soumikrakshit/lol-dataset",
                                       path=raw_dir, quiet=False)
    zip_path = os.path.join(raw_dir, "lol-dataset.zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(os.path.join(raw_dir, "lol-dataset"))
    os.remove(zip_path)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="okhater/lolv2-real", repo_type="dataset",
                       local_dir=os.path.join(raw_dir, "lolv2-real"),
                       ignore_patterns=["*.gitignore"])


class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_dir   = low_dir
        self.high_dir  = high_dir
        self.filenames = sorted(os.listdir(low_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname    = self.filenames[idx]
        low_img  = Image.open(os.path.join(self.low_dir,  fname)).convert("RGB")
        high_img = Image.open(os.path.join(self.high_dir, fname)).convert("RGB")
        low  = torch.from_numpy(np.array(low_img,  dtype=np.float32) / 255.0).permute(2, 0, 1)
        high = torch.from_numpy(np.array(high_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return low, high


class AugmentSubset(Dataset):
    def __init__(self, subset, augment=False):
        self.subset  = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        low, high = self.subset[idx]
        low  = Image.fromarray((low.permute(1, 2, 0).numpy()  * 255).astype(np.uint8))
        high = Image.fromarray((high.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        if self.augment:
            i = random.randint(0, low.height - 128)
            j = random.randint(0, low.width  - 128)
            low  = low.crop((j, i, j + 128, i + 128))
            high = high.crop((j, i, j + 128, i + 128))
            if random.random() > 0.5:
                low  = low.transpose(Image.FLIP_LEFT_RIGHT)
                high = high.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                low  = low.transpose(Image.FLIP_TOP_BOTTOM)
                high = high.transpose(Image.FLIP_TOP_BOTTOM)

        low  = torch.from_numpy(np.array(low,  dtype=np.float32) / 255.0).permute(2, 0, 1)
        high = torch.from_numpy(np.array(high, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return low, high


def _build_split(base_dir, seed=42):
    dataset_v1 = LowLightDataset(
        low_dir  = os.path.join(base_dir, "raw/lol-dataset/lol_dataset/our485/low"),
        high_dir = os.path.join(base_dir, "raw/lol-dataset/lol_dataset/our485/high"),
    )

    dataset_v2 = LowLightDataset(
        low_dir  = os.path.join(base_dir, "raw/lolv2-real/Train/Input"),
        high_dir = os.path.join(base_dir, "raw/lolv2-real/Train/GT"),
    )

    full_dataset = ConcatDataset([dataset_v1, dataset_v2])

    n       = len(full_dataset)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)
    n_test  = n - n_train - n_val

    return random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )


def get_dataset_split(base_dir, seed=42):
    train_set, val_set, test_set = _build_split(base_dir, seed)

    train_data = AugmentSubset(train_set, augment=True)
    val_data   = AugmentSubset(val_set,   augment=False)
    test_data  = AugmentSubset(test_set,  augment=False)

    return train_data, val_data, test_data


def get_dataloaders(base_dir, batch_size_train=8, batch_size_val=None, batch_size_test=None,
                     seed=42, num_workers=0, pin_memory=False):
    if batch_size_val is None:
        batch_size_val = batch_size_train
    if batch_size_test is None:
        batch_size_test = batch_size_val

    train_data, val_data, test_data = get_dataset_split(base_dir, seed)

    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True,
                               num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_data,   batch_size=batch_size_val,   shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_data,  batch_size=batch_size_test,  shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
