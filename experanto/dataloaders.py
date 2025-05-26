from typing import Any, List, Optional, Tuple
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pathlib import Path

from .datasets import ChunkDataset
from .utils import MultiEpochsDataLoader, LongCycler


def get_multisession_dataloader(paths, config: DictConfig) -> DataLoader:
    dataloaders = {}
    for i, path in enumerate(paths):
        dataset_name = Path(path).name
        dataset = ChunkDataset(path, **config.dataset,)
        if len(dataset) == 0:
            config.dataloader.shuffle = False # random sampler fails for empty datasets
        dataloaders[dataset_name] = MultiEpochsDataLoader(dataset,
                                               **config.dataloader,
                                               )
    return LongCycler(dataloaders)