from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset #, EgoDataset
from l5kit.rasterization import build_rasterizer
from torch.utils.data import get_worker_info
from typing import Optional
import numpy as np

class MyTrainDataset:
    def __init__(self, cfg, dm, agents_mask: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.dm = dm
        self.has_init = False
        self.agents_mask = agents_mask
    def initialize(self, worker_id):
        print('initialize called with worker_id', worker_id)
        rasterizer = build_rasterizer(self.cfg, self.dm)
        train_cfg = self.cfg["train_data_loader"]
        train_zarr = ChunkedDataset(self.dm.require(train_cfg["key"])).open(cached=False)  # try to turn off cache
        self.dataset = AgentDataset(self.cfg, train_zarr, rasterizer, self.agents_mask)
        self.has_init = True
    def reset(self):
        self.dataset = None
        self.has_init = False
    def __len__(self):
        # note you have to figure out the actual length beforehand since once the rasterizer and/or AgentDataset been constructed, you cannot pickle it anymore! So we can't compute the size from the real dataset. However, DataLoader require the len to determine the sampling.
        return 111634
    def __getitem__(self, index):
        return self.dataset[index]    

def my_dataset_worker_init_func(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.initialize(worker_id)