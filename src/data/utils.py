


import torch


class CollateMnist():
    def __init__():
        pass

    def collate_fun(batch):
        # Assuming pairs
        b1 = torch.cat([b[0] for b in batch], dim=0)
        b2 = torch.cat([b[1] for b in batch], dim=0)
        return b1, b2




