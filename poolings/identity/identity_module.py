from torch import nn

from utils.tools import *


class Identity_Module(nn.Module):
    def __init__(self, ocr_rep_dim: int, ocr_num_slots: int, config: dict, num_stacked_obss: int=1) -> None:
        super(Identity_Module, self).__init__()
        self.flatten = config.flatten is True
        self.rep_dim = ocr_rep_dim * ocr_num_slots * num_stacked_obss

    def forward(self, state: Tensor) -> Tensor:
        if self.flatten:
            return state.flatten(start_dim=1) if len(state.shape) == 3 else state

        return state
