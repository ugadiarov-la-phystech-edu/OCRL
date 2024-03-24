import gym
import wandb
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ocrs.dinosaur.dinosaur import Dinosaur
from utils.tools import *
import ocrs
import poolings


class OCRExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param rep_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, config=None):
        if config.ocr.name == "DINOSAUR":
            rep_dim = config.ocr.slotattr.slot_size
        else:
            ocr = getattr(ocrs, config.ocr.name)(config.ocr, config.env)
            pooling = getattr(poolings, config.pooling.name)(ocr, config.pooling)
            rep_dim = pooling.rep_dim
            del ocr, pooling

        super(OCRExtractor, self).__init__(observation_space, rep_dim)
        self._num_stacked_obss = config.env.num_stacked_obss
        self._obs_size = config.env.obs_size
        self._obs_channels = config.env.obs_channels
        self._num_envs = config.num_envs
        self._viz_step = 0
        self._viz_interval = config.viz_interval
        self._visualize = True
        if config.ocr.name == 'DINOSAUR':
            model_name = 'vit_base_patch16_224_dino'
            input_feature_dim = 768
            num_patches = 196
            features = (2048, 2048, 2048)
            dinosaur = Dinosaur(dino_model_name=model_name, n_slots=config.ocr.slotattr.num_slots,
                                 slot_dim=config.ocr.slotattr.slot_size,
                                 intput_feature_dim=input_feature_dim, num_patches=num_patches, features=features)
            state_dict = torch.load(config.pooling.ocr_checkpoint)['state_dict']
            state_dict = {key[len('models.'):]: value for key, value in state_dict.items()}

            dinosaur.load_state_dict(state_dict)
            dinosaur = dinosaur.eval()
            for param in dinosaur.parameters():
                param.requires_grad = False

            dinosaur.rep_dim = config.ocr.slotattr.slot_size
            dinosaur.num_slots = config.ocr.slotattr.num_slots
            self._ocr = dinosaur
            self._ocr_pretraining = True
            self._visualize = False
        else:
            self._ocr, self._ocr_pretraining = get_ocr(
                config.ocr, config.env, config.pooling.ocr_checkpoint, config.device
            )
        self._pooling = getattr(poolings, config.pooling.name + "_Module")(
            self._ocr.rep_dim, self._ocr.num_slots, config.pooling
        )

    def forward(self, observations: Tensor) -> Tensor:
        if self._ocr_pretraining and self._visualize:
            if observations.shape[0] == self._num_envs:
                if self._viz_step % self._viz_interval == 0:
                    samples = self._ocr.get_samples(observations)
                    wandb.log(
                        {k: [wandb.Image(_v) for _v in v] for k, v in samples.items()},
                    )
                self._viz_step += 1
        return self._pooling(self._ocr(observations))
