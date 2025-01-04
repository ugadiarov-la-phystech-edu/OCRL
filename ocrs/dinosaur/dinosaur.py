from functools import partial

import torch
import torchvision

from ocrs.base import Base

from ocrs.dinosaur.conditioning import RandomConditioning
from ocrs.dinosaur.decoding import PatchDecoder
from ocrs.dinosaur.feature_extractors.timm import TimmFeatureExtractor
from ocrs.dinosaur.neural_networks.convenience import build_two_layer_mlp, build_mlp
from ocrs.dinosaur.neural_networks.positional_embedding import DummyPositionEmbed
from ocrs.dinosaur.neural_networks.wrappers import Sequential
from ocrs.dinosaur.perceptual_grouping import SlotAttentionGrouping


class Dinosaur(Base):
    def __init__(self, ocr_config, env_config):
        self._module = Dinosaur_Module(ocr_config, env_config)
        super().__init__(ocr_config, env_config)

    def load(self, checkpoint: str) -> None:
        state_dict = {key[len('models.'):]: value for key, value in checkpoint['state_dict'].items()}
        self._module.load_state_dict(state_dict)


class Dinosaur_Module(torch.nn.Module):
    def __init__(self, ocr_config, env_config):
        super().__init__()
        self._dino_model_name = ocr_config['model_name']
        self.num_slots = ocr_config['num_slots']
        self.rep_dim = ocr_config['slot_size']
        self._input_feature_dim = ocr_config['input_feature_dim']
        self._num_patches = ocr_config['num_patches']
        self._features = ocr_config['features']
        self._normalization = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.feature_extractor = TimmFeatureExtractor(model_name=self._dino_model_name, feature_level=12,
                                                      pretrained=True, freeze=True)
        self.conditioning = RandomConditioning(object_dim=self.rep_dim, n_slots=self.num_slots, learn_mean=True,
                                               learn_std=True)

        pos_embedding = Sequential(DummyPositionEmbed(),
                                   build_two_layer_mlp(input_dim=self._input_feature_dim, output_dim=self.rep_dim,
                                                       hidden_dim=self._input_feature_dim, initial_layer_norm=True))
        ff_mlp = build_two_layer_mlp(input_dim=self.rep_dim, output_dim=self.rep_dim, hidden_dim=4 * self.rep_dim,
                                     initial_layer_norm=True, residual=True)
        self.perceptual_grouping = SlotAttentionGrouping(feature_dim=self.rep_dim, object_dim=self.rep_dim, ff_mlp=ff_mlp,
                                                         positional_embedding=pos_embedding, use_projection_bias=False,
                                                         use_implicit_differentiation=False,
                                                         use_empty_slot_for_masked_slots=False, use_graph_gru=False)

        decoder = partial(build_mlp, features=self._features)
        self.object_decoder = PatchDecoder(object_dim=self.rep_dim, output_dim=self._input_feature_dim,
                                           num_patches=self._num_patches, decoder=decoder, )

    def forward(self, image, prev_slots=None):
        image = self._normalization(image)
        feature_extraction_output = self.feature_extractor(image)
        conditioning_output = prev_slots
        if conditioning_output is None:
            conditioning_output = self.conditioning(feature_extraction_output.features.size()[0])

        perceptual_grouping_output = self.perceptual_grouping(feature_extraction_output, conditioning_output)
        # patch_reconstruction_output = self._patch_decoder(perceptual_grouping_output.objects,
        #                                                   feature_extraction_output.features, image)

        return perceptual_grouping_output.objects.detach()
