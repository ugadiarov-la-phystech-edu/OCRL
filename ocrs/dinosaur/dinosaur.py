from functools import partial

import torch
import torchvision

from ocrs.dinosaur.conditioning import RandomConditioning
from ocrs.dinosaur.decoding import PatchDecoder
from ocrs.dinosaur.feature_extractors.timm import TimmFeatureExtractor
from ocrs.dinosaur.neural_networks.convenience import build_two_layer_mlp, build_mlp
from ocrs.dinosaur.neural_networks.positional_embedding import DummyPositionEmbed
from ocrs.dinosaur.neural_networks.wrappers import Sequential
from ocrs.dinosaur.perceptual_grouping import SlotAttentionGrouping


class Dinosaur(torch.nn.Module):
    def __init__(self, dino_model_name, n_slots, slot_dim, intput_feature_dim, num_patches, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dino_model_name = dino_model_name
        self._n_slots = n_slots
        self._slot_dim = slot_dim
        self._input_feature_dim = intput_feature_dim
        self._num_patches = num_patches
        self._features = features
        self._normalization = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.feature_extractor = TimmFeatureExtractor(model_name=self._dino_model_name, feature_level=12,
                                                      pretrained=True, freeze=True)
        self.conditioning = RandomConditioning(object_dim=self._slot_dim, n_slots=self._n_slots, learn_mean=True,
                                               learn_std=True)

        pos_embedding = Sequential(DummyPositionEmbed(),
                                   build_two_layer_mlp(input_dim=self._input_feature_dim, output_dim=self._slot_dim,
                                                       hidden_dim=self._input_feature_dim, initial_layer_norm=True))
        ff_mlp = build_two_layer_mlp(input_dim=self._slot_dim, output_dim=self._slot_dim, hidden_dim=4 * self._slot_dim,
                                     initial_layer_norm=True, residual=True)
        self.perceptual_grouping = SlotAttentionGrouping(feature_dim=self._slot_dim, object_dim=self._slot_dim, ff_mlp=ff_mlp,
                                                         positional_embedding=pos_embedding, use_projection_bias=False,
                                                         use_implicit_differentiation=False,
                                                         use_empty_slot_for_masked_slots=False, use_graph_gru=False)

        decoder = partial(build_mlp, features=self._features)
        self.object_decoder = PatchDecoder(object_dim=self._slot_dim, output_dim=self._input_feature_dim,
                                           num_patches=self._num_patches, decoder=decoder,)

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
