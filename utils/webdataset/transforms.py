"""Module with data pipe transforms.

Transforms are callables which transform a input torchdata datapipe into a new datapipe.
For further information see [ocl.transforms.Transform][].
"""
import math
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torchdata.datapipes.iter import IterDataPipe

import utils.webdataset.dataset_patches  # noqa: F401


class Transform(ABC):
    """Abstract Base Class representing a transformation of the input data pipe.

    A transform is a callable which when called with a [torchdata.datapipes.iter.IterDataPipe][]
    applies a transformation and returns a new [torchdata.datapipes.iter.IterDataPipe][].


    Attributes:
        is_batch_transform: True if the transform should be applied to a batch of
            examples instead of individual examples. False otherwise.
        fields: Tuple of strings, that indicate which elements of the input are needed
            for this transform to be applied.  This allows to avoid decoding parts of the
            input which are not needed for training/evaluating a particular model.

    """

    is_batch_transform: bool

    @property
    @abstractmethod
    def fields(self) -> Tuple[str]:
        """Fields that will be transformed with this transform."""

    @abstractmethod
    def __call__(self, input_pipe: IterDataPipe) -> IterDataPipe:
        """Application of transform to input pipe.

        Args:
            input_pipe: Input data pipe

        Returns:
            Transformed data pipe.
        """


class SimpleTransform(Transform):
    """Transform of individual key in input dict using different callables.

    Example:
        ```python
        from torchdata.datapipes.iter import IterableWrapper
        from ocl.transforms import SimpleTransform

        input_dicts = [{"object_a": 1, "object_b": 2}]
        transform = SimpleTransform(
            transforms={
                "object_a": lambda a: a*2,
                "object_b": lambda b: b*3
            }
        )

        input_pipe = IterableWrapper(input_dicts)
        transformed_pipe = transform(input_pipe)

        for transformed_dict in transformed_pipe:
            assert transformed_dict["object_a"] == 1 * 2
            assert transformed_dict["object_b"] == 2 * 3
        ```
    """

    def __init__(self, transforms: Dict[str, Callable], batch_transform: bool):
        """Initialize SimpleTransform.

        Args:
            transforms: Mapping of dict keys to callables that should be used to transform them.
            batch_transform: Set to true if you want your transform to be applied after the
                data has been batched.
        """
        self.transforms = transforms
        self.is_batch_transform = batch_transform

    @property
    def fields(self) -> Tuple[str]:
        return tuple(self.transforms.keys())

    def __call__(self, input_pipe: IterDataPipe) -> IterDataPipe:
        """Transform input data pipe using transforms.

        Args:
            input_pipe: Input data pipe

        Returns:
            Transformed data pipe.
        """
        return input_pipe.map_dict(self.transforms)


class DuplicateFields(Transform):
    """Transform to duplicate a key of a dictionary.

    This is useful if your pipeline requires the same input to be transformed in different ways.

    Example:
        ```python
        from torchdata.datapipes.iter import IterableWrapper
        from ocl.transforms import DuplicateFields

        input_dicts = [{"object_a": 1, "object_b": 2}]
        transform = DuplicateFields(
            mapping={
                "object_a": "copy_of_object_a",
                "object_b": "copy_of_object_b"
            }
        )

        input_pipe = IterableWrapper(input_dicts)
        transformed_pipe = transform(input_pipe)

        for transformed_dict in transformed_pipe:
            assert transformed_dict["object_a"] == 1
            assert transformed_dict["copy_of_object_a"] == 1
            assert transformed_dict["object_b"] == 2
            assert transformed_dict["copy_of_object_b"] == 2
        ```
    """

    def __init__(self, mapping: Dict[str, str], batch_transform: bool):
        """Initialize DuplicateFields.

        Args:
            mapping: Source to target mapping for dupplicated fields. Keys are sources,
                values are the key for duplicated field.
            batch_transform: Apply to batched input.
        """
        self.mapping = mapping
        self.is_batch_transform = batch_transform

    def duplicate_keys(self, input_dict: Dict[str, Any]):
        for key, value in self.mapping.items():
            input_dict[value] = input_dict[key]
        return input_dict

    @property
    def fields(self) -> Tuple[str]:
        return tuple(self.mapping.keys())

    def __call__(self, input_pipe: IterDataPipe) -> IterDataPipe:
        return input_pipe.map(self.duplicate_keys)


class Map(Transform):
    """Apply a function to the whole input dict to create a new output dict.

    This transform requires explicitly defining the input fields as this
    cannot be determined from the provided callable alone.

    Example:
        ```python
        from torchdata.datapipes.iter import IterableWrapper
        from ocl.transforms import Map

        input_dicts = [{"object_a": 1, "object_b": 2}]

        def combine_a_and_b(input_dict):
            output_dict = input_dict.copy()
            output_dict["combined"] = input_dict["object_a"] + input_dict["object_b"]
            return output_dict

        transform = Map(
            transform=combine_a_and_b,
            fields=("object_a", "object_b")
        )

        input_pipe = IterableWrapper(input_dicts)
        transformed_pipe = transform(input_pipe)

        for transformed_dict in transformed_pipe:
            a = transformed_dict["object_a"]
            b = transformed_dict["object_b"]
            assert transformed_dict["combined"] == a + b
        ```
    """

    def __init__(
        self,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]],
        fields: Tuple[str],
        batch_transform: bool,
    ):
        """Initialize Map transform.

        Args:
            transform: Callable which is applied to the individual input dictionaries.
            fields: The fields the transform requires to operate.
            batch_transform: Apply to batched input.
        """
        self.transfrom = transform
        self._fields = fields
        self.is_batch_transform = batch_transform

    @property
    def fields(self) -> Tuple[str]:
        return self._fields

    def __call__(self, input_pipe: IterDataPipe) -> IterDataPipe:
        return input_pipe.map(self.transfrom)


class Filter(Transform):
    """Filter samples according to predicate.

    Remove samples from input data pipe by evaluating a predicate.

    Example:
        ```python
        from torchdata.datapipes.iter import IterableWrapper
        from ocl.transforms import Filter

        input_dicts = [{"myvalue": 5}, {"myvalue": 10}]

        transform = Filter(
            predicate=lambda a: a > 5,
            fields=("myvalue",)
        )

        input_pipe = IterableWrapper(input_dicts)
        transformed_pipe = transform(input_pipe)

        for transformed_dict in transformed_pipe:
            assert transformed_dict["myvalue"] > 5
        ```


    """

    def __init__(self, predicate: Callable[..., bool], fields: Sequence[str]):
        """Transform to create a subset of a dataset by discarding samples.

        Args:
            predicate: Function which determines if elements should be kept (return value is True)
                or discarded (return value is False). The function is only provided with the fields
                specified in the `fields` parameter.
            fields (Sequence[str]): The fields from the input which should be passed on to the
                predicate for evaluation.
        """
        self.predicate = predicate
        self._fields = tuple(fields)
        self.is_batch_transform = False

    @property
    def fields(self):
        return self._fields

    def _filter_using_predicate(self, d: Dict[str, Any]):
        return self.predicate(*(d[field] for field in self._fields))

    def __call__(self, input_pipe: IterDataPipe):
        return input_pipe.filter(self._filter_using_predicate)


class SampleSlices(Transform):
    """Transform to sample slices from input tensors / numpy arrays.

    If multiple fields are provided the input tensors are assumed to be of
    same length along slicing axes and the same slices will be returned.

    Example:
        ```python
        import numpy as np
        from torchdata.datapipes.iter import IterableWrapper
        from ocl.transforms import SampleSlices

        my_array = np.random.randn(100, 10)

        input_dicts = [{"array1": my_array, "array2": my_array.copy()}]

        transform = SampleSlices(
            n_slices_per_input=5,
            fields=("array1", "array2"),
            dim=0,
            shuffle_buffer_size=1
        )

        input_pipe = IterableWrapper(input_dicts)
        transformed_pipe = transform(input_pipe)

        for transformed_dict in transformed_pipe:
            assert transformed_dict["array1"].shape == (5, 10)
            assert transformed_dict["array2"].shape == (5, 10)
            assert np.allclose(transformed_dict["array1"], transformed_dict["array2"])
        ```
    """

    def __init__(
        self,
        n_slices_per_input: int,
        fields: Sequence[str],
        dim: int = 0,
        seed: int = 39480234,
        per_epoch: bool = False,
        shuffle_buffer_size: int = 1000,
    ):
        """Initialize SampleSlices.

        Args:
            n_slices_per_input: Number of slices per input to sample. -1 indicates that all possible
                slices should be sampled.
            fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during training.
            dim: The dimension along which to slice the tensors.
            seed: Random number generator seed to deterministic sampling during evaluation.
            per_epoch: Sampling of frames over epochs, this ensures that after
                n_frames / n_frames_per_video epochs all frames have been seen at least once.
                In the case of uneven division, some frames will be seen more than once.
            shuffle_buffer_size: Size of shuffle buffer used during training. An additional
                shuffling step ensures each batch contains a diverse set of images and not only
                images from the same video.
        """
        self.n_slices_per_input = n_slices_per_input
        self._fields = tuple(fields)
        self.dim = dim
        self.seed = seed
        self.per_epoch = per_epoch
        self.shuffle_buffer_size = shuffle_buffer_size
        self.is_batch_transform = False

    def slice_data(self, data, index: int):
        """Small utility method to slice a numpy array along a specified axis."""
        n_dims_before = self.dim
        n_dims_after = data.ndim - 1 - self.dim
        slices = (slice(None),) * n_dims_before + (index,) + (slice(None),) * n_dims_after
        return data[slices]

    def sample_frames_using_key(self, data):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            # Initialize random number generator dependent on instance key. This should make the
            # sampling process deterministic, which is useful when sampling frames for the
            # validation/test data.
            key = sample["__key__"]
            # TODO (hornmax): We assume all fields to have the same size. I do not want to check
            # this here as it seems a bit verbose.
            n_frames = sample[self._fields[0]].shape[self.dim]
            frames_per_video = self.n_slices_per_input if self.n_slices_per_input != -1 else n_frames

            if self.per_epoch and self.n_slices_per_input != -1:
                n_different_epochs_per_seed = int(math.ceil(n_frames / frames_per_video))
                try:
                    epoch = int(os.environ["EPOCH"])
                except KeyError:
                    raise RuntimeError(
                        "Using SampleSlices with per_epoch=True "
                        "requires environment variable `EPOCH` to be set. "
                        "You might need to add the SetEpochEnvironmentVariable callback."
                    )
                # Only update the seed after n_frames / n_frames_per_video epochs.
                # This ensures that we get the same random order of frames until
                # we have sampled all of them.
                rand = np.random.RandomState(
                    int(key) + self.seed + (epoch // n_different_epochs_per_seed)
                )
                indices = rand.permutation(n_frames)
                selected_frames = indices[
                    epoch * self.n_slices_per_input : (epoch + 1) * self.n_slices_per_input
                ].tolist()
                if len(selected_frames) < self.n_slices_per_input:
                    # Input cannot be evenly split, take some frames from the first batch of frames.
                    n_missing = self.n_slices_per_input - len(selected_frames)
                    selected_frames.extend(indices[0:n_missing].tolist())
            else:
                rand = random.Random(int(key) + self.seed)
                selected_frames = rand.sample(range(n_frames), k=frames_per_video)

            for frame in selected_frames:
                # Slice the fields according to the frame.
                sliced_fields = {
                    field: self.slice_data(sample[field], frame) for field in self._fields
                }
                # Leave all fields besides the sliced ones as before, augment the __key__ field to
                # include the frame number.
                sliced_fields["__key__"] = f"{key}_{frame}"
                yield {**sample, **sliced_fields}

            # Delete fields to be sure we remove all references.
            for field in self.fields:
                del sample[field]

    @property
    def fields(self):
        return self._fields

    def __call__(self, input_pipe: IterDataPipe) -> IterDataPipe:
        output_pipe = input_pipe.then(self.sample_frames_using_key)
        if self.shuffle_buffer_size > 1:
            output_pipe = output_pipe.shuffle(buffer_size=self.shuffle_buffer_size)
        return output_pipe


class SplitConsecutive(Transform):
    def __init__(
        self,
        n_consecutive: int,
        fields: Sequence[str],
        dim: int = 0,
        shuffle_buffer_size: int = 1000,
        drop_last: bool = True,
    ):
        self.n_consecutive = n_consecutive
        self._fields = tuple(fields)
        self.dim = dim
        self.shuffle_buffer_size = shuffle_buffer_size
        self.drop_last = drop_last
        self.is_batch_transform = False

    @property
    def fields(self):
        return self._fields

    def split_to_consecutive_frames(self, data):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            key = sample["__key__"]
            n_frames = sample[self._fields[0]].shape[self.dim]

            splitted_fields = [
                np.array_split(
                    sample[field],
                    range(self.n_consecutive, n_frames, self.n_consecutive),
                    axis=self.dim,
                )
                for field in self._fields
            ]

            for i, slices in enumerate(zip(*splitted_fields)):
                if self.drop_last and slices[0].shape[self.dim] < self.n_consecutive:
                    # Last slice of not equally divisible input, discard.
                    continue

                sliced_fields = dict(zip(self._fields, slices))
                sliced_fields["__key__"] = f"{key}_{i}"
                yield {**sample, **sliced_fields}

    def __call__(self, input_pipe: IterDataPipe) -> IterDataPipe:
        output_pipe = input_pipe.then(self.split_to_consecutive_frames)
        if self.shuffle_buffer_size > 1:
            output_pipe = output_pipe.shuffle(buffer_size=self.shuffle_buffer_size)
        return output_pipe


class SampleConsecutive(Transform):
    """Select a random consecutive subsequence of frames in a strided manner.

    Given a sequence of [1, 2, 3, 4, 5, 6, 7, 8, 9] this will return one of
    [1, 2, 3] [4, 5, 6] [7, 8, 9].
    """

    def __init__(
        self,
        n_consecutive: int,
        fields: Sequence[str],
        dim: int = 0,
    ):
        self.n_consecutive = n_consecutive
        self._fields = tuple(fields)
        self.dim = dim
        self._random = None
        self.is_batch_transform = False

    @property
    def fields(self):
        return self._fields

    @property
    def random(self):
        if not self._random:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info:
                self._random = random.Random(worker_info.seed)
            else:
                self._random = random.Random(torch.initial_seed())

        return self._random

    def split_to_consecutive_frames(self, sample):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        key = sample["__key__"]
        n_frames = sample[self._fields[0]].shape[self.dim]

        splitted_fields = [
            np.array_split(
                sample[field],
                range(self.n_consecutive, n_frames, self.n_consecutive),
                axis=self.dim,
            )
            for field in self._fields
        ]

        n_fragments = len(splitted_fields[0])

        if len(splitted_fields[0][-1] < self.n_consecutive):
            # Discard last fragment if too short.
            n_fragments -= 1

        fragment_id = self.random.randint(0, n_fragments - 1)
        sliced_fields: Dict[str, Any] = {
            field_name: splitted_field[fragment_id]
            for field_name, splitted_field in zip(self._fields, splitted_fields)
        }
        sliced_fields["__key__"] = f"{key}_{fragment_id}"
        return {**sample, **sliced_fields}

    def __call__(self, input_pipe: IterDataPipe) -> IterDataPipe:
        return input_pipe.map(self.split_to_consecutive_frames)


class SpatialSlidingWindow(Transform):
    """Split image data spatially by sliding a window across."""

    def __init__(
        self,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int, int, int],
        fields: Sequence[str],
        expected_n_windows: Optional[int] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.expected_n_windows = expected_n_windows
        self._fields = tuple(fields)
        self.is_batch_transform = False

    @property
    def fields(self):
        return self._fields

    @staticmethod
    def pad(elem, padding):
        if elem.shape[-1] != 1 and elem.shape[-1] != 3:
            elem = elem[..., None]
        orig_height = elem.shape[-3]
        orig_width = elem.shape[-2]

        p_left, p_top, p_right, p_bottom = padding
        height = orig_height + p_top + p_bottom
        width = orig_width + p_left + p_right

        padded_shape = list(elem.shape[:-3]) + [height, width, elem.shape[-1]]
        elem_padded = np.zeros_like(elem, shape=padded_shape)
        elem_padded[..., p_top : p_top + orig_height, p_left : p_left + orig_width, :] = elem

        return elem_padded

    def sliding_window(self, data):
        for sample in data:
            key = sample["__key__"]

            window_x, window_y = self.window_size
            stride_x, stride_y = self.stride
            padded_elems = {key: self.pad(sample[key], self.padding) for key in self._fields}

            n_windows = 0
            x = 0
            y = 0
            while True:
                shape = None
                windowed_fields = {}
                for key in self._fields:
                    elem_padded = padded_elems[key]
                    if shape is None:
                        shape = elem_padded.shape
                    else:
                        if shape[-3:-1] != elem_padded.shape[-3:-1]:
                            raise ValueError("Element height, width after padding do not match")
                    windowed_fields[key] = elem_padded[..., y : y + window_y, x : x + window_x, :]

                    window_height, window_width = windowed_fields[key].shape[-3:-1]
                    assert (
                        window_y == window_height and window_x == window_width
                    ), f"Expected {window_y}, {window_x}, received {window_height}, {window_width}"

                windowed_fields["__key__"] = f"{key}_{x - self.padding[0]}_{y - self.padding[1]}"
                yield {**sample, **windowed_fields}
                n_windows += 1

                x += stride_x
                if x >= shape[-2]:
                    y += stride_y
                    x = 0
                if y >= shape[-3]:
                    break

            if self.expected_n_windows is not None and self.expected_n_windows != n_windows:
                raise ValueError(f"Expected {self.expected_n_windows} windows, but got {n_windows}")

    def __call__(self, input_pipe: IterDataPipe):
        return input_pipe.then(self.sliding_window)
