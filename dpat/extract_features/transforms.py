from pl_bolts.utils import _TORCHVISION_AVAILABLE
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    GaussianBlur,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


class HHGSimCLRTrainDataTransform:
    """Transforms for SimCLR during training step of the pre-training stage.

    Adapted from [1].

    Transform
    ---------
    -   `RandomResizedCrop(size=self.input_height)`
    -   `RandomHorizontalFlip()`
    -   `RandomApply([color_jitter], p=0.8)`
    -   `RandomGrayscale(p=0.2)`
    -   `RandomApply([GaussianBlur(kernel_size=int(0.1 * self.input_height))], p=0.5)`
    -   `ToTensor()`

    Example
    -------
    >>> from dpat.extract_features.transforms import HHGSimCLRTrainDataTransform
    >>> transform = SimCLRTrainDataTransform(input_height=32)
    >>> x = sample()
    >>> (xi, xj, xk) = transform(x) # xk is only for the online evaluator if used.

    References
    ----------
    [1] `pl_bolts.models.self_supervised.simclr.transforms.SimCLRTrainDataTransform`
        https://github.com/Lightning-AI/lightning-bolts
    """

    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        normalize=None,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `transforms` from `torchvision` which is not installed yet."
            )

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            RandomResizedCrop(size=self.input_height),
            RandomHorizontalFlip(p=0.5),
            RandomApply([self.color_jitter], p=0.8),
            RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(
                RandomApply([GaussianBlur(kernel_size=kernel_size)], p=0.5)
            )

        self.data_transforms = Compose(data_transforms)

        if normalize is None:
            self.final_transform = ToTensor()
        else:
            self.final_transform = Compose([ToTensor(), normalize])

        self.train_transform = Compose([self.data_transforms, self.final_transform])

        # add online train transform of the size of global view
        self.online_transform = Compose(
            [
                RandomResizedCrop(self.input_height),
                RandomHorizontalFlip(),
                self.final_transform,
            ]
        )

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, self.online_transform(sample)


class HHGSimCLREvalDataTransform(HHGSimCLRTrainDataTransform):
    """Transforms for SimCLR during the validation step of the pre-training
    stage.

    Adapted from [1].

    Transform
    ---------
    -   `Resize(input_height + 10, interpolation=3)`
    -   `CenterCrop(input_height)`
    -   `ToTensor()`

    Example
    -------
    >>> from dpat.data. import HHGSimCLREvalDataTransform
    >>> transform = SimCLREvalDataTransform(input_height=32)
    >>> x = sample()
    >>> (xi, xj, xk) = transform(x) # xk is only for the online evaluator if used

    References
    ----------
    [1] `pl_bolts.models.self_supervised.simclr.transforms.SimCLREvalDataTransform`
        https://github.com/Lightning-AI/lightning-bolts
    """

    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        normalize=None,
    ):
        super().__init__(
            normalize=normalize,
            input_height=input_height,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength,
        )

        # replace online transform with eval time transform
        self.online_transform = Compose(
            [
                Resize(int(self.input_height + 0.1 * self.input_height)),
                CenterCrop(self.input_height),
                self.final_transform,
            ]
        )
