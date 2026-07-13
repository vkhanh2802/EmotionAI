from torchvision.datasets import ImageFolder


class AffectNetDataset(ImageFolder):

    def __init__(
        self,
        root,
        transform=None
    ):

        super().__init__(
            root=root,
            transform=transform
        )

    @property
    def num_classes(self):

        return len(self.classes)

    def get_class_names(self):

        return self.classes