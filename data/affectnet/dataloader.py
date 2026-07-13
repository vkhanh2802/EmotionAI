from torch.utils.data import DataLoader

from data.affectnet.imagefolder import AffectNetDataset

from data.affectnet.transforms import (
    train_transform,
    test_transform
)


def build_dataloader(
    train_dir,
    test_dir,
    batch_size=32,
    num_workers=4
):

    train_dataset = AffectNetDataset(
        train_dir,
        transform=train_transform
    )

    test_dataset = AffectNetDataset(
        test_dir,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        test_loader,
        train_dataset.classes
    )