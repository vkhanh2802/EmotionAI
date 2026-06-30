from dataloader import build_dataloader

train_loader, test_loader, classes = build_dataloader(
    "train",
    "test",
    batch_size=8
)

print(classes)

images, labels = next(iter(train_loader))

print(images.shape)

print(labels)