import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="  [Train]", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += images.size(0)

        current_loss = running_loss / total_samples
        current_acc = (running_corrects / total_samples) * 100
        pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="  [Valid]", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Thống kê kết quả
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)
            
            # Cập nhật thông tin nhanh
            current_loss = running_loss / total_samples
            current_acc = (running_corrects / total_samples) * 100
            pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")
            
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc