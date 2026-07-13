import torch
import os
from tqdm import tqdm
import csv
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    
    pbar = tqdm(dataloader, desc="  [Train]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)
        
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{(running_corrects/total)*100:.2f}%")
        
    return running_loss / total, running_corrects / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0
    
    pbar = tqdm(dataloader, desc="  [Valid]", leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)
            
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{(running_corrects/total)*100:.2f}%")
            
    return running_loss / total, running_corrects / total


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device):

    epochs = cfg['training'].get('epochs', 30)

    output_dir = cfg.get('output_dir', 'outputs')
    logs_dir = os.path.join(output_dir, 'logs')
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file_path = os.path.join(logs_dir, 'training_metrics.csv')
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])
    
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        with open(log_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{train_loss:.6f}", 
                f"{train_acc:.6f}", 
                f"{val_loss:.6f}", 
                f"{val_acc:.6f}"
            ])
        saved_msg = ""

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'last_model.pth'))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_model.pth'))
            saved_msg = "[Saved Best Model]"

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        # F. In ra Terminal
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc*100:.2f}% {saved_msg}")
        
    print(f"\n[*] Huấn luyện hoàn tất! File log được lưu tại: {log_file_path}")
    print(f"[*] Độ chính xác cao nhất: {best_val_acc*100:.2f}% (Đã lưu best_model.pth)")