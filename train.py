import argparse
import yaml
import torch
from collections import Counter

from datasets.affectnet.dataloader import build_dataloader
from models import FacialAffectModel
from losses.FocalLoss import WeightedFocalLoss
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from engines.trainer import fit

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_class_weights(dataset, class_names):
    targets = getattr(dataset, 'targets', None)
    if targets is None and hasattr(dataset, 'dataset'):
        targets = getattr(dataset.dataset, 'targets', None)

    if targets is None:
        return torch.ones(len(class_names), dtype=torch.float)

    class_counts = Counter(targets)
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]

    if any(count == 0 for count in counts):
        return torch.ones(len(class_names), dtype=torch.float)

    total_samples = sum(counts)
    class_weights = [total_samples / (len(class_names) * count) for count in counts]
    return torch.tensor(class_weights, dtype=torch.float)

def main():
    # 1. Cấu hình Argument Parser để dễ dàng đổi file config qua Terminal
    parser = argparse.ArgumentParser(description="Pipeline Huấn luyện Mô hình Facial Affect")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Đường dẫn tới file cấu hình YAML')
    args = parser.parse_args()

    # 2. Load tham số cấu hình
    cfg = load_config(args.config)
    print("[*] Đã tải cấu hình thành công từ:", args.config)

    # 3. Thiết lập Thiết bị Tính toán (Ưu tiên GPU CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị sử dụng huấn luyện: {device}")

    # 4. Khởi tạo Pipeline Dữ liệu (Dataloader)
    train_loader, val_loader, class_names = build_dataloader(
        train_dir=cfg['dataset']['train_dir'],
        test_dir=cfg['dataset']['test_dir'],
        batch_size=cfg['dataset']['batch_size'],
        num_workers=cfg['dataset']['num_workers']
    )
    print(f"[*] Tập dữ liệu chứa {len(class_names)} lớp cảm xúc: {class_names}")

    # 5. KỸ THUẬT NÂNG CAO: Tự động tính toán Class Weights để xử lý Imbalance dữ liệu
    alpha = get_class_weights(train_loader.dataset, class_names).to(device)
    print(f"[*] Đã tính toán Trọng số lớp (Alpha) cho Focal Loss: {alpha.tolist()}")

    # 6. Khởi tạo Mô hình Custom ResNet-50
    model = FacialAffectModel(num_classes=cfg['model']['num_classes'])
    model = model.to(device)

    # 7. Khởi tạo Hàm mất mát (Weighted Focal Loss)
    criterion = WeightedFocalLoss(alpha=alpha, gamma=2.0)

    # 8. Khởi tạo Bộ tối ưu hóa (Optimizer) & Lịch trình giảm LR (Scheduler)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # 9. KÍCH HOẠT VÒNG LẶP HUẤN LUYỆN CHÍNH (FIT ENGINE)
    print("\n[!] Bắt đầu quá trình huấn luyện mạng...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device
    )

if __name__ == "__main__":
    main()