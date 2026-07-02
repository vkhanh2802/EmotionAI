import os
import argparse
import yaml
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from datasets.affectnet.dataloader import build_dataloader
from models import FacialAffectModel

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Đánh giá Chuyên sâu Mô hình")
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--weights', type=str, default='outputs/checkpoints/best_model.pth', help='Đường dẫn file .pth tốt nhất')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị sử dụng đánh giá: {device}")

    # 1. Load Dataloader tập Test/Val
    _, test_loader, class_names = build_dataloader(
        train_dir=cfg['dataset']['train_dir'],
        test_dir=cfg['dataset']['test_dir'],
        batch_size=cfg['dataset']['batch_size'],
        num_workers=cfg['dataset']['num_workers']
    )

    # 2. Khởi tạo mô hình và tải Trọng số đã huấn luyện (Weights)
    model = FacialAffectModel(num_classes=cfg['model']['num_classes'])
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"[!] Lỗi: Không tìm thấy file trọng số tại {args.weights}. Bạn cần chạy train.py trước!")
        
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"[*] Đã tải thành công trọng số tốt nhất từ: {args.weights}")

    all_preds = []
    all_labels = []

    print("[*] Đang chạy suy luận trên toàn bộ tập dữ liệu kiểm thử...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Đẩy kết quả về CPU để xử lý mảng Numpy
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. TÍNH TOÁN CÁC CHỈ SỐ PHÂN TÍCH CHUYÊN SÂU (DÀNH CHO CV)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # A. Báo cáo phân loại chi tiết (Classification Report)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    # B. Ma trận nhầm lẫn (Confusion Matrix)
    matrix = confusion_matrix(all_labels, all_preds)

    # 4. IN VÀ LƯU KẾT QUẢ BÁO CÁO
    print("\n" + "="*20 + " KẾT QUẢ ĐÁNH GIÁ CHI TIẾT " + "="*20)
    print(report)
    print("="*67)
    
    print("\n[*] Ma trận nhầm lẫn (Confusion Matrix):")
    print(matrix)

    # Lưu kết quả ra file txt trong thư mục logs để sau này đưa vào tài liệu GitHub
    eval_log_path = os.path.join(cfg.get('output_dir', 'outputs'), 'logs', 'evaluation_report.txt')
    with open(eval_log_path, 'w', encoding='utf-8') as f:
        f.write("=== BAO CAO DANH GIA MO HINH CAO CAP ===\n\n")
        f.write(report)
        f.write("\n\n=== MA TRAN NHAM LAN (CONFUSION MATRIX) ===\n")
        f.write(str(matrix))
        
    print(f"\n[+] Báo cáo chi tiết đã được xuất ra file thành công tại: {eval_log_path}")

if __name__ == "__main__":
    main()