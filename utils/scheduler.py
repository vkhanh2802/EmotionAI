import torch.optim.lr_scheduler as lr_scheduler

def build_scheduler(optimizer, cfg):
    sched_name = cfg['training'].get('scheduler', 'steplr').lower()
    
    if sched_name == 'steplr':
        step_size = cfg['training'].get('step_size', 10)
        gamma = cfg['training'].get('gamma', 0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"[*] Đã khởi tạo StepLR Scheduler (Drop mỗi {step_size} epochs, Gamma: {gamma})")
        
    elif sched_name == 'cosine':
        epochs = cfg['training'].get('epochs', 30)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        print(f"[*] Đã khởi tạo CosineAnnealingLR Scheduler (T_max: {epochs})")
        
    elif sched_name == 'plateau':
        patience = cfg['training'].get('patience', 3)
        factor = cfg['training'].get('factor', 0.5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
        print(f"[*] Đã khởi tạo ReduceLROnPlateau Scheduler (Patience: {patience}, Factor: {factor})")
        
    elif sched_name == 'none':
        scheduler = None
        print("[-] Không sử dụng LR Scheduler.")
        
    else:
        raise ValueError(f"[!] Lỗi: Scheduler '{sched_name}' chưa được hỗ trợ.")

    return scheduler