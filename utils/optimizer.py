import torch.optim as optim

def build_optimizer(model, cfg):

    opt_name = cfg['training'].get('optimizer', 'adam').lower()
    lr = cfg['training'].get('learning_rate', 1e-4)
    weight_decay = cfg['training'].get('weight_decay', 1e-4)
    
    parameters = model.parameters()

    
    if opt_name == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        print(f"[*] Đã khởi tạo Adam Optimizer (LR: {lr})")
        
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        print(f"[*] Đã khởi tạo AdamW Optimizer (LR: {lr})")
        
    elif opt_name == 'sgd':
        momentum = cfg['training'].get('momentum', 0.9)
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        print(f"[*] Đã khởi tạo SGD Optimizer (LR: {lr}, Momentum: {momentum})")
    
    return optimizer