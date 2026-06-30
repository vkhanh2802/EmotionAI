import torch.nn as nn

class EmotionHead(nn.Module):
    def __init__(self, in_features=2048, hidden_features=512, num_classes=7, dropout_rate=0.5):
        super(EmotionHead, self).__init__()
        
        self.head = nn.Sequential(

            nn.Linear(in_features, hidden_features),

            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_features, num_classes)
        )

    def forward(self, x):
        return self.head(x)

# TODO (Giai đoạn Multi-task Learning): 
# class VAHead(nn.Module): ...
# class AUHead(nn.Module): ...