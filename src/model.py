import torch
import torch.nn as nn
import torchvision.models as models

class PlantCNN(nn.Module):
    """
    Yaprak sınıflandırması için geliştirilmiş özel Evrişimsel Sinir Ağı (CNN) mimarisi.
    4 evrişim bloğu (Conv + BatchNorm + ReLU + MaxPool) ve ardışık olarak dropout içeren
    tam bağlantılı (fully connected) katmanlardan oluşur.
    """
    def __init__(self, num_classes=30, dropout_rate=0.3):
        super(PlantCNN, self).__init__()
        
        # Blok 1: Girdi (3, 224, 224) -> (32, 112, 112)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Blok 2: (32, 112, 112) -> (64, 56, 56)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Blok 3: (64, 56, 56) -> (128, 28, 28)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Blok 4: (128, 28, 28) -> (256, 14, 14)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Değişken girdi boyutlarını sabitlemek için adaptif ortalama havuzlama
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Sınıflandırıcı Başlık
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.avgpool(x)
        x = self.fc(x)
        return x

def get_transfer_model(num_classes=30, pretrained=True, freeze_backbone=False, backbone='resnet18'):
    """
    Önceden eğitilmiş ağırlıklara sahip transfer öğrenme modellerini yükler.
    Desteklenen modeller: resnet18, resnet34, resnet50, efficientnet_b0, mobilenet_v3_large
    Sınıflandırıcı katmanı güncellenir.
    """
    if backbone == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif backbone == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif backbone == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif backbone == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif backbone == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Desteklenmeyen backbone tipi: {backbone}")

    # Backbone dondurma işlemi (Sadece en son sınıflandırıcı katman hariç dondurulur)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
        # Son katmanların requires_grad parametresini True yapalım ki eğitilebilsinler
        if backbone in ['resnet18', 'resnet34', 'resnet50']:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif backbone == 'efficientnet_b0':
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif backbone == 'mobilenet_v3_large':
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model

# Eski çağrılarla geriye dönük uyumluluk için alias
def get_resnet_model(num_classes=30, pretrained=True, freeze_backbone=False, backbone='resnet18'):
    return get_transfer_model(num_classes, pretrained, freeze_backbone, backbone)

if __name__ == '__main__':
    # Modelleri test et
    print("Ozel CNN modeli test ediliyor...")
    custom_model = PlantCNN(num_classes=30)
    x = torch.randn(2, 3, 224, 224)
    out_custom = custom_model(x)
    print(f"Ozel CNN Cikti Boyutu: {out_custom.shape}")  # [2, 30] olmalidir
    
    backbones = ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_large']
    for bb in backbones:
        print(f"\n{bb} baseline modeli test ediliyor (freeze_backbone=True)...")
        model = get_transfer_model(num_classes=30, pretrained=True, freeze_backbone=True, backbone=bb)
        out = model(x)
        print(f"{bb} Cikti Boyutu: {out.shape}")  # [2, 30] olmalidir
        
        # Dondurma kontrolü
        grad_params = sum(p.requires_grad for p in model.parameters() if p.requires_grad)
        print(f"{bb} egitilebilir parametre grubu sayisi: {grad_params}")
        
    print("\nTum modeller basariyla dogrulandi!")


