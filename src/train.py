import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src import config

def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
    """
    Modeli bir epoch boyunca eğitir.
    Eğitim kaybını (loss) ve eğitim doğruluğunu (accuracy) döner.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # İlerleme çubuğu ile veri grubu (batch) üzerinde dön
    for images, labels in tqdm(dataloader, desc="Eğitim", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # İleri besleme (forward pass)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Geri yayılım (backward pass) ve optimizasyon
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        
        # İstatistikleri hesapla
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Modeli doğrulama (validation) veri seti üzerinde test eder.
    Doğrulama kaybını (loss) ve doğruluğunu (accuracy) döner.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Gradyan hesaplamasını devre dışı bırak (hız ve bellek tasarrufu için)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Doğrulama", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def run_training(model, train_loader, val_loader, epochs=config.EPOCHS, 
                 lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                 device=config.DEVICE, save_path=None, optimizer_name='AdamW',
                 early_stopping_patience=10, use_class_weights=True, label_smoothing=0.05):
    """
    Belirtilen parametrelerle tam eğitim sürecini yürütür.
    En iyi doğrulama başarımına sahip modeli kaydeder.
    Erken durdurma (Early Stopping) mekanizması içerir.
    """
    model = model.to(device)
    
    # Çoklu sınıf sınıflandırması için CrossEntropyLoss (Sınıf Ağırlıklı veya Ağırlıksız)
    if use_class_weights and hasattr(train_loader.dataset, 'samples') and len(train_loader.dataset.samples) > 0:
        import numpy as np
        # train_loader içindeki örneklerin etiketlerini say
        labels = [sample[1] for sample in train_loader.dataset.samples]
        num_classes = len(train_loader.dataset.classes) if hasattr(train_loader.dataset, 'classes') else len(np.unique(labels))
        class_counts = np.bincount(labels, minlength=num_classes)
        class_counts = np.maximum(class_counts, 1) # 0'a bölünmeyi önlemek için
        
        # Ağırlık = Toplam Örnek / (Sınıf Sayısı * Sınıf Başına Örnek Sayısı)
        weights = sum(class_counts) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print(f" => Sinif dengesizligi icin agirlikli CrossEntropyLoss kullaniliyor.")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f" => Standart CrossEntropyLoss kullaniliyor.")
    
    # Seçilen optimizasyon algoritmasını tanımla
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Egitilebilir model parametresi bulunamadi. freeze_backbone ayarini kontrol edin.")
    
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # Doğrulama kaybına göre öğrenme oranını düşüren scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    print(f"Egitim basliyor... Cihaz: {device} | Optimizer: {optimizer_name} | Maksimum Epoch: {epochs}")
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Öğrenme oranı planlayıcıyı güncelle (Doğrulama doğruluğuna göre)
        scheduler.step(val_acc)
        
        # Geçmişe kaydet
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch}/{epochs}] -> "
              f"Egitim Kaybi: {train_loss:.4f} | Egitim Dogrulugu: {train_acc*100:.2f}% || "
              f"Dogrulama Kaybi: {val_loss:.4f} | Dogrulama Dogrulugu: {val_acc*100:.2f}%")
              
        # En iyi model ağırlıklarını kaydet (Doğrulama doğruluğuna göre)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f" => En iyi model kaydedildi! Dogrulama Dogrulugu: {val_acc*100:.2f}%")
        else:
            epochs_no_improve += 1
            
        # Erken durdurma kontrolü
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n[Erken Durdurma] Dogrulama dogrulugu {early_stopping_patience} epoch boyunca gelismedi. Egitim sonlandiriliyor.")
            break
                
    print(f"Egitim tamamlandi! En iyi Dogrulama Dogrulugu: {best_val_acc*100:.2f}%")
    return history

if __name__ == '__main__':
    # Test egitimi calistirmasi (1 Epoch)
    from src.dataset import get_dataloaders
    from src.model import PlantCNN
    
    print("Test egitimi baslatiliyor...")
    train_loader, val_loader, _, classes = get_dataloaders(batch_size=16)
    model = PlantCNN(num_classes=len(classes))
    
    test_save_path = os.path.join(config.CHECKPOINT_DIR, 'test_model.pth')
    history = run_training(
        model, 
        train_loader, 
        val_loader, 
        epochs=1, 
        lr=1e-3, 
        device=config.DEVICE,
        save_path=test_save_path
    )
    print("Egitim modulu basariyla dogrulandi!")
