import os
import optuna
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
from src.dataset import get_dataloaders
from src.model import PlantCNN, get_resnet_model
from src.train import train_one_epoch, validate

# Optuna uyarılarını azaltmak için günlük düzeyini yapılandır
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    """
    Optuna optimizasyon hedef fonksiyonu.
    Her bir deneme (trial) için hiperparametreleri seçer ve modeli eğitip
    doğrulama doğruluğunu döner.
    """
    # Optimizasyon yapılacak hiperparametre aralıklarını tanımla
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
    model_type = trial.suggest_categorical('model_type', ['custom_cnn', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_large'])
    
    # DataLoader'ları yeni batch_size ile yükle
    train_loader, val_loader, _, classes = get_dataloaders(batch_size=batch_size)
    num_classes = len(classes)
    
    # Seçilen modele göre modeli başlat
    if model_type == 'custom_cnn':
        model = PlantCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        model = get_resnet_model(num_classes=num_classes, pretrained=True, backbone=model_type)
        
    model = model.to(config.DEVICE)
    # Ağırlıklı CrossEntropyLoss kullanalım (daha kararlı optimizasyon için)
    if hasattr(train_loader.dataset, 'samples') and len(train_loader.dataset.samples) > 0:
        import numpy as np
        labels = [sample[1] for sample in train_loader.dataset.samples]
        class_counts = np.bincount(labels, minlength=num_classes)
        class_counts = np.maximum(class_counts, 1)
        weights = sum(class_counts) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(weights).to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Seçilen optimizasyon algoritmasını tanımla
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        
    # Optimizasyonu hızlandırmak için 3 epoch eğitim yapıyoruz
    epochs = 3
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # En iyi doğruluğu güncelle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        # Ara değerlendirmeleri Optuna'ya bildir (pruning/budama için)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val_acc

def plot_optuna_history(study, output_plot_path):
    """
    Deneme adımlarına göre elde edilen doğrulukları çizdiren grafik.
    """
    trials = study.trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("Çizim için tamamlanmış deneme bulunamadı.")
        return
        
    trial_numbers = [t.number + 1 for t in completed_trials]
    accuracies = [t.value * 100 for t in completed_trials]
    
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    # Denemelerin çizimi
    plt.plot(trial_numbers, accuracies, marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=8)
    
    # En iyi denemenin işaretlenmesi
    best_trial_num = study.best_trial.number + 1
    best_acc = study.best_value * 100
    plt.scatter(best_trial_num, best_acc, color='red', s=150, zorder=5, label=f'En İyi: Deneme {best_trial_num} ({best_acc:.2f}%)')
    
    plt.title("Optuna Hiperparametre Optimizasyon Geçmişi", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Deneme Numarası", fontsize=12, fontweight='bold')
    plt.ylabel("Doğrulama Doğruluğu (%)", fontsize=12, fontweight='bold')
    plt.xticks(trial_numbers)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Optuna optimizasyon geçmişi grafiği kaydedildi: {output_plot_path}")

def run_optimization(n_trials=5):
    """
    Optuna çalışmasını yönetir ve en iyi hiperparametreleri döner.
    """
    print(f"Hiperparametre optimizasyonu başlıyor ({n_trials} deneme gerçekleştirilecek)...")
    
    # Doğruluk maksimizasyonu için çalışma oluştur
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\nOptimizasyon Tamamlandı!")
    print(f"En iyi Deneme Doğruluğu: {study.best_value*100:.2f}%")
    print("En iyi Parametreler:")
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")
        
    # Geçmiş grafiğini kaydet
    plot_path = os.path.join(config.PLOT_DIR, 'optuna_optimization_history.png')
    plot_optuna_history(study, plot_path)
    
    return study.best_params

if __name__ == '__main__':
    # Hizlica dogrulamak icin 2 deneme calistir
    params = run_optimization(n_trials=2)
    print("Optimizasyon modulu basariyla dogrulandi!")

