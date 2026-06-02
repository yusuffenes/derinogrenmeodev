import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src import config

def plot_training_curves(history, output_plot_path, fine_tune_epoch=None):
    """
    Egitim ve dogrulama sureclerindeki kayip (loss) ve dogruluk (accuracy) 
    degisimlerini yan yana iki grafik halinde cizer.
    Iki asamali egitim varsa, ince ayar baslangicini dikey kesikli cizgiyle gosterir.
    """
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Sol Grafik: Kayıp (Loss) Eğrisi
    ax1.plot(epochs, history['train_loss'], label='Egitim Kaybi', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Dogrulama Kaybi', color='#ff7f0e', linewidth=2, linestyle='--')
    ax1.set_title('Egitim ve Dogrulama Kaybi (Loss)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Kayip (Loss)', fontsize=11)
    
    if fine_tune_epoch is not None and fine_tune_epoch > 0:
        ax1.axvline(x=fine_tune_epoch, color='#d62728', linestyle=':', label='Fine-Tuning Baslangici', linewidth=1.5)
        
    ax1.legend(fontsize=10)
    
    # Sağ Grafik: Doğruluk (Accuracy) Eğrisi
    ax2.plot(epochs, [x * 100 for x in history['train_acc']], label='Egitim Dogrulugu', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, [x * 100 for x in history['val_acc']], label='Dogrulama Dogrulugu', color='#d62728', linewidth=2, linestyle='--')
    ax2.set_title('Egitim ve Dogrulama Dogrulugu (Accuracy)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Dogruluk (%)', fontsize=11)
    
    if fine_tune_epoch is not None and fine_tune_epoch > 0:
        ax2.axvline(x=fine_tune_epoch, color='#d62728', linestyle=':', label='Fine-Tuning Baslangici', linewidth=1.5)
        
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Egitim grafikleri basariyla kaydedildi: {output_plot_path}")

def evaluate_model(model, test_loader, classes, checkpoint_path=None, device=config.DEVICE):
    """
    Modeli test veri seti üzerinde değerlendirir.
    Hata matrisini (confusion matrix) çizer, sınıflandırma raporunu yazdırır ve kaydeder.
    """
    if checkpoint_path:
        print(f"Model agirliklari yukleniyor: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Tahminleri topla
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Test doğruluğunu hesapla
    test_acc = np.mean(all_preds == all_labels)
    print(f"\nTest Seti Dogrulugu: {test_acc*100:.2f}%")
    
    # Sınıflandırma raporu (Classification Report)
    report = classification_report(all_labels, all_preds, labels=list(range(len(classes))), target_names=classes, zero_division=0)
    print("\nSiniflandirma Raporu:")
    print(report)
    
    # Raporu dosyaya kaydet
    report_path = os.path.join(config.PLOT_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Seti Doğruluğu: {test_acc*100:.2f}%\n\n")
        f.write(report)
    print(f"Siniflandirma raporu kaydedildi: {report_path}")
    
    # Hata Matrisi (Confusion Matrix) Çizimi
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    plt.figure(figsize=(16, 14))
    
    # Isı haritasını (heatmap) oluştur
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=True,
        square=True,
        annot_kws={"size": 8}
    )
    
    plt.title("Hata Matrisi (Confusion Matrix) - Yaprak Hastalığı Sınıflandırma", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Tahmin Edilen Sınıf", fontsize=12, fontweight='bold')
    plt.ylabel("Gerçek Sınıf", fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    cm_path = os.path.join(config.PLOT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Hata matrisi grafigi basariyla kaydedildi: {cm_path}")
    
    # Örnek Tahminler Görselleştirmesi
    plot_prediction_samples(model, test_loader, classes, device)
    
    return test_acc

def plot_prediction_samples(model, test_loader, classes, device, num_samples=12):
    """
    Test setinden rastgele resimler çeker ve üzerlerinde modelin tahminlerini gösterir.
    Doğru tahminler yeşil, yanlış tahminler kırmızı renk başlıkla çizilir.
    """
    model.eval()
    images_hand, labels_hand = next(iter(test_loader))
    
    with torch.no_grad():
        outputs = model(images_hand.to(device))
        _, preds = outputs.max(1)
        
    preds = preds.cpu().numpy()
    labels = labels_hand.numpy()
    
    # Premium görselleştirme için şekil boyutu
    plt.figure(figsize=(14, 10))
    
    # Normalizasyonu geri almak (denormalize) için ortalama ve standart sapma değerleri
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(min(num_samples, len(images_hand))):
        plt.subplot(3, 4, i + 1)
        
        # Resmi numpy formatına dönüştür ve denormalize et
        img = images_hand[i].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.axis('off')
        
        true_cls = classes[labels[i]]
        pred_cls = classes[preds[i]]
        
        # Doğru/yanlış rengini ayarla
        color = 'green' if labels[i] == preds[i] else 'red'
        
        title_str = f"Gerçek: {true_cls.split()[-1]}\nTahmin: {pred_cls.split()[-1]}"
        plt.title(title_str, color=color, fontsize=10, fontweight='bold')
        
    plt.suptitle("Test Verisi Tahmin Örnekleri", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    samples_path = os.path.join(config.PLOT_DIR, 'prediction_samples.png')
    plt.savefig(samples_path, dpi=300)
    plt.close()
    print(f"Tahmin ornekleri gorsellestirmesi kaydedildi: {samples_path}")
