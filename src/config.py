import os

# Dosya Yolları
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data_processed')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# Gerekli klasörleri oluştur
for path in [CHECKPOINT_DIR, PLOT_DIR]:
    os.makedirs(path, exist_ok=True)

# Hiperparametreler (Başlangıç değerleri, Optuna ile optimize edilebilir)
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
NUM_CLASSES = 30

# İki Aşamalı Transfer Öğrenme Parametreleri
FINE_TUNE_EPOCHS = 10         # İkinci aşama (fine-tuning) epoch sayısı
FINE_TUNE_LR_FACTOR = 0.1     # Fine-tuning aşamasında öğrenme oranı çarpanı (örn: lr * 0.1)


# Cihaz yapılandırması (GPU varsa CUDA kullan, yoksa CPU)
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
