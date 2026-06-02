import os
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src import config

class PlantLeafDataset(Dataset):
    """
    Kırpılmış yaprak resimlerini yüklemek için özel PyTorch Dataset sınıfı.
    Doğrulama veya test setinde boş sınıflar olması durumunda tutarlı bir 
    genel sınıf indeksi haritalaması sunmak için sınıf isimlerini indekslerle eşler.
    """
    def __init__(self, split_dir, classes, transform=None):
        self.split_dir = split_dir
        self.classes = classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.transform = transform
        self.samples = []
        
        for cls_name in classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            # Resim yüklenirken hata oluşursa sıfırlardan oluşan boş bir resim döner
            print(f"Resim yüklenirken hata oluştu: {img_path}: {e}")
            import torch
            dummy_img = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            return dummy_img, label

def get_transforms(image_size=config.IMAGE_SIZE):
    """
    Eğitim, doğrulama ve test setleri için veri dönüştürme ve artırma (data augmentation) işlemlerini döner.
    Aşırı öğrenmeyi (overfitting) engellemek amacıyla eğitim setine veri artırma uygulanır.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(batch_size=config.BATCH_SIZE, image_size=config.IMAGE_SIZE):
    """
    Eğitim, doğrulama ve test DataLoader'larını oluşturur ve döner.
    """
    # Sınıf listesini data.yaml dosyasından yükle
    yaml_path = os.path.join(config.DATA_DIR, 'data.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    classes = yaml_data.get('names', [])
    
    train_dir = os.path.join(config.PROCESSED_DIR, 'train')
    val_dir = os.path.join(config.PROCESSED_DIR, 'val')
    test_dir = os.path.join(config.PROCESSED_DIR, 'test')
    
    train_transform, val_test_transform = get_transforms(image_size)
    
    # Özel PlantLeafDataset kullanarak veri setlerini yükle
    train_dataset = PlantLeafDataset(train_dir, classes, transform=train_transform)
    val_dataset = PlantLeafDataset(val_dir, classes, transform=val_test_transform)
    test_dataset = PlantLeafDataset(test_dir, classes, transform=val_test_transform)
    
    num_workers = 2 if os.name == 'nt' else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, classes

if __name__ == '__main__':
    # Veri yükleme işlemini test et ve boyutları yazdır
    print("Özel DataLoader yapılandırması test ediliyor...")
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Sınıf sayısı: {len(classes)}")
    print(f"Eğitim veri adedi: {len(train_loader.dataset)}")
    print(f"Doğrulama veri adedi: {len(val_loader.dataset)}")
    print(f"Test veri adedi: {len(test_loader.dataset)}")
    
    # Tek bir veri grubu (batch) çek
    images, labels = next(iter(train_loader))
    print(f"Batch resim boyutu: {images.shape}")  # [batch_size, 3, image_size, image_size] olmalıdır
    print(f"Batch etiket boyutu: {labels.shape}")  # [batch_size] olmalıdır
    print("Veri yükleyiciler başarıyla başlatıldı!")
