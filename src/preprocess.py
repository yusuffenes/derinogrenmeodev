import os
import yaml
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def crop_and_save_dataset(data_dir, output_dir, classes):
    """
    YOLO biçimindeki etiketleri okur, sınırlayıcı kutuları (bounding box) kırpar
    ve elde edilen yaprak resimlerini kendi sınıflarına göre klasörlere kaydeder.
    """
    splits = ['train', 'valid', 'test']
    
    # Dağılımları takip et
    class_counts = {split: {cls_name: 0 for cls_name in classes} for split in splits}
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        # Hedef klasör isimlendirmesi (valid -> val)
        out_split = 'val' if split == 'valid' else split
        split_out_dir = os.path.join(output_dir, out_split)
        
        # Klasör yapısını oluştur
        for cls_name in classes:
            os.makedirs(os.path.join(split_out_dir, cls_name), exist_ok=True)
            
        if not os.path.exists(images_dir):
            print(f"{split} seti atlanıyor: {images_dir} klasörü bulunamadı.")
            continue
            
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{split} seti işleniyor ({len(image_files)} resim)...")
        
        for img_name in tqdm(image_files):
            img_path = os.path.join(images_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, base_name + '.txt')
            
            if not os.path.exists(label_path):
                # Resme ait etiket dosyası yoksa atla
                continue
                
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    
                    with open(label_path, 'r') as lf:
                        lines = lf.readlines()
                        
                    for idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        if class_id >= len(classes) or class_id < 0:
                            print(f"Uyarı: {label_path} dosyasında sınır dışı class_id {class_id} bulundu.")
                            continue
                            
                        cls_name = classes[class_id]
                        
                        # YOLO normalize koordinatları piksel koordinatlarına dönüştür
                        x_center_px = x_center * img_w
                        y_center_px = y_center * img_h
                        w_px = width * img_w
                        h_px = height * img_h
                        
                        xmin = max(0, int(x_center_px - w_px / 2))
                        ymin = max(0, int(y_center_px - h_px / 2))
                        xmax = min(img_w, int(x_center_px + w_px / 2))
                        ymax = min(img_h, int(x_center_px + h_px / 2))
                        
                        # Çok küçük veya geçersiz kırpıntıları atla (gürültüyü önlemek için)
                        if (xmax - xmin) < 15 or (ymax - ymin) < 15:
                            continue
                            
                        # Kırpma işlemini yap
                        crop = img.crop((xmin, ymin, xmax, ymax))
                        
                        # Resmi kaydet
                        crop_name = f"{base_name}_crop_{idx}.jpg"
                        crop_save_path = os.path.join(split_out_dir, cls_name, crop_name)
                        crop.convert('RGB').save(crop_save_path, 'JPEG', quality=95)
                        
                        # Metrikleri kaydet
                        class_counts[split][cls_name] += 1
                        
            except Exception as e:
                print(f"Resim işlenirken hata oluştu: {img_name}: {e}")
                
    return class_counts

def plot_class_distribution(class_counts, classes, output_plot_path):
    """
    Kırpılmış yaprak resimlerinin sınıflara göre dağılımını gösteren şık bir yatay bar grafik çizer.
    """
    import pandas as pd
    
    # DataFrame için veriyi hazırla
    data_list = []
    for split, counts in class_counts.items():
        split_name = 'Validation' if split == 'valid' else split
        for cls_name, count in counts.items():
            data_list.append({
                'Sınıf Adı': cls_name,
                'Kırpılmış Yaprak Sayısı': count,
                'Veri Seti': split_name.capitalize()
            })
            
    df = pd.DataFrame(data_list)
    
    # Grafik temasını ayarla
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 10))
    
    # Toplam adede göre sıralama yapmak için sınıf sıralamasını belirle
    class_order = df.groupby('Sınıf Adı')['Kırpılmış Yaprak Sayısı'].sum().sort_values(ascending=False).index
    
    # Yatay bar grafik çiz
    palette = sns.color_palette("viridis", 3)
    ax = sns.barplot(
        data=df,
        y='Sınıf Adı',
        x='Kırpılmış Yaprak Sayısı',
        hue='Veri Seti',
        order=class_order,
        palette=palette,
        edgecolor=".2"
    )
    
    plt.title("Yaprak Hastalığı Veri Seti Sınıf Dağılımı (Kırpılmış Yapraklar)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Yaprak Resim Adedi", fontsize=12, fontweight='bold')
    plt.ylabel("Sınıf Adı", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Dağılım grafiği başarıyla kaydedildi: {output_plot_path}")

if __name__ == '__main__':
    data_dir = os.path.abspath('data')
    yaml_path = os.path.join(data_dir, 'data.yaml')
    output_dir = os.path.abspath('data_processed')
    plot_path = os.path.abspath(os.path.join('plots', 'class_distribution.png'))
    
    print("On isleme baslatiliyor...")
    if not os.path.exists(yaml_path):
        print(f"Hata: {yaml_path} bulunamadi. Veri setinin 'data/' klasoru altinda oldugundan emin olun.")
        exit(1)
        
    yaml_data = load_yaml(yaml_path)
    classes = yaml_data.get('names', [])
    print(f"data.yaml dosyasinda {len(classes)} sinif tespit edildi:")
    print(classes)
    
    # Görüntüleri kırp
    counts = crop_and_save_dataset(data_dir, output_dir, classes)
    
    # Özet bilgileri yazdır
    print("\nOn isleme ozeti (kirpilan yaprak adetleri):")
    for split, split_counts in counts.items():
        total_crops = sum(split_counts.values())
        print(f" - {split} seti: {total_crops} adet kirpinti olusturuldu.")
        
    # Dağılım grafiğini çiz
    plot_class_distribution(counts, classes, plot_path)
    print("On isleme basariyla tamamlandi!")
