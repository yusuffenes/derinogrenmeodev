import os
import argparse
import torch
from src import config
from src.dataset import get_dataloaders
from src.model import PlantCNN, get_transfer_model
from src.train import run_training
from src.evaluate import plot_training_curves, evaluate_model

def main():
    """
    Tum CNN egitim, optimizasyon ve degerlendirme surecini koordine eden ana fonksiyon.
    """
    parser = argparse.ArgumentParser(description="Bitki Hastaligi Siniflandirma CNN Pipeline")
    
    parser.add_argument('--preprocess', action='store_true', help='Veri kumesini on islemden gecir ve kirp')
    parser.add_argument('--optimize', action='store_true', help='Optuna ile hiperparametre optimizasyonu gerceklestir')
    parser.add_argument('--trials', type=int, default=5, help='Optuna deneme sayisi (varsayilan: 5)')
    parser.add_argument('--train', action='store_true', help='Modeli egit')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['custom_cnn', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_large'], 
                        help='Egitilecek model mimarisi')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Egitim epoch sayisi')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Ogrenme orani (learning rate)')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Veri grubu boyutu (batch size)')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'], help='Optimizasyon algoritmasi')
    parser.add_argument('--freeze_backbone', action='store_true', help='Govdeyi dondur (sadece siniflandiriciyi egit)')
    parser.add_argument('--fine_tune', action='store_true', help='Iki asamali egitim gerceklestir (Ozellik Cikarimi + Ince Ayar)')
    parser.add_argument('--no_class_weights', action='store_true', help='Sinif agirliklarini kullanma (dengesiz veri seti icin)')
    parser.add_argument('--evaluate', action='store_true', help='En iyi modeli test seti uzerinde degerlendir')
    
    args = parser.parse_args()
    
    # 1. Asama: On Isleme (Preprocessing)
    if args.preprocess:
        print("\n=== [1. Asama] Veri On Isleme (Kirpma) ===")
        from src.preprocess import crop_and_save_dataset, plot_class_distribution, load_yaml
        
        yaml_path = os.path.join(config.DATA_DIR, 'data.yaml')
        yaml_data = load_yaml(yaml_path)
        classes = yaml_data.get('names', [])
        
        counts = crop_and_save_dataset(config.DATA_DIR, config.PROCESSED_DIR, classes)
        plot_path = os.path.join(config.PLOT_DIR, 'class_distribution.png')
        plot_class_distribution(counts, classes, plot_path)
        print("On isleme basariyla tamamlandi!")
        
    # 2. Asama: Hiperparametre Optimizasyonu (Optuna)
    best_params = {}
    if args.optimize:
        print(f"\n=== [2. Asama] Hiperparametre Optimizasyonu ({args.trials} Deneme) ===")
        from src.optimize import run_optimization
        best_params = run_optimization(n_trials=args.trials)
        
        # En iyi parametreleri konfigurasyona aktar
        args.lr = best_params.get('lr', args.lr)
        args.batch_size = best_params.get('batch_size', args.batch_size)
        args.model = best_params.get('model_type', args.model)
        args.optimizer = best_params.get('optimizer', args.optimizer)
        print(f"Optimizasyondan secilen en iyi model: {args.model}, Batch Size: {args.batch_size}, LR: {args.lr:.6f}, Optimizer: {args.optimizer}")
        
    # 3. Asama: Model Egitimi (Training)
    if args.train:
        print(f"\n=== [3. Asama] Model Egitimi ({args.model.upper()}) ===")
        train_loader, val_loader, _, classes = get_dataloaders(batch_size=args.batch_size)
        num_classes = len(classes)
        
        # Sinif agirliklari kullanimi
        use_class_weights = not args.no_class_weights
        
        # Secilen modeli yukle ve egit
        if args.model == 'custom_cnn':
            model = PlantCNN(num_classes=num_classes, dropout_rate=config.DROPOUT)
            save_path = os.path.join(config.CHECKPOINT_DIR, f'best_{args.model}.pth')
            
            history = run_training(
                model, 
                train_loader, 
                val_loader, 
                epochs=args.epochs, 
                lr=args.lr, 
                device=config.DEVICE,
                save_path=save_path,
                optimizer_name=args.optimizer,
                early_stopping_patience=10,
                use_class_weights=use_class_weights
            )
        else:
            # Transfer Ogrenme Modeli
            if args.fine_tune:
                print("\n>>> ASAMA 1: Ozellik Cikarimi (Feature Extraction) - Govde Donduruldu")
                model = get_transfer_model(num_classes=num_classes, pretrained=True, freeze_backbone=True, backbone=args.model)
                save_path_stage1 = os.path.join(config.CHECKPOINT_DIR, f'best_{args.model}_stage1.pth')
                
                # Siniflandirici egitimi icin 5 epoch yeterlidir
                stage1_epochs = max(3, args.epochs // 3)
                history_stage1 = run_training(
                    model, 
                    train_loader, 
                    val_loader, 
                    epochs=stage1_epochs, 
                    lr=args.lr, 
                    device=config.DEVICE,
                    save_path=save_path_stage1,
                    optimizer_name=args.optimizer,
                    early_stopping_patience=5,
                    use_class_weights=use_class_weights
                )
                
                print("\n>>> ASAMA 2: Ince Ayar (Fine-Tuning) - Tum Katmanlar Cozuldu")
                # 1. Asama en iyi agirliklarini yukle
                if os.path.exists(save_path_stage1):
                    model.load_state_dict(torch.load(save_path_stage1, map_location=config.DEVICE))
                
                # Tum parametreleri egitilebilir yap (unfreeze)
                for param in model.parameters():
                    param.requires_grad = True
                    
                # Daha dusuk bir ogrenme orani hesapla
                fine_tune_lr = args.lr * config.FINE_TUNE_LR_FACTOR
                save_path_final = os.path.join(config.CHECKPOINT_DIR, f'best_{args.model}.pth')
                
                history = run_training(
                    model, 
                    train_loader, 
                    val_loader, 
                    epochs=config.FINE_TUNE_EPOCHS, 
                    lr=fine_tune_lr, 
                    device=config.DEVICE,
                    save_path=save_path_final,
                    optimizer_name=args.optimizer,
                    early_stopping_patience=8,
                    use_class_weights=use_class_weights
                )
                
                # Iki asamali egitim gecmisini birlestirerek tek bir grafikte goster
                for key in history.keys():
                    history[key] = history_stage1[key] + history[key]
            else:
                # Normal tek asamali egitim (kullanici isterse --freeze_backbone gecebilir)
                freeze = args.freeze_backbone
                print(f"Tek Asamali Egitim | Govde Dondurulmus mu: {freeze}")
                model = get_transfer_model(num_classes=num_classes, pretrained=True, freeze_backbone=freeze, backbone=args.model)
                save_path = os.path.join(config.CHECKPOINT_DIR, f'best_{args.model}.pth')
                
                history = run_training(
                    model, 
                    train_loader, 
                    val_loader, 
                    epochs=args.epochs, 
                    lr=args.lr, 
                    device=config.DEVICE,
                    save_path=save_path,
                    optimizer_name=args.optimizer,
                    early_stopping_patience=10,
                    use_class_weights=use_class_weights
                )
        
        # Egitim grafiklerini kaydet (ince ayar baslangicini dikey cizgiyle isaretle)
        curves_path = os.path.join(config.PLOT_DIR, f'training_curves_{args.model}.png')
        if args.model != 'custom_cnn' and args.fine_tune:
            plot_training_curves(history, curves_path, fine_tune_epoch=stage1_epochs)
        else:
            plot_training_curves(history, curves_path)
        print("Model egitimi ve grafik cizimleri tamamlandi!")
        
    # 4. Asama: Test Verisi Degerlendirme (Evaluation)
    if args.evaluate:
        print(f"\n=== [4. Asama] Test Seti Degerlendirmesi ===")
        # En guncel batch boyutuyla test loader'i getir
        _, _, test_loader, classes = get_dataloaders(batch_size=args.batch_size)
        num_classes = len(classes)
        
        if args.model == 'custom_cnn':
            model = PlantCNN(num_classes=num_classes)
        else:
            model = get_transfer_model(num_classes=num_classes, pretrained=False, backbone=args.model)
            
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'best_{args.model}.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Hata: Degerlendirilecek egitilmis model dosyasi bulunamadi: {checkpoint_path}")
            print("Lutfen once --train parametresiyle egitim yapin.")
            return
            
        # Degerlendirmeyi calistir
        evaluate_model(model, test_loader, classes, checkpoint_path=checkpoint_path, device=config.DEVICE)
        print("Degerlendirme islemleri basariyla tamamlandi!")

if __name__ == '__main__':
    main()
