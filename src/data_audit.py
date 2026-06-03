import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


RAW_SPLITS = {
    'train': 'train',
    'valid': 'val',
    'test': 'test',
}


def _image_files(folder):
    if not os.path.isdir(folder):
        return []
    return [
        name for name in os.listdir(folder)
        if name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]


def _load_classes(data_dir):
    yaml_path = os.path.join(data_dir, 'data.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data.get('names', []), yaml_data


def _audit_raw_yolo(data_dir, classes):
    summary = {}
    class_counts = {split: Counter() for split in RAW_SPLITS}
    invalid_labels = []

    for raw_split in RAW_SPLITS:
        split_dir = os.path.join(data_dir, raw_split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        images = _image_files(images_dir)
        labels = [
            name for name in os.listdir(labels_dir)
            if os.path.isfile(os.path.join(labels_dir, name)) and name.lower().endswith('.txt')
        ] if os.path.isdir(labels_dir) else []

        image_stems = {os.path.splitext(name)[0] for name in images}
        label_stems = {os.path.splitext(name)[0] for name in labels}
        summary[raw_split] = {
            'images': len(images),
            'labels': len(labels),
            'missing_labels': len(image_stems - label_stems),
            'orphan_labels': len(label_stems - image_stems),
        }

        for label_name in labels:
            label_path = os.path.join(labels_dir, label_name)
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                        coords = [float(value) for value in parts[1:5]]
                    except ValueError:
                        invalid_labels.append((label_path, line_no, 'okunamayan deger'))
                        continue

                    if len(parts) < 5 or class_id < 0 or class_id >= len(classes):
                        invalid_labels.append((label_path, line_no, 'gecersiz sinif veya eksik kutu'))
                        continue
                    if any(value < 0 or value > 1 for value in coords):
                        invalid_labels.append((label_path, line_no, 'YOLO koordinati 0-1 disinda'))
                        continue
                    class_counts[raw_split][classes[class_id]] += 1

    return summary, class_counts, invalid_labels


def _audit_processed(processed_dir, classes):
    counts = defaultdict(Counter)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(processed_dir, split)
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            counts[split][class_name] = len(_image_files(class_dir))
    return counts


def _plot_processed_counts(processed_counts, plot_dir):
    rows = []
    for split, counts in processed_counts.items():
        for class_name, count in counts.items():
            rows.append({'Sınıf': class_name, 'Split': split, 'Kırpılmış Görüntü': count})

    df = pd.DataFrame(rows)
    order = df.groupby('Sınıf')['Kırpılmış Görüntü'].sum().sort_values(ascending=False).index

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(13, 10))
    ax = sns.barplot(
        data=df,
        y='Sınıf',
        x='Kırpılmış Görüntü',
        hue='Split',
        order=order,
        palette=['#2f80ed', '#27ae60', '#f2994a'],
        edgecolor='.2'
    )
    ax.set_title('PlantDoc Kırpılmış Görüntü Dağılımı', fontsize=16, fontweight='bold')
    ax.set_xlabel('Görüntü adedi', fontsize=12)
    ax.set_ylabel('Sınıf', fontsize=12)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, 'data_audit_processed_distribution.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def _write_report(data_dir, plot_dir, classes, yaml_data, raw_summary, raw_counts, processed_counts, invalid_labels):
    report_path = os.path.join(plot_dir, 'data_audit_report.txt')
    source = yaml_data.get('roboflow', {})

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('PlantDoc Veri Denetimi\n')
        f.write('======================\n\n')
        f.write(f"Sinif sayisi: {len(classes)}\n")
        f.write(f"data.yaml nc: {yaml_data.get('nc')}\n")
        if source:
            f.write(f"Yerel metadata kaynagi: Roboflow Universe / {source.get('project')} v{source.get('version')}\n")
            f.write(f"URL: {source.get('url')}\n")
        else:
            f.write('Yerel metadata kaynagi: data.yaml icinde Kaggle/Roboflow bilgisi bulunamadi.\n')

        f.write('\nHam YOLO split ozeti:\n')
        for split, item in raw_summary.items():
            f.write(
                f"- {split}: {item['images']} image, {item['labels']} label, "
                f"{item['missing_labels']} eksik label, {item['orphan_labels']} sahipsiz label\n"
            )

        f.write('\nEtiket kontrolu:\n')
        f.write(f"- Gecersiz/uyumsuz satir: {len(invalid_labels)}\n")
        for label_path, line_no, reason in invalid_labels[:25]:
            f.write(f"  {label_path}:{line_no} -> {reason}\n")

        f.write('\nKirilmis goruntu toplamlar:\n')
        for split in ['train', 'val', 'test']:
            total = sum(processed_counts[split].values())
            zero_classes = [name for name, count in processed_counts[split].items() if count == 0]
            f.write(f"- {split}: {total} goruntu, bos sinif sayisi: {len(zero_classes)}\n")
            if zero_classes:
                f.write(f"  Bos siniflar: {', '.join(zero_classes)}\n")

        f.write('\nRaw label toplamlar:\n')
        for split in RAW_SPLITS:
            f.write(f"- {split}: {sum(raw_counts[split].values())} kutu\n")

    return report_path


def audit_dataset(data_dir, processed_dir, plot_dir):
    classes, yaml_data = _load_classes(data_dir)
    raw_summary, raw_counts, invalid_labels = _audit_raw_yolo(data_dir, classes)
    processed_counts = _audit_processed(processed_dir, classes)
    plot_path = _plot_processed_counts(processed_counts, plot_dir)
    report_path = _write_report(
        data_dir,
        plot_dir,
        classes,
        yaml_data,
        raw_summary,
        raw_counts,
        processed_counts,
        invalid_labels
    )

    print(f"Sinif sayisi: {len(classes)}")
    for split, item in raw_summary.items():
        print(f"{split}: {item['images']} image, {item['labels']} label, eksik label: {item['missing_labels']}")
    print(f"Gecersiz etiket satiri: {len(invalid_labels)}")
    print(f"Veri denetim raporu: {report_path}")
    print(f"Veri denetim grafigi: {plot_path}")
    return report_path, plot_path


if __name__ == '__main__':
    from src import config

    audit_dataset(config.DATA_DIR, config.PROCESSED_DIR, config.PLOT_DIR)
