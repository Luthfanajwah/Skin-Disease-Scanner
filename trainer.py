#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Klasifikasi Penyakit Kulit Wajah dengan MobileNetV2
Optimized for Face Skin Diseases Dataset
"""

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from tqdm import tqdm
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ======================== KONFIGURASI PATH ========================
# BASE_DIR adalah folder tempat trainer.py berada (CompVis)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Path spesifik untuk dataset Mendeley
MENDELEY_DIR = os.path.join(DATA_DIR, 'Skin Disease Classification Dataset', 'setdata_ klasifikasi_penyakit_kulit')

# Path spesifik untuk dataset Face Skin Diseases
FACE_TRAIN_DIR = os.path.join(DATA_DIR, 'Face Skin Diseases', 'train')
FACE_TEST_DIR  = os.path.join(DATA_DIR, 'Face Skin Diseases', 'testing')


# Konfigurasi hardware
USE_GPU = True                     # Aktifkan GPU jika tersedia (lebih cepat)
CPU_THREADS = 8                    # Sesuaikan dengan CPU-mu
BATCH_SIZE = 32                    # Naikkan jika RAM cukup (16 GB+)
IMG_SIZE = (300, 300)              # Ukuran input MobileNetV2 (asli 224)
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Atur GPU/CPU
if not USE_GPU:
    tf.config.set_visible_devices([], 'GPU')
    print("GPU dinonaktifkan, menggunakan CPU.")
else:
    # Aktifkan memory growth untuk menghindari alokasi penuh
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU ditemukan: {gpus}")
    else:
        print("Tidak ada GPU, menggunakan CPU.")

tf.config.threading.set_inter_op_parallelism_threads(CPU_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)

# Timestamp untuk folder hasil
now = datetime.now()
timestamp = now.strftime("%H%M_%d_%m_%Y")

base_fig_dir = os.path.join(BASE_DIR, 'figures', f'hasil_{timestamp}')
base_report_dir = os.path.join(BASE_DIR, 'laporan', f'hasil_{timestamp}')
os.makedirs(base_fig_dir, exist_ok=True)
os.makedirs(base_report_dir, exist_ok=True)

# Subfolder untuk dataset
dataset_sub = 'FSD'  # Face Skin Diseases
normal_dir = os.path.join(base_fig_dir, dataset_sub, 'normal')
fine_dir = os.path.join(base_fig_dir, dataset_sub, 'fine_tuned')
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(fine_dir, exist_ok=True)

# ======================== FUNGSI UTILITAS ========================
def count_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Hitung file gambar dalam direktori (filter ekstensi)."""
    return len([f for f in directory.iterdir() 
                if f.is_file() and f.suffix.lower() in extensions])

def save_training_plot(history, save_path):
    """Plot kurva akurasi dan loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Buat confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_classification_report_csv(report_dict, class_names, accuracy, save_path):
    """Simpan classification report ke CSV."""
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'precision', 'recall', 'f1-score', 'support'])
        for class_name in class_names:
            metrics = report_dict[class_name]
            writer.writerow([class_name, metrics['precision'], metrics['recall'],
                             metrics['f1-score'], metrics['support']])
        writer.writerow([])
        writer.writerow(['accuracy', '', '', '', accuracy])
        macro = report_dict['macro avg']
        writer.writerow(['macro avg', macro['precision'], macro['recall'],
                         macro['f1-score'], macro['support']])
        weighted = report_dict['weighted avg']
        writer.writerow(['weighted avg', weighted['precision'], weighted['recall'],
                         weighted['f1-score'], weighted['support']])

# ======================== GRAD‑CAM (disederhanakan) ========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Menghasilkan heatmap Grad‑CAM."""
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def save_gradcam_samples(model, test_ds, class_names, save_dir, num_samples=5):
    """
    Simpan sampel Grad‑CAM menggunakan base_model dan rekonstruksi head manual.
    """
    # Ambil base_model dari model utama (asumsi nama layer 'mobilenetv2_1.00_224')
    base_model = model.get_layer('mobilenetv2_1.00_224')

    # Cari layer Conv2D terakhir di base_model (seharusnya 'Conv_1')
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        print("Tidak dapat menemukan layer Conv2D terakhir, Grad‑CAM dilewati.")
        return []

    # Ambil layer classifier head dari model utama
    gap_layer = model.get_layer('global_average_pooling2d')
    dropout_layer = model.get_layer('dropout')
    dense_layer = model.get_layer('dense')

    # Ambil beberapa sampel gambar dari test_ds
    sample_images, sample_labels = next(iter(test_ds.take(1)))
    sample_images = sample_images[:num_samples]

    saved_paths = []
    for i in range(len(sample_images)):
        img_tensor = sample_images[i]
        img_array = tf.expand_dims(tf.cast(img_tensor, tf.float32), axis=0)

        # Preprocess input sesuai MobileNetV2
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Buat feature extractor dari base_model.input ke [last_conv_layer.output, base_model.output]
        feature_extractor = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, pooled_features = feature_extractor(img_preprocessed)
            # Rekonstruksi classifier head
            x = gap_layer(conv_outputs)
            x = dropout_layer(x)
            predictions = dense_layer(x)

            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        # Resize heatmap ke ukuran gambar asli
        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE)
        heatmap_resized = tf.squeeze(heatmap_resized).numpy()

        img_np = img_tensor.numpy().astype("uint8")
        heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3] * 255
        superimposed = img_np * 0.6 + heatmap_colored * 0.4
        superimposed = np.clip(superimposed, 0, 255).astype("uint8")

        # Plot tiga panel
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        true_idx = np.argmax(sample_labels[i].numpy())
        plt.title(f"True: {class_names[true_idx]}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.8)
        plt.title("Grad‑CAM Heatmap")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(superimposed)
        plt.title("Overlay")
        plt.axis('off')

        save_path = os.path.join(save_dir, f'sample_{i+1}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Grad‑CAM sample {i+1} saved to {save_path}")
        saved_paths.append(save_path)

    return saved_paths

# ======================== MEMBANGUN MODEL ========================
def build_model(num_classes):
    """MobileNetV2 dengan classifier head."""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # UPDATED: Dropout lebih besar untuk regularisasi
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

# ======================== MAIN ========================
def main():
    print("=" * 50)
    print("KLASIFIKASI PENYAKIT KULIT DENGAN MOBILENETV2")
    print("=" * 50)
    
    # Pilih dataset
    print("\nPilih dataset yang akan digunakan:")
    print("  1. Face Skin Diseases (FSD) - train/test terpisah")
    print("  2. Mendeley Skin Disease (SDCD) - satu folder dengan split otomatis")
    choice_dataset = input("Masukkan pilihan (1/2): ").strip()
    
    if choice_dataset == '1':
        dataset_choice = 'FSD'
        dataset_sub = 'FSD'
        train_dir = pathlib.Path(FACE_TRAIN_DIR)
        test_dir = pathlib.Path(FACE_TEST_DIR)
        print(f"\n>>> Dataset: Face Skin Diseases ({dataset_sub})")
    elif choice_dataset == '2':
        dataset_choice = 'Mendeley'
        dataset_sub = 'SDCD'
        train_dir = pathlib.Path(MENDELEY_DIR)
        test_dir = None   # Mendeley tidak punya folder test terpisah
        print(f"\n>>> Dataset: Mendeley Skin Disease ({dataset_sub})")
    else:
        raise ValueError("Pilihan tidak valid. Harus 1 atau 2.")
    
    # Buat subfolder untuk dataset di figures
    normal_dir = os.path.join(base_fig_dir, dataset_sub, 'normal')
    fine_dir = os.path.join(base_fig_dir, dataset_sub, 'fine_tuned')
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(fine_dir, exist_ok=True)
    
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Ukuran input: {IMG_SIZE}")
    print("=" * 50)

    # Validasi path
    if not train_dir.exists():
        raise FileNotFoundError(f"Folder train tidak ditemukan: {train_dir}")
    if dataset_choice == 'FSD' and (not test_dir or not test_dir.exists()):
        raise FileNotFoundError(f"Folder test tidak ditemukan: {test_dir}")

    # Tampilkan kelas yang ditemukan
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if dataset_choice == 'FSD':
        test_classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
        if train_classes != test_classes:
            raise ValueError(f"Nama folder train dan test tidak sama!\nTrain: {train_classes}\nTest: {test_classes}")
    print("Kelas ditemukan:", train_classes)

    # ======================== LOAD DATASET ========================
    if dataset_choice == 'FSD':
        # Face Skin Diseases: train/test terpisah
        train_ds = image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="training",
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        val_ds = image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="validation",
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        test_ds = image_dataset_from_directory(
            test_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=False
        )
    else:
        # Mendeley: satu folder, split 70/15/15
        full_train_ds = image_dataset_from_directory(
            train_dir,
            validation_split=0.3,   # 30% untuk val+test
            subset="training",
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        val_test_ds = image_dataset_from_directory(
            train_dir,
            validation_split=0.3,
            subset="validation",
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        
        # Bagi validation set menjadi val (50%) dan test (50%)
        val_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
        test_batches = val_batches // 2
        val_ds = val_test_ds.skip(test_batches)
        test_ds = val_test_ds.take(test_batches)
        train_ds = full_train_ds
        
        print(f"Dataset Mendeley: train={tf.data.experimental.cardinality(train_ds).numpy()} batches, "
              f"val={tf.data.experimental.cardinality(val_ds).numpy()} batches, "
              f"test={tf.data.experimental.cardinality(test_ds).numpy()} batches")

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Augmentasi data
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    def augment_data(image, label):
        return data_augmentation(image, training=True), label

    train_ds = train_ds.map(augment_data, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Hitung class weights
    print("\nMenghitung class weights...")
    if dataset_choice == 'FSD':
        class_counts = [count_image_files(train_dir / cls) for cls in class_names]
    else:
        class_counts = [count_image_files(train_dir / cls) for cls in class_names]
    
    total_train = sum(class_counts)
    class_weight_dict = {
        i: total_train / (num_classes * count) for i, count in enumerate(class_counts)
    }
    print("Class counts (train):", dict(zip(class_names, class_counts)))
    print("Class weights:", {class_names[i]: round(w, 2) for i, w in class_weight_dict.items()})

    # ======================== TRAINING NORMAL ========================
    print("\n>>> MEMBANGUN MODEL (NORMAL)")
    model, base_model = build_model(num_classes)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-4
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
    )

    print("\n>>> PELATIHAN NORMAL (Classifier Head)")
    history = model.fit(
        train_ds,
        epochs=30,
        validation_data=val_ds,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Simpan plot & evaluasi
    save_training_plot(history, os.path.join(normal_dir, 'training_plot.png'))

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest Accuracy (normal): {test_acc:.4f}")

    # Prediksi test set
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    save_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(normal_dir, 'confusion_matrix.png'))

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    save_classification_report_csv(report_dict, class_names, accuracy,
                                   os.path.join(base_report_dir, f'{dataset_sub}_normal.csv'))

    print("\nClassification Report (Normal):")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Grad‑CAM normal
    save_gradcam_samples(model, test_ds, class_names, normal_dir, num_samples=5)

    # ======================== FINE‑TUNING ========================
    print("\n>>> FINE‑TUNING? (y/n) ", end='')
    choice = input().strip().lower()
    if choice == 'y':
        print("\n>>> MEMPERSIAPKAN FINE‑TUNING")
        base_model.trainable = True
        for layer in base_model.layers[:80]:
            layer.trainable = False

        optimizer_fine = tf.keras.optimizers.AdamW(
            learning_rate=1e-5,
            weight_decay=1e-5
        )
        model.compile(
            optimizer=optimizer_fine,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history_fine = model.fit(
            train_ds,
            epochs=15,
            validation_data=val_ds,
            class_weight=class_weight_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        save_training_plot(history_fine, os.path.join(fine_dir, 'training_plot.png'))

        test_loss_ft, test_acc_ft = model.evaluate(test_ds, verbose=0)
        print(f"\nTest Accuracy (fine‑tuned): {test_acc_ft:.4f}")

        y_true_ft, y_pred_ft = [], []
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_true_ft.extend(np.argmax(labels.numpy(), axis=1))
            y_pred_ft.extend(np.argmax(preds, axis=1))

        save_confusion_matrix(y_true_ft, y_pred_ft, class_names,
                              os.path.join(fine_dir, 'confusion_matrix.png'))

        report_dict_ft = classification_report(y_true_ft, y_pred_ft, target_names=class_names, output_dict=True)
        accuracy_ft = accuracy_score(y_true_ft, y_pred_ft)
        save_classification_report_csv(report_dict_ft, class_names, accuracy_ft,
                                       os.path.join(base_report_dir, f'{dataset_sub}_fine_tuned.csv'))

        print("\nClassification Report (Fine‑tuned):")
        print(classification_report(y_true_ft, y_pred_ft, target_names=class_names))

        save_gradcam_samples(model, test_ds, class_names, fine_dir, num_samples=5)

    # ======================== SIMPAN MODEL ========================
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model.save_weights(os.path.join(models_dir, f'{dataset_sub}_normal.weights.h5'))
    if choice == 'y':
        model.save_weights(os.path.join(models_dir, f'{dataset_sub}_fine_tuned.weights.h5'))

    print("\n✅ Pelatihan selesai!")
    print(f"Figure disimpan di: {base_fig_dir}")
    print(f"Laporan disimpan di: {base_report_dir}")

if __name__ == "__main__":
    main()