import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Instruksi: 
# 1. Download dataset dari Kaggle: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
# 2. Ekstrak ke folder, misal './chest-xray-masks-and-labels'
# 3. Struktur: Lung Segmentation/CXR_png/ berisi images .png
# 4. Label diambil dari nama file, contoh: CHNCXR_0002_0.png -> label 0 (Normal), CHNCXR_0435_1.png -> label 1 (TB)
# 5. Jalankan kode ini di environment dengan TensorFlow, seperti Google Colab.

# Path ke folder images
image_dir = "F:\\lung_segmentation\\Lung Segmentation\\CXR_png"  # Sesuaikan jika perlu

# Fungsi untuk extract label dari nama file
def get_label_from_filename(filename):
    try:
        # Ambil angka terakhir dari nama file, misal CHNCXR_xxxx_0.png -> 0
        label = int(filename.split('_')[-1].split('.')[0])
        if label not in [0, 1]:
            raise ValueError(f"Label {label} tidak valid di file {filename}. Harus 0 (Normal) atau 1 (TB).")
        return label  # 0: Normal, 1: TB
    except Exception as e:
        raise ValueError(f"Error memproses nama file {filename}: {e}")

# Load daftar images dan labels
images = []
labels = []
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        try:
            label = get_label_from_filename(filename)
            img_path = os.path.join(image_dir, filename)
            img = image.load_img(img_path, target_size=(224, 224))  # Resize untuk VGG16
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error memproses {filename}: {e}")
            continue

images = np.array(images)
labels = np.array(labels)
labels = to_categorical(labels, num_classes=2)  # One-hot encoding untuk binary classification

# Normalize images
images = images / 255.0

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load VGG16 pretrained on ImageNet, tanpa top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers untuk transfer learning awal
base_model.trainable = False

# Tambah custom classifier
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # Binary: Normal or TB
])

# Compile model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model tanpa augmentasi
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test)
)

# Evaluasi akurasi awal
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Akurasi test awal: {test_acc * 100:.2f}%')

# Jika akurasi < 90%, lakukan fine-tuning
if test_acc < 0.90:
    print("Melakukan fine-tuning untuk meningkatkan akurasi...")
    
    # Unfreeze top layers VGG16 (misal last 4 layers)
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 4
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Compile ulang dengan learning rate lebih kecil
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train lagi
    fine_tune_history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_test, y_test)
    )
    
    # Evaluasi akhir
    final_loss, final_acc = model.evaluate(X_test, y_test)
    print(f'Akurasi test setelah fine-tuning: {final_acc * 100:.2f}%')

# Buat prediksi pada test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Hitung confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Tampilkan confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'TB'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
if 'fine_tune_history' in locals():
    plt.plot(fine_tune_history.history['accuracy'], label='Fine-tune Train Acc')
    plt.plot(fine_tune_history.history['val_accuracy'], label='Fine-tune Val Acc')
plt.legend()
plt.show()

# Save model
model.save('tb_classification_vgg16_model.h5')
print("Model disimpan sebagai 'tb_classification_vgg16_model.h5'")