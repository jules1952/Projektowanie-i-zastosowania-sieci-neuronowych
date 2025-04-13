from PIL import Image
import numpy as np
import os

print("Bieżący katalog roboczy:", os.getcwd())


IMAGE_SIZE = (64, 64)              
BATCH_SIZE = 16                    
TRAIN_DIR = r'C:\Users\julia\Desktop\autoencoder\photos\train\happy'
#TEST_DIR = 'Autoencoder/photo/test/happy'
TRANSFORMATION = None              

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            path = os.path.join(folder, filename)
            img = Image.open(path)
            if img.mode != 'L':
                img = img.convert('L')
            if IMAGE_SIZE is not None:
                img = img.resize(IMAGE_SIZE)
            if TRANSFORMATION is not None:
                img = TRANSFORMATION(img)
            # Konwersja do tablicy NumPy i normalizacja
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            #print(np.max(img_array))
            #print(img_array[0:5, 0:5])


        if not images:
            print(f"Brak plików .jpg w folderze: {folder}")    
    return np.array(images)
train_images = load_images_from_folder(TRAIN_DIR)
#test_images = load_images_from_folder(TEST_DIR)

print("Kształt zbioru treningowego:", train_images.shape)
#print("Kształt zbioru testowego:", test_images.shape)

# DataLoader – generator zwracający mini-batche
def data_loader(data, batch_size=BATCH_SIZE, shuffle=True):
    indices = np.arange(len(data))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(data), batch_size):
        batch_indices = indices[start:start+batch_size]
        yield data[batch_indices]

for batch in data_loader(train_images):    
    print("Batch shape:", batch.shape)
    break  # Wyświetlamy tylko pierwszy batch

    
