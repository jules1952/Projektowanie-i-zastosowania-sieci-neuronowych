import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ====================================================
# 1. Funkcje wczytywania obrazów
# ====================================================

IMAGE_SIZE = (32, 32)  # Ustawiamy obrazy 32x32
TRAIN_DIR = r'C:\Users\julia\Desktop\autoencoder\photos\train\happy'
TEST_DIR  = r'C:\Users\julia\Desktop\autoencoder\photos\test\happy'
TRANSFORMATION = None

def load_images_from_folder(folder, max_images=None):
    images = []
    count = 0
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            path = os.path.join(folder, filename)
            img = Image.open(path)
            if img.mode != 'L':  # konwersja do skali szarości
                img = img.convert('L')
            if IMAGE_SIZE is not None:
                img = img.resize(IMAGE_SIZE)
            if TRANSFORMATION is not None:
                img = TRANSFORMATION(img)
            # Normalizacja: wartości pikseli w zakresie [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            count += 1
            if max_images is not None and count >= max_images:
                break
    if not images:
        print(f"Brak plików .jpg w folderze: {folder}")
    return np.array(images)

# Wczytanie danych
train_images = load_images_from_folder(TRAIN_DIR, max_images=900)
test_images  = load_images_from_folder(TEST_DIR)
print("Kształt zbioru treningowego:", train_images.shape)
print("Kształt zbioru testowego:", test_images.shape)

# Jeśli chcesz przetestować działanie bez obrazów, możesz zamiast tego:
# train_images = np.random.rand(900, 32, 32).astype(np.float32)
# test_images  = np.random.rand(1774, 32, 32).astype(np.float32)

# ====================================================
# 2. Implementacja im2col i col2im
# ====================================================

def im2col(x, filter_h, filter_w, stride=1, pad=0):
    N, H, W = x.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    x_padded = np.pad(x, ((0,0), (pad, pad), (pad, pad)), mode='constant')
    
    col = np.zeros((N, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x_idx in range(filter_w):
            x_max = x_idx + stride * out_w
            col[:, y, x_idx, :, :] = x_padded[:, y:y_max:stride, x_idx:x_max:stride]
    col = col.transpose(0, 3, 4, 1, 2).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, x_shape, filter_h, filter_w, stride=1, pad=0):
    N, H, W = x_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, filter_h, filter_w).transpose(0, 3, 4, 1, 2)
    x_padded = np.zeros((N, H + 2 * pad, W + 2 * pad))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x_idx in range(filter_w):
            x_max = x_idx + stride * out_w
            x_padded[:, y:y_max:stride, x_idx:x_max:stride] += col[:, y, x_idx, :, :]
    if pad == 0:
        return x_padded
    return x_padded[:, pad:-pad, pad:-pad]

def conv_forward_im2col_multi(x, w, b, stride=1, pad=0):
    # Sprawdź, czy dane wejściowe mają cztery wymiary
    if x.ndim == 4:
        N, C, H, W = x.shape
        # Przekształć dane do kształtu (N*C, H, W)
        x_reshaped = x.reshape(N * C, H, W)
    else:
        x_reshaped = x
        N, H, W = x.shape
        C = 1  # Zakładamy pojedynczy kanał, jeśli C nie jest określone

    F, filter_h, filter_w = w.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = im2col(x_reshaped, filter_h, filter_w, stride, pad)
    w_col = w.reshape(F, -1).T
    out = np.dot(col, w_col) + b
    out = out.reshape(N, C, out_h, out_w, F).transpose(0, 4, 1, 2, 3).reshape(N, F, out_h, out_w)
    cache = (x, w, b, stride, pad, col, w_col, out_h, out_w)
    return out, cache

# ====================================================
# 3. Wektoryzowana konwolucja (forward i backward) – im2col
# ====================================================

def conv_forward_im2col(x, w, b, stride=1, pad=0):
    N, H, W = x.shape
    filter_h, filter_w = w.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
    col = im2col(x, filter_h, filter_w, stride, pad)
    w_col = w.reshape(-1, 1)
    out = np.dot(col, w_col) + b
    out = out.reshape(N, out_h, out_w)
    
    cache = (x, w, b, stride, pad, col, w_col, out_h, out_w)
    return out, cache

def conv_backward_im2col(dout, cache):
    x, w, b, stride, pad, col, w_col, out_h, out_w = cache
    N, H, W = x.shape
    
    dout_reshaped = dout.reshape(-1, 1)
    dw_col = np.dot(col.T, dout_reshaped)
    dw = dw_col.reshape(w.shape)
    db = np.sum(dout)
    
    dcol = np.dot(dout_reshaped, w_col.T)
    dx = col2im(dcol, x.shape, w.shape[0], w.shape[1], stride, pad)
    return dx, dw, db

def conv_backward_im2col_multi(dout, cache):
    """
    dout: gradient z wyjścia o kształcie (N, F, out_h, out_w)
    cache: krotka z wartościami z forward pass: (x, w, b, stride, pad, col, w_col, out_h, out_w)
    """
    x, w, b, stride, pad, col, w_col, out_h, out_w = cache
    N, H, W = x.shape
    F, filter_h, filter_w = w.shape

    # Przekształcenie dout: (N, F, out_h, out_w) -> (N*out_h*out_w, F)
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, F)

    # Obliczanie gradientu dla wag
    dw_col = np.dot(col.T, dout_reshaped)  # wynik: (filter_h*filter_w, F)
    dw = dw_col.T.reshape(w.shape)

    # Gradient dla bias – sumujemy gradient
    db = np.sum(dout_reshaped, axis=0)

    # Propagacja gradientu do wejścia
    dcol = np.dot(dout_reshaped, w_col.T)  # wynik: (N*out_h*out_w, filter_h*filter_w)
    dx = col2im(dcol, x.shape, filter_h, filter_w, stride, pad)

    return dx, dw, db




# ====================================================
# 4. Wektoryzowana transponowana konwolucja (forward i backward)
# ====================================================

def conv_transpose_forward_im2col(x, w, b, stride=1, pad=0, output_shape=None):
    N, H_in, W_in = x.shape
    filter_h, filter_w = w.shape
    H_upsampled = (H_in - 1) * stride + 1
    W_upsampled = (W_in - 1) * stride + 1
    x_upsampled = np.zeros((N, H_upsampled, W_upsampled), dtype=x.dtype)
    x_upsampled[:, ::stride, ::stride] = x

    pad_eff = filter_h - 1 - pad
    out, conv_cache = conv_forward_im2col(x_upsampled, np.flip(np.flip(w, axis=0), axis=1), b, stride=1, pad=pad_eff)
    
    if output_shape is not None:
        N_out, H_out_desired, W_out_desired = output_shape
        H_out_current, W_out_current = out.shape[1], out.shape[2]
        if H_out_current < H_out_desired or W_out_current < W_out_desired:
            pad_H = H_out_desired - H_out_current
            pad_W = W_out_desired - W_out_current
            out = np.pad(out, ((0,0), (0, pad_H), (0, pad_W)), mode='constant')
        else:
            out = out[:, :H_out_desired, :W_out_desired]
    
    cache = (x, w, b, stride, pad, x_upsampled, pad_eff, conv_cache)
    return out, cache

def conv_transpose_backward_im2col(dout, cache):
    x, w, b, stride, pad, x_upsampled, pad_eff, conv_cache = cache
    # conv_cache zawiera oryginalne out_h oraz out_w (przed dopełnieniem) w pozycjach 7 i 8
    orig_out_h, orig_out_w = conv_cache[7], conv_cache[8]
    
    # Jeśli dout (gradient z wyjścia) ma większe wymiary, usuwamy dodatkowy padding
    current_H, current_W = dout.shape[1], dout.shape[2]
    if current_H > orig_out_h or current_W > orig_out_w:
        dout_unpadded = dout[:, :orig_out_h, :orig_out_w]
    else:
        dout_unpadded = dout

    dx_upsampled, dw_flip, db = conv_backward_im2col(dout_unpadded, conv_cache)
    dx = dx_upsampled[:, ::stride, ::stride]
    dw = np.flip(np.flip(dw_flip, axis=0), axis=1)
    return dx, dw, db

# ====================================================
# 5. Funkcje aktywacji i ich backward
# ====================================================

def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(dout, x):
    s = sigmoid(x)
    return dout * s * (1 - s)

# ====================================================
# 6. Autoenkoder – forward i backward
# ====================================================
# 6. PIERWSZA WARSTWA ENCODERA Autoenkoder – forward i backward
def autoencoder_forward_extended(x, params):
    # Jeśli wejście ma kształt (N, 1, H, W), usuń dodatkowy wymiar kanału.
    if x.ndim == 4 and x.shape[1] == 1:
        x = np.squeeze(x, axis=1)  # x będzie miało kształt (N, H, W)

    # --- Enkoder ---
    conv_enc1, cache_enc1 = conv_forward_im2col_multi(x, params['w_enc1'], params['b_enc1'],
                                                       stride=params['stride'], pad=params['pad'])
    a_enc1 = relu(conv_enc1)  # kształt: (N, 8, H, W)
    
    conv_enc2, cache_enc2 = conv_forward_im2col_multi(a_enc1, params['w_enc2'], params['b_enc2'],
                                                       stride=params['stride'], pad=params['pad'])
    a_enc2 = relu(conv_enc2)  # kształt: (N, 16, H, W)
    
    latent = a_enc2  # reprezentacja latentna, kształt (N, 16, H, W)
    
    # --- Dekoder ---
    output_shape_dec1 = (params['w_enc1'].shape[0], x.shape[1], x.shape[2])  # np. (8, H, W)
    conv_dec1, cache_dec1 = conv_transpose_forward_im2col(latent, params['w_dec1'], params['b_dec1'],
                                                           stride=params['stride'], pad=params['pad'],
                                                           output_shape=output_shape_dec1)
    a_dec1 = relu(conv_dec1)  # kształt: (N, 8, H, W)
    
    conv_dec2, cache_dec2 = conv_transpose_forward_im2col(a_dec1, params['w_dec2'], params['b_dec2'],
                                                           stride=params['stride'], pad=params['pad'],
                                                           output_shape=x.shape)
    out = sigmoid(conv_dec2)  # ostateczny obraz, wartości w [0,1]
    
    caches = {
        'enc': (cache_enc1, conv_enc1, cache_enc2, conv_enc2),
        'dec': (cache_dec1, conv_dec1, cache_dec2, conv_dec2)
    }
    return out, caches



def mse_loss(y_true, y_pred):
    loss = np.mean((y_true - y_pred) ** 2)
    dloss = 2 * (y_pred - y_true) / y_true.size
    return loss, dloss

def autoencoder_backward_extended(dout, caches, params):
    grads = {}
    # --- Dekoder ---
    cache_dec1, conv_dec1, cache_dec2, conv_dec2 = caches['dec']
    
    dconv_dec2 = sigmoid_backward(dout, conv_dec2)
    dx_dec2, dw_dec2, db_dec2 = conv_transpose_backward_im2col(dconv_dec2, cache_dec2)
    grads['w_dec2'] = dw_dec2
    grads['b_dec2'] = db_dec2
    
    dconv_dec1 = relu_backward(dx_dec2, conv_dec1)
    dx_dec1, dw_dec1, db_dec1 = conv_transpose_backward_im2col(dconv_dec1, cache_dec1)
    grads['w_dec1'] = dw_dec1
    grads['b_dec1'] = db_dec1
    
    # --- Enkoder ---
    cache_enc1, conv_enc1, cache_enc2, conv_enc2 = caches['enc']
    
    dconv_enc2 = relu_backward(dx_dec1, conv_enc2)
    dx_enc2, dw_enc2, db_enc2 = conv_backward_im2col_multi(dconv_enc2, cache_enc2)
    grads['w_enc2'] = dw_enc2
    grads['b_enc2'] = db_enc2
    
    dconv_enc1 = relu_backward(dx_enc2, conv_enc1)
    dx_enc1, dw_enc1, db_enc1 = conv_backward_im2col_multi(dconv_enc1, cache_enc1)
    grads['w_enc1'] = dw_enc1
    grads['b_enc1'] = db_enc1
    
    return grads


# ====================================================
# 7. Schemat treningowy autoenkodera
# ====================================================

np.random.seed(42)
num_filters_enc1 = 8
num_filters_enc2 = 16
num_filters_dec1 = num_filters_enc2  # 16
num_filters_dec2 = num_filters_enc1  # 8

params = {
    'w_enc1': np.random.randn(num_filters_enc1, 3, 3) * 0.1,
    'b_enc1': np.zeros(num_filters_enc1),
    'w_enc2': np.random.randn(num_filters_enc2, 3, 3) * 0.1,
    'b_enc2': np.zeros(num_filters_enc2),
    'w_dec1': np.random.randn(num_filters_dec1, 3, 3) * 0.1,
    'b_dec1': np.zeros(num_filters_dec1),
    'w_dec2': np.random.randn(num_filters_dec2, 3, 3) * 0.1,
    'b_dec2': np.zeros(num_filters_dec2),
    'stride': 1,
    'pad': 1
}

x_train = train_images  # (900, 32, 32)

learning_rate = 0.00001
epochs = 200
loss_history = []


for epoch in range(epochs):
    out, cache = autoencoder_forward_extended(x_train, params)
    loss, dloss = mse_loss(x_train, out)
    loss_history.append(loss)
    
    grads = autoencoder_backward_extended(dloss, cache, params)
    
    params['w_enc1'] -= learning_rate * grads['w_enc1']
    params['b_enc1'] -= learning_rate * grads['b_enc1']
    params['w_enc2'] -= learning_rate * grads['w_enc2']
    params['b_enc2'] -= learning_rate * grads['b_enc2']
    params['w_dec1'] -= learning_rate * grads['w_dec1']
    params['b_dec1'] -= learning_rate * grads['b_dec1']
    params['w_dec2'] -= learning_rate * grads['w_dec2']
    params['b_dec2'] -= learning_rate * grads['b_dec2']

    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# ====================================================
# 8. Wizualizacja wyników
# ====================================================

plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.title("Spadek straty (MSE)")
plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.show()

out, _ = autoencoder_forward_extended(x_train, params)
sample_idx = 0

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Oryginalny obraz")
plt.imshow(x_train[sample_idx], cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Rekonstrukcja")
plt.imshow(out[sample_idx], cmap='gray')
plt.axis('off')
plt.show()
