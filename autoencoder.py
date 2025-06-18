import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ====================================================
# 1. Funkcje wczytywania obrazów
# ====================================================

IMAGE_SIZE = (64, 64)  # Ustawiamy obrazy 32x32
#TRAIN_DIR = r'C:\Users\julia\Desktop\autoencoder\photos\train\happy'
#TRAIN_DIR = r'C:\Users\julia\Desktop\autoencoder\banana'

#TEST_DIR  = r'C:\Users\julia\Desktop\autoencoder\banana_train'

TRAIN_DIR = r'C:\Users\julia\Desktop\autoencoder\banana_train'
TEST_DIR  = r'C:\Users\julia\Desktop\autoencoder\banana'
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
    """
    x: wejście o wymiarach (N, H, W)
    w: wagi o wymiarach (F, filter_h, filter_w) – F to liczba filtrów
    b: bias o wymiarach (F,)
    """
    N, H, W = x.shape
    F, filter_h, filter_w = w.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # Zamiana obrazu na macierz kolumnową
    col = im2col(x, filter_h, filter_w, stride, pad)  # kształt: (N*out_h*out_w, filter_h*filter_w)
    
    # Przekształcenie wag: (F, filter_h, filter_w) -> (filter_h*filter_w, F)
    w_col = w.reshape(F, -1).T
    
    # Mnożenie macierzowe: wynik kształtu (N*out_h*out_w, F)
    out = np.dot(col, w_col) + b  # bias zostanie dodany do każdego "patcha"
    
    # Reshape wyniku do (N, out_h, out_w, F) i transpozycja do (N, F, out_h, out_w)
    out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
    
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


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Binarize both true and predicted images przy zadanym progu,
    a następnie policz odsetek pikseli zgadzających się.
    """
    y_t = (y_true >= threshold).astype(np.float32)
    y_p = (y_pred >= threshold).astype(np.float32)
    return np.mean(y_t == y_p)

# ====================================================
# 6. Autoenkoder – forward i backward
# ====================================================
# 6. PIERWSZA WARSTWA ENCODERA Autoenkoder – forward i backward
def autoencoder_forward(x, params):
    # Enkoder: konwolucja (wielofiltrowa) -> ReLU
    conv_enc, cache_enc = conv_forward_im2col_multi(x, params['w_enc'], params['b_enc'],
                                                     stride=params['stride'], pad=params['pad'])
    a_enc = relu(conv_enc)  # a_enc ma teraz kształt (N, num_filters, H, W)
    
    # Połącz kanały – np. średnia po osi kanałów,
    # aby uzyskać pojedynczą mapę (N, H, W)
    a_enc_combined = np.mean(a_enc, axis=1)
    
    cache_enc = (cache_enc, conv_enc)
    
    # Aby funkcja transponowanej konwolucji działała poprawnie,
    # przygotuj pojedynczy filtr dekodera, uśredniając filtry (o oryginalnym kształcie (num_filters, 3, 3))
    w_dec_single = np.mean(params['w_dec'], axis=0)  # wynik: (3, 3)
    b_dec_single = np.mean(params['b_dec'])           # wynik: skalar
    
    # Dekoder: transponowana konwolucja -> Sigmoid
    conv_dec, cache_dec = conv_transpose_forward_im2col(a_enc_combined, w_dec_single, b_dec_single,
                                                         stride=params['stride'], pad=params['pad'],
                                                         output_shape=x.shape)
    out = sigmoid(conv_dec)
    cache = {'enc': cache_enc, 'dec': cache_dec, 'conv_dec': conv_dec}
    return out, cache



def mse_loss(y_true, y_pred):
    loss = np.mean((y_true - y_pred) ** 2)
    dloss = 2 * (y_pred - y_true) / y_true.size
    return loss, dloss

def autoencoder_backward(dout, cache, params):
    grads = {}
    conv_dec = cache['conv_dec']
    dconv_dec = sigmoid_backward(dout, conv_dec)
    
    dx_dec, dw_dec, db_dec = conv_transpose_backward_im2col(dconv_dec, cache['dec'])
    grads['w_dec'] = dw_dec
    grads['b_dec'] = db_dec
    
    cache_enc, conv_enc = cache['enc']

    # Rozszerzenie gradientu z dekodera do wymiarów (N, num_filters, H, W)
    num_filters = params['w_enc'].shape[0]
    dx_dec_expanded = np.repeat(dx_dec[:, np.newaxis, :, :], num_filters, axis=1) / num_filters

    # Teraz przekazujemy gradient do funkcji relu_backward, która oczekuje 4-wymiarowych tablic
    dconv_enc = relu_backward(dx_dec_expanded, conv_enc)
    
    dx_enc, dw_enc, db_enc = conv_backward_im2col_multi(dconv_enc, cache_enc)
    grads['w_enc'] = dw_enc
    grads['b_enc'] = db_enc
    grads['dx'] = dx_enc
    return grads


# ====================================================
# 7. Schemat treningowy autoenkodera
# ====================================================

np.random.seed(42)
num_filters = 100

params = {
    'w_enc': np.random.randn(num_filters, 3, 3) * 0.1,  # Wagi enkodera o wymiarach (8, 3, 3)
    'b_enc': np.zeros(num_filters),                     # Bias dla każdego filtra, wymiar (8,)
    'w_dec': np.random.randn(num_filters, 3, 3) * 0.1,    # Wagi dekodera analogicznie
    'b_dec': np.zeros(num_filters),
    'stride': 1,  # Kontynuujemy z ustawionym wcześniej stride=1
    'pad': 1
}

x_train = train_images  # (900, 32, 32)

learning_rate = 0.000005
epochs = 100
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    out, cache = autoencoder_forward(x_train, params)
    loss, dloss = mse_loss(x_train, out)
    acc = binary_accuracy(x_train, out)

    loss_history.append(loss)
    accuracy_history.append(acc)

    grads = autoencoder_backward(dloss, cache, params)
    params['w_enc'] -= learning_rate * grads['w_enc']
    params['b_enc'] -= learning_rate * grads['b_enc']
    params['w_dec'] -= learning_rate * grads['w_dec']
    params['b_dec'] -= learning_rate * grads['b_dec']

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}, Accuracy: {acc:.4f}")

# ====================================================
# 7. Wizualizacja Loss i Accuracy
# ====================================================

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Spadek straty (MSE)")
plt.xlabel("Epoka")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(accuracy_history)
plt.title("Accuracy rekonstrukcji")
plt.xlabel("Epoka")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()

# ====================================================
# 8. Przykładowa rekonstrukcja
# ====================================================

# Generujemy rekonstrukcje na całym zbiorze treningowym
out_final, _ = autoencoder_forward(x_train, params)

# --- 1) MSE ---
mse_val, _ = mse_loss(x_train, out_final)

# --- 2) PSNR ---
def psnr(y_true, y_pred, data_range=1.0):
    mse = np.mean((y_true - y_pred)**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((data_range**2) / mse)

psnr_val = psnr(x_train, out_final)

# --- 3) Ręczne SSIM ---
def ssim_global(y_true, y_pred, data_range=1.0, K1=0.01, K2=0.03):
    x = y_true.flatten()
    y = y_pred.flatten()
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x2 = x.var()
    sigma_y2 = y.var()
    sigma_xy = ((x - mu_x)*(y - mu_y)).mean()
    C1 = (K1*data_range)**2
    C2 = (K2*data_range)**2
    num = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1)*(sigma_x2 + sigma_y2 + C2)
    return num/den

ssim_vals = [ssim_global(x_train[i], out_final[i]) for i in range(len(x_train))]
ssim_mean = np.mean(ssim_vals)

# --- Drukujemy wyniki ---
print(f"MSE:  {mse_val:.6f}")
print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_mean:.4f}")

# --- 4) Wizualizacja jednej pary oryginał ↔ rekonstrukcja ---
sample_idx = 0
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.plot(loss_history, epochs)
plt.xlabel("Loss (MSE)")
plt.ylabel("Epoka")
plt.title("Spadek straty (MSE)")

# 2) Accuracy na osi X, Epoka na osi Y
plt.subplot(1,2,2)
plt.plot(accuracy_history, epochs)
plt.xlabel("Accuracy rekonstrukcji")
plt.ylabel("Epoka")
plt.title("Accuracy vs Epoka")

plt.tight_layout()
plt.show()
# ====================================================
# 9. Ewaluacja na zbiorze testowym
# ====================================================
# Generujemy rekonstrukcje na zbiorze testowym
out_test, _ = autoencoder_forward(test_images, params)

# 1) MSE
mse_test, _ = mse_loss(test_images, out_test)

# 2) PSNR
psnr_test = psnr(test_images, out_test)

# 3) SSIM
ssim_test_vals = [ssim_global(test_images[i], out_test[i]) for i in range(len(test_images))]
ssim_test_mean = np.mean(ssim_test_vals)

# Drukujemy wyniki
print(f"Test set MSE:  {mse_test:.6f}")
print(f"Test set PSNR: {psnr_test:.2f} dB")
print(f"Test set SSIM: {ssim_test_mean:.4f}")
# --- 1) Średnia accuracy po treningu ---
mean_train_acc = np.mean(accuracy_history)
print(f"Średnia accuracy na zbiorze treningowym: {mean_train_acc:.4f}")

# --- 2) Accuracy na zbiorze testowym ---
# Zakładam, że masz już out_test z autoencoder_forward(test_images, params)
acc_test = binary_accuracy(test_images, out_test)
print(f"Accuracy na zbiorze testowym:        {acc_test:.4f}")



plt.tight_layout()
plt.show()
