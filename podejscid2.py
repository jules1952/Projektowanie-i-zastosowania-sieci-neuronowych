import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# ====================================================
# 1. Ustawienia
# ====================================================
IMAGE_SIZE = (64, 64)
TRAIN_DIR   = r'C:\Users\julia\Desktop\autoencoder\banana_train'
TEST_DIR    = r'C:\Users\julia\Desktop\autoencoder\banana_test'
TRANSFORMATION = None

# ====================================================
# 2. Funkcja wczytywania obrazów
# ====================================================
def load_images_from_folder(folder, max_images=None):
    images = []
    count = 0
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
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            count += 1
            if max_images is not None and count >= max_images:
                break
    if not images:
        print(f"Brak plików .jpg w folderze: {folder}")
    return np.array(images)

# ====================================================
# 3. im2col i col2im
# ====================================================
def im2col(x, filter_h, filter_w, stride=1, pad=0):
    N, H, W = x.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    x_padded = np.pad(x, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    col = np.zeros((N, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x_idx in range(filter_w):
            x_max = x_idx + stride*out_w
            col[:, y, x_idx, :, :] = x_padded[:, y:y_max:stride, x_idx:x_max:stride]
    col = col.transpose(0,3,4,1,2).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, x_shape, filter_h, filter_w, stride=1, pad=0):
    N, H, W = x_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, filter_h, filter_w).transpose(0,3,4,1,2)

    x_padded = np.zeros((N, H+2*pad, W+2*pad))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x_idx in range(filter_w):
            x_max = x_idx + stride*out_w
            x_padded[:, y:y_max:stride, x_idx:x_max:stride] += col[:, y, x_idx, :, :]
    if pad == 0:
        return x_padded
    return x_padded[:, pad:-pad, pad:-pad]

# ====================================================
# 4. Wektoryzowana konwolucja (forward + backward)
# ====================================================
def conv_forward_im2col(x, w, b, stride=1, pad=0):
    N, H, W = x.shape
    filter_h, filter_w = w.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    col = im2col(x, filter_h, filter_w, stride, pad)
    w_col = w.reshape(-1,1)
    out = np.dot(col, w_col) + b
    out = out.reshape(N, out_h, out_w)
    cache = (x, w, b, stride, pad, col, w_col, out_h, out_w)
    return out, cache

def conv_backward_im2col(dout, cache):
    x, w, b, stride, pad, col, w_col, out_h, out_w = cache
    dout_flat = dout.reshape(-1,1)
    dw_col = np.dot(col.T, dout_flat)
    dw = dw_col.reshape(w.shape)
    db = np.sum(dout)
    dcol = np.dot(dout_flat, w_col.T)
    dx = col2im(dcol, x.shape, w.shape[0], w.shape[1], stride, pad)
    return dx, dw, db

def conv_forward_im2col_multi(x, w, b, stride=1, pad=0):
    N, H, W = x.shape
    F, filter_h, filter_w = w.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    col = im2col(x, filter_h, filter_w, stride, pad)
    w_col = w.reshape(F, -1).T
    out = np.dot(col, w_col) + b
    out = out.reshape(N, out_h, out_w, F).transpose(0,3,1,2)
    cache = (x, w, b, stride, pad, col, w_col, out_h, out_w)
    return out, cache

def conv_backward_im2col_multi(dout, cache):
    x, w, b, stride, pad, col, w_col, out_h, out_w = cache
    N, H, W = x.shape
    F, filter_h, filter_w = w.shape

    dout_flat = dout.transpose(0,2,3,1).reshape(-1, F)
    dw_col = np.dot(col.T, dout_flat)
    dw = dw_col.T.reshape(w.shape)
    db = np.sum(dout_flat, axis=0)
    dcol = np.dot(dout_flat, w_col.T)
    dx = col2im(dcol, x.shape, filter_h, filter_w, stride, pad)
    return dx, dw, db

# ====================================================
# 5. Wektoryzowana transponowana konwolucja
# ====================================================
def conv_transpose_forward_im2col(x, w, b, stride=1, pad=0, output_shape=None):
    N, H_in, W_in = x.shape
    filter_h, filter_w = w.shape

    # upsample
    H_up = (H_in - 1)*stride + 1
    W_up = (W_in - 1)*stride + 1
    x_up = np.zeros((N, H_up, W_up), dtype=x.dtype)
    x_up[:, ::stride, ::stride] = x

    # effective padding
    pad_eff = filter_h - 1 - pad
    out, conv_cache = conv_forward_im2col(x_up,
                                          np.flip(np.flip(w,0),1),
                                          b,
                                          stride=1,
                                          pad=pad_eff)

    if output_shape is not None:
        _, H_des, W_des = output_shape
        H_cur, W_cur = out.shape[1], out.shape[2]
        if H_cur < H_des or W_cur < W_des:
            out = np.pad(out, ((0,0),(0,H_des-H_cur),(0,W_des-W_cur)), mode='constant')
        else:
            out = out[:, :H_des, :W_des]
    cache = (x, w, b, stride, pad, x_up, pad_eff, conv_cache)
    return out, cache

def conv_transpose_backward_im2col(dout, cache):
    x, w, b, stride, pad, x_up, pad_eff, conv_cache = cache
    orig_h, orig_w = conv_cache[7], conv_cache[8]

    # unpad dout
    H_cur, W_cur = dout.shape[1], dout.shape[2]
    if H_cur>orig_h or W_cur>orig_w:
        dout = dout[:, :orig_h, :orig_w]

    dx_up, dw_flip, db = conv_backward_im2col(dout, conv_cache)
    dx = dx_up[:, ::stride, ::stride]
    dw = np.flip(np.flip(dw_flip,0),1)
    return dx, dw, db

# ====================================================
# 6. Funkcje aktywacji, strat i accuracy
# ====================================================
def relu(x):    return np.maximum(0, x)
def relu_backward(dout, x):
    dx = dout.copy()
    dx[x<=0] = 0
    return dx

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_backward(dout, x):
    s = sigmoid(x)
    return dout * s * (1-s)

def mse_loss(y_true, y_pred):
    loss = np.mean((y_true-y_pred)**2)
    dloss = 2*(y_pred-y_true)/y_true.size
    return loss, dloss

def binary_accuracy(y_true, y_pred, threshold=0.5):
    y_t = (y_true>=threshold).astype(np.float32)
    y_p = (y_pred>=threshold).astype(np.float32)
    return np.mean(y_t==y_p)

# ====================================================
# 7. Definicja autoenkodera
# ====================================================
def autoencoder_forward(x, params):
    # encoder
    conv_enc, cache_enc = conv_forward_im2col_multi(x, params['w_enc'], params['b_enc'],
                                                    stride=params['stride'], pad=params['pad'])
    a_enc = relu(conv_enc)
    a_enc_comb = np.mean(a_enc, axis=1)
    cache_enc = (cache_enc, conv_enc)

    # decoder (jeden kanał)
    w_dec_s = np.mean(params['w_dec'], axis=0)
    b_dec_s = np.mean(params['b_dec'])
    conv_dec, cache_dec = conv_transpose_forward_im2col(a_enc_comb, w_dec_s, b_dec_s,
                                                        stride=params['stride'], pad=params['pad'],
                                                        output_shape=x.shape)
    out = sigmoid(conv_dec)
    return out, {'enc': cache_enc, 'dec': cache_dec, 'conv_dec': conv_dec}

def autoencoder_backward(dout, cache, params):
    grads = {}
    conv_dec = cache['conv_dec']
    dconv_dec = sigmoid_backward(dout, conv_dec)
    dx_dec, dw_dec, db_dec = conv_transpose_backward_im2col(dconv_dec, cache['dec'])
    grads['w_dec'], grads['b_dec'] = dw_dec, db_dec

    cache_enc, conv_enc = cache['enc']
    num_f = params['w_enc'].shape[0]
    dx_dec_exp = np.repeat(dx_dec[:,np.newaxis,:,:], num_f, axis=1)/num_f
    dconv_enc = relu_backward(dx_dec_exp, conv_enc)
    dx_enc, dw_enc, db_enc = conv_backward_im2col_multi(dconv_enc, cache_enc)
    grads['w_enc'], grads['b_enc'], grads['dx'] = dw_enc, db_enc, dx_enc
    return grads

# ====================================================
# 8. Przygotowanie i trening
# ====================================================
np.random.seed(42)
train_images = load_images_from_folder(TRAIN_DIR, max_images=900)
test_images  = load_images_from_folder(TEST_DIR,  max_images=100)
print("Train:", train_images.shape, "Test:", test_images.shape)

params = {
    'w_enc': np.random.randn(100,3,3)*0.1,
    'b_enc': np.zeros(100),
    'w_dec': np.random.randn(100,3,3)*0.1,
    'b_dec': np.zeros(100),
    'stride': 1,
    'pad': 1
}
lr = 5e-6
epochs = 260
loss_hist, acc_hist = [], []

for ep in range(epochs):
    out, cache = autoencoder_forward(train_images, params)
    loss, dloss = mse_loss(train_images, out)
    acc = binary_accuracy(train_images, out)
    loss_hist.append(loss)
    acc_hist.append(acc)

    grads = autoencoder_backward(dloss, cache, params)
    params['w_enc'] -= lr * grads['w_enc']
    params['b_enc'] -= lr * grads['b_enc']
    params['w_dec'] -= lr * grads['w_dec']
    params['b_dec'] -= lr * grads['b_dec']

    if ep % 10 == 0:
        print(f"Epoch {ep}, Loss: {loss:.6f}, Acc: {acc:.4f}")

# ====================================================
# 9. Ewaluacja na teście
# ====================================================
out_test, _ = autoencoder_forward(test_images, params)
loss_test, _ = mse_loss(test_images, out_test)
acc_test    = binary_accuracy(test_images, out_test)
print(f"\nTest Loss: {loss_test:.6f}, Test Acc: {acc_test:.4f}")

# ====================================================
# 10. Wizualizacje
# ====================================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(loss_hist); plt.title("MSE Loss"); plt.xlabel("Epoka")
plt.subplot(1,2,2)
plt.plot(acc_hist);  plt.title("Binary Accuracy"); plt.xlabel("Epoka")
plt.tight_layout()
plt.show()

# kilka rekonstrukcji z testu
num_show = 5
idxs = random.sample(range(len(test_images)), num_show)
plt.figure(figsize=(12,4))
for i, idx in enumerate(idxs):
    # oryginał
    plt.subplot(2, num_show, i+1)
    plt.imshow(test_images[idx], cmap='gray'); plt.axis('off')
    # rekonstrukcja
    plt.subplot(2, num_show, num_show+i+1)
    plt.imshow(out_test[idx], cmap='gray'); plt.axis('off')
plt.tight_layout()
plt.show()
