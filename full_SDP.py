# Generatror

from tensorflow.keras.layers import Input, Conv2D, Concatenate, BatchNormalization
from tensorflow.keras.models import Model

def build_encoder(input_shape=(256, 256, 3)):
    cover_input = Input(shape=input_shape, name='cover_input')
    secret_input = Input(shape=input_shape, name='secret_input')

    x = Concatenate(axis=-1)([cover_input, secret_input])  # (256, 256, 6)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    stego_output = Conv2D(3, (1, 1), activation='sigmoid', padding='same', name='stego_output')(x)
    return Model(inputs=[cover_input, secret_input], outputs=stego_output, name='Encoder')

# Discriminator

def build_decoder(input_shape=(256, 256, 3)):
    stego_input = Input(shape=input_shape, name='stego_input')

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(stego_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    secret_reconstructed = Conv2D(3, (1, 1), activation='sigmoid', padding='same', name='secret_reconstructed')(x)
    return Model(inputs=stego_input, outputs=secret_reconstructed, name='Decoder')

# End to end workflow

from tensorflow.keras.models import Model

def build_steganography_model(input_shape=(256, 256, 3)):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(input_shape)

    cover_input = Input(shape=input_shape, name='cover_input')
    secret_input = Input(shape=input_shape, name='secret_input')

    stego_output = encoder([cover_input, secret_input])
    secret_reconstructed = decoder(stego_output)

    full_model = Model(inputs=[cover_input, secret_input], outputs=[stego_output, secret_reconstructed], name='SteganographyModel')
    return full_model

# To train and compile the model

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

model = build_steganography_model()

# Losses: stego should match cover, reconstructed secret should match original
mse = MeanSquaredError()
lambda_cover = 1.0
lambda_secret = 1.0

def custom_loss(y_true, y_pred):
    return mse(y_true, y_pred)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        "stego_output": lambda y_true, y_pred: lambda_cover * mse(y_true, y_pred),
        "secret_reconstructed": lambda y_true, y_pred: lambda_secret * mse(y_true, y_pred)
    }
)

# Evolution matrics code

import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Assume test images are float32 and in range [0, 1]
# cover_images: true cover images
# stego_images: predicted images from encoder
# secret_images: true secret images
# recovered_secrets: output from decoder

def evaluate_steganography(cover_images, stego_images, secret_images, recovered_secrets):
    num_samples = cover_images.shape[0]

    psnr_cover_stego = []
    psnr_secret_recon = []
    ssim_cover_stego = []
    ssim_secret_recon = []
    mse_cover_stego = []
    mse_secret_recon = []

    for i in range(num_samples):
        c = cover_images[i]
        s = secret_images[i]
        stego = stego_images[i]
        recovered = recovered_secrets[i]

        # Convert to 8-bit for SSIM/PSNR (optional)
        c_uint8 = (c * 255).astype(np.uint8)
        s_uint8 = (s * 255).astype(np.uint8)
        stego_uint8 = (stego * 255).astype(np.uint8)
        recovered_uint8 = (recovered * 255).astype(np.uint8)

        psnr_cover_stego.append(psnr(c_uint8, stego_uint8, data_range=255))
        psnr_secret_recon.append(psnr(s_uint8, recovered_uint8, data_range=255))

        ssim_cover_stego.append(ssim(c_uint8, stego_uint8, multichannel=True, data_range=255))
        ssim_secret_recon.append(ssim(s_uint8, recovered_uint8, multichannel=True, data_range=255))

        mse_cover_stego.append(mean_squared_error(c.flatten(), stego.flatten()))
        mse_secret_recon.append(mean_squared_error(s.flatten(), recovered.flatten()))

    results = {
        "PSNR Cover vs Stego": np.mean(psnr_cover_stego),
        "PSNR Secret vs Recovered": np.mean(psnr_secret_recon),
        "SSIM Cover vs Stego": np.mean(ssim_cover_stego),
        "SSIM Secret vs Recovered": np.mean(ssim_secret_recon),
        "MSE Cover vs Stego": np.mean(mse_cover_stego),
        "MSE Secret vs Recovered": np.mean(mse_secret_recon)
    }

    return results