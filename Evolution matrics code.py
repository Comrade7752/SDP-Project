// Evolution matrics code

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