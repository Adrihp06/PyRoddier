from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass, shift

def load_fits_image(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data

def filter_noise(img, gaussian_sigma=1, threshold_ratio=0.09):
    smoothed_img = gaussian_filter(img, sigma=gaussian_sigma)
    smoothed_img = smoothed_img / np.max(smoothed_img)
    threshold = threshold_ratio * np.max(smoothed_img)
    binary_mask = smoothed_img > threshold
    return smoothed_img * binary_mask

def center_image(img):
    center = center_of_mass(img)
    img_center = np.array(img.shape) // 2
    shift_values = img_center - center
    return shift(img, shift_values, mode="constant", cval=0)

def crop_to_center(img, size):
    center = np.array(img.shape) // 2
    half_size = size // 2
    return img[center[0] - half_size:center[0] + half_size, center[1] - half_size:center[1] + half_size]


def generate_binary_mask(img1, img2, threshold=0.1):
    image_sum = img1 + img2
    image_sum = image_sum / np.max(image_sum)
    return image_sum > threshold

def preprocess_images(intra_img, extra_img, threshold=0.09):
    # Filtrar ruido
    intra_img = filter_noise(intra_img, threshold)
    extra_img = filter_noise(extra_img, threshold)

    # Centrar imágenes
    intra_img = center_image(intra_img)
    extra_img = center_image(extra_img)
    # Recortar a tamaño deseado
    #intra_img = crop_to_center(intra_img, crop_size)
    #extra_img = crop_to_center(extra_img, crop_size)

    return intra_img, extra_img