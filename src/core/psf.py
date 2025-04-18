import numpy as np

def calcular_psf(W_rec, pupila_mask):
    fase_W = 2 * np.pi * W_rec
    pupil_function = pupila_mask * np.exp(1j * fase_W)
    E_focal = np.fft.fftshift(np.fft.fft2(pupil_function))
    PSF = np.abs(E_focal)**2
    PSF /= PSF.max()
    PSF_log = np.log10(PSF + 1e-8)
    return PSF, PSF_log