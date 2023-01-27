import numpy as np
from scipy.fft import rfft

DELTA_ANGLE = 1.0


def get_fft(data: np.ndarray, n_of_waves):

    # https://docs.scipy.org/doc/scipy/tutorial/fft.html

    x_values = np.arange(0, 360, DELTA_ANGLE)

    data_rfft = rfft(data)

    cos_coeffs = 2 * np.real(data_rfft) / len(data)
    sin_coeffs = 2 * np.imag(data_rfft) / len(data)

    cos_mask = np.argsort(np.abs(cos_coeffs))[-1:-n_of_waves-1:-1]
    sin_mask = np.argsort(np.abs(sin_coeffs))[-1:-n_of_waves-1:-1]

    cos_coeffs = cos_coeffs[cos_mask]
    sin_coeffs = sin_coeffs[sin_mask]

    cos_wave = - cos_mask[:, np.newaxis] * x_values[np.newaxis, :] * (np.pi / 180)
    cos_wave = cos_coeffs[:, np.newaxis] * np.cos(cos_wave)

    sin_wave = - sin_mask[:, np.newaxis] * x_values[np.newaxis, :] * (np.pi / 180)
    sin_wave = sin_coeffs[:, np.newaxis] * np.sin(sin_wave)

    wave = np.sum(cos_wave + sin_wave, axis=0)

    return wave