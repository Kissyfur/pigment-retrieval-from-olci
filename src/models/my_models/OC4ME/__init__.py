from src.models.my_models import Model
import numpy as np


class OC4ME(Model):
    class_name = 'OC4ME'

    def __init__(self, name='OC4ME'):
        super().__init__(name)
        self.x_wavelengths = [442, 490, 510, 560]

    def predict(self, x):
        x_ext = self.extract_x_variables(x)
        max_Rrs = np.max(x_ext[:, 0:3], axis=1)
        Rrs_560 = x_ext[:, 3]
        log_x = np.log10(max_Rrs / Rrs_560)
        chla = 10**(0.3255 - 2.7677 * log_x + 2.4409 * pow(log_x, 2) - 1.12259 * pow(log_x, 3) + 0.5683 * pow(log_x, 4))
        return chla

