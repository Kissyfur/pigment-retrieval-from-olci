import pandas as pd

from src.models import BaseModel
import numpy as np


class Annalisa(BaseModel):
    class_name = 'Annalisa'

    def __init__(self, name='Annalisa'):
        super().__init__(name)
        self.y_variables = ['chlide_a[mg*m^3]', 'chla[mg*m^3]', 'chlb[mg*m^3]', 'chlc1+c2[mg*m^3]',
                            'fucox[mg*m^3]', "19'hxfcx[mg*m^3]", "19'btfcx[mg*m^3]", "diadino[mg*m^3]",
                            "allox[mg*m^3]", "diatox[mg*m^3]", "zeaxan[mg*m^3]", "beta_car[mg*m^3]",
                            "peridinin[mg*m^3]"]
        self.y_variables_short = ['chlide_a', 'chla', 'chlb', 'chlc1+c2', 'fucox', "19'hxfcx", "19'btfcx",
                                  "diadino", "allox", "diatox", "zeaxan", "beta_car", "peridinin"]
        self.pft_names = ['diato', 'dino', 'crypto', 'hapto', 'green', 'prokar']
        self.psc_names = ['micro', 'nano', 'pico']

    def pft_to_array(self, phy):
        return np.array([phy[pft_name] for pft_name in self.pft_names]).T

    def psc_to_array(self, phy):
        return np.array([phy[pft_name] for pft_name in self.psc_names]).T

    def predict(self, x, y):
        TChla_ast = self.extract_y_tchla_ast(y)
        chla = self.extract_x_variables(x)
        pft = groups_pfts(chla)
        print(chla, TChla_ast)
        return self.pigments_from_pft(pft, chla, chla)

    def pigments_from_pft(self, pft, TChla, TChla_ast):
        lam = 0.95
        ans = np.empty((len(TChla), len(self.y_variables)))
        ans[:] = np.nan
        pigment = 'chla[mg*m^3]'
        ans[:, self.pigment_index(pigment)] = TChla

        pigment = "fucox[mg*m^3]"
        ans[:, self.pigment_index(pigment)] = TChla_ast * pft['diato'] / 1.6

        pigment = "peridinin[mg*m^3]"
        ans[:, self.pigment_index(pigment)] = TChla_ast * pft['dino'] / 1.67

        pigment = "19'hxfcx[mg*m^3]"
        ans[:, self.pigment_index(pigment)] = TChla_ast * pft['dino'] / 1.18 * lam

        pigment = "19'btfcx[mg*m^3]"
        ans[:, self.pigment_index(pigment)] = TChla_ast * pft['hapto'] / 0.57 * (1-lam)

        pigment = "allox[mg*m^3]"
        ans[:, self.pigment_index(pigment)] = TChla_ast * pft['crypto'] / 2.7

        pigment = "chlb[mg*m^3]"
        ans[:, self.pigment_index(pigment)] = TChla_ast * pft['green'] / 0.88

        pigment = "zeaxan[mg*m^3]"
        ans[:, self.pigment_index(pigment)] = TChla_ast * pft['prokar'] / 1.79
        return ans

    def pft_from_pigments(self, py):
        phy = self.phyto_from_pigments(py)
        return phy[self.pft_names]

    def psc_from_pigments(self, py):
        phy = self.phyto_from_pigments(py)
        return phy[self.psc_names]

    def phyto_from_pigments(self, py):
        Fuco = py["fucox[mg*m^3]"]
        Peri = py["peridinin[mg*m^3]"]
        Allo = py["allox[mg*m^3]"]
        Hex_fuco = py["19'hxfcx[mg*m^3]"]
        But_fuco = py["19'btfcx[mg*m^3]"]
        TChlb = py["chlb[mg*m^3]"]
        Zea = py["zeaxan[mg*m^3]"]

        TChla = (1.6 * Fuco + 1.67 * Peri + 1.18*Hex_fuco + 0.57*But_fuco + 2.7*Allo +
                 0.88*TChlb+1.79*Zea)

        # TChla = self.extract_y_tchla_ast(y)
        phy = pd.DataFrame()
        phy['diato'] = 1.6 * Fuco / TChla
        phy['dino'] = 1.67 * Peri / TChla
        phy['crypto'] = 2.7 * Allo / TChla
        phy['hapto'] = (1.18 * Hex_fuco + 0.57 * But_fuco) / TChla
        phy['green'] = 0.88 * TChlb / TChla
        phy['prokar'] = 1.79 * Zea / TChla
        phy['micro'] = (1.6 * Fuco + 1.67 * Peri) / TChla
        var = np.where(TChla > 0.08, 1, 12.5 * TChla)
        phy['nano'] = ( var * 1.18 * Hex_fuco + 0.57 * But_fuco + 2.7 * Allo ) / TChla
        var = np.where(TChla > 0.08, 0,  (-12.5 * TChla + 1) * 1.18 * Hex_fuco)
        phy['pico'] = (var + 0.88 * TChlb + 1.79 * Zea) / TChla

        return phy

    def predict_from_chla(self, x):
        chla = self.extract_x_variables(x)
        return groups_pfts(chla)

    def extract_x_variables(self, x):
        return x["chla_mg_m__3_"]

    def extract_y_tchla_ast(self, y):
        y_extr = self.extract_y_variables(y)

        Fuco = self.extract_pigment(y_extr, "fucox[mg*m^3]")
        Peri = self.extract_pigment(y_extr, "peridinin[mg*m^3]")
        Allo = self.extract_pigment(y_extr, "allox[mg*m^3]")
        Hex_fuco = self.extract_pigment(y_extr, "19'hxfcx[mg*m^3]")
        But_fuco = self.extract_pigment(y_extr, "19'btfcx[mg*m^3]")
        TChlb = self.extract_pigment(y_extr, "chlb[mg*m^3]")
        Zea = self.extract_pigment(y_extr, "zeaxan[mg*m^3]")

        TChla_ast = (1.6 * Fuco + 1.67 * Peri + 1.18*Hex_fuco + 0.57*But_fuco + 2.7*Allo +
                 0.88*TChlb+1.79*Zea)
        return TChla_ast


def groups_pfts(chl):
    groups = {}
    x_log = np.log10(chl)

    # PSC - micro
    a_micro = 0.0667
    b_micro = 0.1939
    c_micro = 0.2743
    d_micro = 0.2994
    micro = a_micro * (x_log ** 3) + b_micro * (x_log ** 2) + c_micro * x_log + d_micro
    groups['micro'] = micro

    # PSC - nano
    a_nano = -0.1740
    b_nano = -0.0851
    c_nano = 0.4725
    nano = a_nano * (x_log ** 2) + b_nano * x_log + c_nano
    groups['nano'] = nano

    # PSC - pico
    pico = 1 - micro - nano
    groups['pico'] = pico

    # PFT - Diatoms (fuco)
    a_diato = 0.0482
    b_diato = 0.1877
    c_diato = 0.2946
    d_diato = 0.2533
    diato = a_diato * (x_log ** 3) + b_diato * (x_log ** 2) + c_diato * x_log + d_diato
    groups['diato'] = diato

    # PFT - Dinoflagellates (peri)
    dino = micro - diato
    groups['dino'] = dino

    # PFT - Cryptophytes (allo)
    a_crypto = 0.0171
    b_crypto = 0.0667
    c_crypto = 0.1153
    d_crypto = 0.0952
    crypto = a_crypto * (x_log ** 3) + b_crypto * (x_log ** 2) + c_crypto * x_log + d_crypto
    groups['crypto'] = crypto

    # PFT - Green algae & Prochlorophytes (TChlb)
    a_green = -1.5780
    b_green = 2.1841
    c_green = 22.6833
    green = 1 / (np.exp(a_green * x_log + b_green) + c_green * x_log)
    groups['green'] = green

    # PFT - Prokaryotes (zea)
    a_prokar = 0.0664
    b_prokar = 0.1410
    c_prokar = -0.2097
    d_prokar = 0.0979
    prokar = a_prokar * (x_log ** 3) + b_prokar * (x_log ** 2) + c_prokar * x_log + d_prokar
    groups['prokar'] = prokar

    # PFT - Haptophytes (hex & but)
    hapto = 1 - micro - crypto - green - prokar
    groups['hapto'] = hapto

    # Calculate chl concentration for each group (mg m^-3)
    micro_chl = micro * chl
    nano_chl = nano * chl
    pico_chl = pico * chl

    pdiato = diato * chl
    pdino = dino * chl
    pcrypto = crypto * chl
    pgreen = green * chl
    pprokar = prokar * chl
    phapto = hapto * chl
    return groups