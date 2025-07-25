import pandas as pd
import numpy as np
import glob
import os 
from scipy import integrate
from astropy.io import fits

class MagnitudeFactory():
    def __init__(self):
        self.transmission_curves = {}
        self.zero_points = None
        self.wavelength = None
        self.flux = None
        self.magnitudes = {}
        self.integrals = {}

    def load_filters(self, files):
        for file in files:
            filter_name = file.split('.')[-2]
            filter =  pd.read_csv(file, skip_blank_lines=True, 
                            comment='#', sep='\s+', 
                            names=['wavelength', 'flux'],engine='python')
            self.transmission_curves[filter_name] = filter

    def load_spectrum(self, path):
        spectrum =  pd.read_csv(path, skip_blank_lines=True, 
                            comment='#', sep='\s+', header=0,
                            names=['wavelength', 'flux'],engine='python')
        self.wavelength = spectrum['wavelength']
        self.flux = spectrum['flux']

    def load_fits_spectrum(self, path):
        hdul = fits.open(path)
        self.flux = hdul[0].data
        hdr = hdul[0].header
        delta = hdr['CDELT1']
        start = hdr['CRVAL1']
        number = len(self.flux)
        self.wavelength = np.linspace(10**(start)*10, 10**(start+(number-1)*delta)*10, number)
        
    def load_fits_spec_XSLstar(self, path):
        hdul = fits.open(path)
        data = hdul[1].data
        self.wavelength = data.field(0)
        self.flux = data.field(1)
 
    def compute_magnitude(self):
        if self.flux is None:
            print('Spectrum not loaded yet.')
            return None
        if self.zero_points is None:
            print('Zero Points not defined yet.')
            return None
        if self.transmission_curves == {}:
            print('Filter transmission curves not defined yet.')
            return None
        for filter, curve in self.transmission_curves.items():
            min_wavelength_filter = min(curve['wavelength'])
            max_wavelength_filter = max(curve['wavelength'])
            band_limits = self.wavelength[(self.wavelength >= min_wavelength_filter) & (self.wavelength <= max_wavelength_filter)]
            bandpass = np.interp(band_limits, curve['wavelength'], curve['flux'])
            spectral_flux = self.flux[(self.wavelength >= min_wavelength_filter) & (self.wavelength <= max_wavelength_filter)]
            flux_filter = (bandpass * spectral_flux)
            integral_flux = np.trapz(flux_filter, band_limits) 
            integral_bandpass = np.trapz(bandpass, band_limits) 
            mag_filter = -2.5*np.log10(integral_flux/integral_bandpass) #+ self.zero_points[filter]
            self.magnitudes[filter] = mag_filter
            self.integrals[filter] = integral_flux

class HSTMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'F275W':2.5*np.log10(1.48298e-8), 
                            'F336W':2.5*np.log10(9.67405e-9), 
                            'F438W':2.5*np.log10(5.81602e-9), 
                            'F606W':2.5*np.log10(3.10404e-9), 
                            'F814W':2.5*np.log10(1.68166e-9)}
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterHST', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)

class JPASMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'uJava': 2.5*np.log10(8.66112e-9),
                            'u'	   : 2.5*np.log10(7.96752e-9),
                            'J0378': 2.5*np.log10(7.55142e-9),
                            'J0390': 2.5*np.log10(7.14084e-9),
                            'J0400': 2.5*np.log10(6.77949e-9),
                            'J0410': 2.5*np.log10(6.42469e-9),
                            'J0420': 2.5*np.log10(6.14183e-9),
                            'J0430': 2.5*np.log10(5.85763e-9),
                            'J0440': 2.5*np.log10(5.60069e-9),
                            'J0450': 2.5*np.log10(5.33976e-9),
                            'J0460': 2.5*np.log10(5.12449e-9),
                            'J0470': 2.5*np.log10(4.91721e-9),
                            'J0480': 2.5*np.log10(4.70778e-9),
                            'gSDSS': 2.5*np.log10(4.78826e-9),
                            'J0490': 2.5*np.log10(4.52121e-9),
                            'J0500': 2.5*np.log10(4.34415e-9),
                            'J0510': 2.5*np.log10(4.18154e-9),
                            'J0520': 2.5*np.log10(4.01268e-9),
                            'J0530': 2.5*np.log10(3.86929e-9),
                            'J0540': 2.5*np.log10(3.73668e-9),
                            'J0550': 2.5*np.log10(3.59574e-9),
                            'J0560': 2.5*np.log10(3.46539e-9),
                            'J0570': 2.5*np.log10(3.33849e-9),
                            'J0580': 2.5*np.log10(3.2228e-9),
                            'J0590': 2.5*np.log10(3.10583e-9),
                            'J0600': 2.5*np.log10(3.01339e-9),
                            'J0610': 2.5*np.log10(2.90854e-9),
                            'J0620': 2.5*np.log10(2.82339e-9),
                            'rSDSS': 2.5*np.log10(2.79082e-9),
                            'J0630': 2.5*np.log10(2.73145e-9),
                            'J0640': 2.5*np.log10(2.64874e-9),
                            'J0650': 2.5*np.log10(2.57046e-9),
                            'J0660': 2.5*np.log10(2.49319e-9),
                            'J0670': 2.5*np.log10(2.41543e-9),
                            'J0680': 2.5*np.log10(2.34662e-9),
                            'J0690': 2.5*np.log10(2.27565e-9),
                            'J0700': 2.5*np.log10(2.2161e-9),
                            'J0710': 2.5*np.log10(2.14614e-9),
                            'J0720': 2.5*np.log10(2.09281e-9),
                            'J0730': 2.5*np.log10(2.03589e-9),
                            'J0740': 2.5*np.log10(1.97923e-9),
                            'J0750': 2.5*np.log10(1.93425e-9),
                            'J0760': 2.5*np.log10(1.88486e-9),
                            'iSDSS': 2.5*np.log10(1.85704e-9),
                            'J0770': 2.5*np.log10(1.82615e-9),
                            'J0780': 2.5*np.log10(1.78546e-9),
                            'J0790': 2.5*np.log10(1.74052e-9),
                            'J0800': 2.5*np.log10(1.69785e-9),
                            'J0810': 2.5*np.log10(1.65328e-9),
                            'J0820': 2.5*np.log10(1.61231e-9),
                            'J0830': 2.5*np.log10(1.57298e-9),
                            'J0840': 2.5*np.log10(1.53871e-9),
                            'J0850': 2.5*np.log10(1.50343e-9),
                            'J0860': 2.5*np.log10(1.47247e-9),
                            'J0870': 2.5*np.log10(1.43518e-9),
                            'J0880': 2.5*np.log10(1.39961e-9),
                            'J0890': 2.5*np.log10(1.37102e-9),
                            'J0900': 2.5*np.log10(1.34291e-9),
                            'J0910': 2.5*np.log10(1.3157e-9),
                            'J1007': 2.5*np.log10(1.18348e-9)
                            }
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterOAJ_JPAS', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)
       
class JPLUSMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'uJAVA': 2.5*np.log10(8.71928e-9),
                            'J0378': 2.5*np.log10(7.60886e-9),
                            'J0395': 2.5*np.log10(7.01445e-9),
                            'J0410': 2.5*np.log10(6.45002e-9),
                            'J0430': 2.5*np.log10(5.87821e-9),
                            'gSDSS': 2.5*np.log10(4.74459e-9),
                            'J0515': 2.5*np.log10(4.11879e-9),
                            'rSDSS': 2.5*np.log10(2.78078e-9),
                            'J0660': 2.5*np.log10(2.496e-9),
                            'iSDSS': 2.5*np.log10(1.85704e-9),
                            'J0861': 2.5*np.log10(1.46822e-9),
                            'zSDSS': 2.5*np.log10(1.35449e-9)
                            }
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterOAJ_JPLUS', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)

class LSSTMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'u': 2.5*np.log10(8.03787e-9),
                            'g': 2.5*np.log10(4.7597e-9),
                            'r': 2.5*np.log10(2.8156e-9),
                            'i': 2.5*np.log10(1.91864e-9),
                            'z': 2.5*np.log10(1.44312e-9),
                            'y': 2.5*np.log10(1.14978e-9)
                            }
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterLSST', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)

class EuclidMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'vis': 2.5*np.log10(2.15734e-9),
                            'Y': 2.5*np.log10(9.35783e-10),
                            'J': 2.5*np.log10(5.86749e-10),
                            'H':2.5*np.log10(3.49475e-10)
                            }
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterEuclid', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)
   
