import pandas as pd
import numpy as np
import glob
import os 
from scipy import integrate

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
            integral_flux = integrate.trapz(flux_filter, band_limits) 
            integral_bandpass = integrate.trapz(bandpass, band_limits) 
            mag_filter = -2.5*np.log10(integral_flux/integral_bandpass) + self.zero_points[filter]
            self.magnitudes[filter] = mag_filter
            self.integrals[filter] = integral_flux

class HSTMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'F275W':2.5*np.log10(3.74e-9), 
                            'F336W':2.5*np.log10(3.26e-9), 
                            'F438W':2.5*np.log10(6.73e-9), 
                            'F606W':2.5*np.log10(2.87e-9), 
                            'F814W':2.5*np.log10(1.14e-9)}
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterHST', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)

class JPASMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'uJava': 2.5*np.log10(3.27e-9),
                            'u'	   : 2.5*np.log10(4.75e-9),
                            'J0378': 2.5*np.log10(5.22e-9),
                            'J0390': 2.5*np.log10(6.82e-9),
                            'J0400': 2.5*np.log10(7.75e-9),
                            'J0410': 2.5*np.log10(7.34e-9),
                            'J0420': 2.5*np.log10(7.77e-9),
                            'J0430': 2.5*np.log10(6.42e-9),
                            'J0440': 2.5*np.log10(6.27e-9),
                            'J0450': 2.5*np.log10(6.37e-9),
                            'J0460': 2.5*np.log10(6.03e-9),
                            'J0470': 2.5*np.log10(5.7e-9 ),
                            'J0480': 2.5*np.log10(4.78e-9),
                            'gSDSS': 2.5*np.log10(5.26e-9),
                            'J0490': 2.5*np.log10(4.47e-9),
                            'J0500': 2.5*np.log10(4.73e-9),
                            'J0510': 2.5*np.log10(4.47e-9),
                            'J0520': 2.5*np.log10(4.19e-9),
                            'J0530': 2.5*np.log10(3.97e-9),
                            'J0540': 2.5*np.log10(3.77e-9),
                            'J0550': 2.5*np.log10(3.57e-9),
                            'J0560': 2.5*np.log10(3.38e-9),
                            'J0570': 2.5*np.log10(3.2e-9 ),
                            'J0580': 2.5*np.log10(3.03e-9),
                            'J0590': 2.5*np.log10(2.86e-9),
                            'J0600': 2.5*np.log10(2.73e-9),
                            'J0610': 2.5*np.log10(2.58e-9),
                            'J0620': 2.5*np.log10(2.47e-9),
                            'rSDSS': 2.5*np.log10(2.42e-9),
                            'J0630': 2.5*np.log10(2.34e-9),
                            'J0640': 2.5*np.log10(2.23e-9),
                            'J0650': 2.5*np.log10(1.98e-9),
                            'J0660': 2.5*np.log10(1.87e-9),
                            'J0670': 2.5*np.log10(1.94e-9),
                            'J0680': 2.5*np.log10(1.86e-9),
                            'J0690': 2.5*np.log10(1.78e-9),
                            'J0700': 2.5*np.log10(1.7e-9 ),
                            'J0710': 2.5*np.log10(1.62e-9),
                            'J0720': 2.5*np.log10(1.56e-9),
                            'J0730': 2.5*np.log10(1.49e-9),
                            'J0740': 2.5*np.log10(1.42e-9),
                            'J0750': 2.5*np.log10(1.37e-9),
                            'J0760': 2.5*np.log10(1.32e-9),
                            'iSDSS': 2.5*np.log10(1.29e-9),
                            'J0770': 2.5*np.log10(1.25e-9),
                            'J0780': 2.5*np.log10(1.21e-9),
                            'J0790': 2.5*np.log10(1.16e-9),
                            'J0800': 2.5*np.log10(1.12e-9),
                            'J0810': 2.5*np.log10(1.07e-9),
                            'J0820': 2.5*np.log10(1.03e-9),
                            'J0830': 2.5*np.log10(9.9e-10),
                            'J0840': 2.5*np.log10(9.56e-10),
                            'J0850': 2.5*np.log10(9.23e-10),
                            'J0860': 2.5*np.log10(8.97e-10),
                            'J0870': 2.5*np.log10(8.73e-10),
                            'J0880': 2.5*np.log10(8.56e-10),
                            'J0890': 2.5*np.log10(8.62e-10),
                            'J0900': 2.5*np.log10(8.35e-10),
                            'J0910': 2.5*np.log10(8.48e-10),
                            'J1007': 2.5*np.log10(7.14e-10)
                            }
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterOAJ_JPAS', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)
       
class JPLUSMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'uJAVA': 2.5*np.log10(3.24e-9),
                            'J0378': 2.5*np.log10(4.78e-9),
                            'J0395': 2.5*np.log10(6.9e-9),
                            'J0410': 2.5*np.log10(7.61e-9),
                            'J0430': 2.5*np.log10(6.68e-9),
                            'gSDSS': 2.5*np.log10(5.2e-9),
                            'J0515': 2.5*np.log10(4.37e-9),
                            'rSDSS': 2.5*np.log10(2.41e-9),
                            'J0660': 2.5*np.log10(1.87e-9),
                            'iSDSS': 2.5*np.log10(1.29e-9),
                            'J0861': 2.5*np.log10(9.04e-10),
                            'zSDSS': 2.5*np.log10(8.35e-10)
                            }
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterOAJ_JPLUS', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)

class LSSTMagnitudeFactory(MagnitudeFactory):
    def __init__(self):
        super().__init__()
        self.zero_points = {'u': 2.5*np.log10(4.51e-9),
                            'g': 2.5*np.log10(5.23e-9),
                            'r': 2.5*np.log10(2.45e-9),
                            'i': 2.5*np.log10(1.36e-9),
                            'z': 2.5*np.log10(8.99e-10),
                            'y': 2.5*np.log10(6.86e-10)
                            }
        module_path = os.path.dirname(__file__)
        filter_path = os.path.join(module_path,'transmission_curves', 'filterLSST', '*.dat')
        files = glob.glob(filter_path)
        self.load_filters(files)
