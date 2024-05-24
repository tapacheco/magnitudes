import pandas as pd
import numpy as np
import spectres 
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
        if not self.flux:
            print('Spectrum not loaded yet.')
            return None
        if not self.zero_points:
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
        