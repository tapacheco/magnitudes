import tap_magnitudes.magnitudeFactory as mag

magFact = mag.JPASMagnitudeFactory()
magFact.load_fits_spectrum('~/Documents/spectraStellarGrids/specsCoelho/starsCoelho14_SED/t03000_g+0.0_m01p04_sed.fits')
magFact.compute_magnitude(input_type='f_lambda', output_type='ab_mag')

print(magFact.magnitudes)