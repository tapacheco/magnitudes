import tap_magnitudes.magnitudeFactory as mag

magFact = mag.HSTMagnitudeFactory()
magFact.load_spectrum('NGC2808_total_ssp.dat')
magFact.compute_magnitude()

print(magFact.magnitudes)