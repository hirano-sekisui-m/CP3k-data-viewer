import numpy as np
from scipy.interpolate import interp1d

y_points = [2.6, 9.7, 14.3, 19.9, 29.3, 51.1]
x_points = [0.0, 5.3, 14.0, 30.4, 56.7, 139.8]

# Spline interpolation
f_spline = interp1d(y_points, x_points, kind='cubic', fill_value="extrapolate")
print("Cubic spline:")
print(f"Req 0002 (y=17.5): {f_spline(17.5)}")
print(f"Req 0003 (y=10.3): {f_spline(10.3)}")
print(f"Req 0004 (y=11.3): {f_spline(11.3)}")
