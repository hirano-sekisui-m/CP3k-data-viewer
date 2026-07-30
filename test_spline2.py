import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

y_points = [2.6, 9.7, 14.3, 19.9, 29.3, 51.1]
x_points = [0.0, 5.3, 14.0, 30.4, 56.7, 139.8]

f_akima = Akima1DInterpolator(y_points, x_points)
f_pchip = PchipInterpolator(y_points, x_points)
print("Akima:")
print(f"Req 0002 (y=17.5): {f_akima(17.5)}")
print(f"Req 0003 (y=10.3): {f_akima(10.3)}")
print(f"Req 0004 (y=11.3): {f_akima(11.3)}")

print("Pchip:")
print(f"Req 0002 (y=17.5): {f_pchip(17.5)}")
print(f"Req 0003 (y=10.3): {f_pchip(10.3)}")
print(f"Req 0004 (y=11.3): {f_pchip(11.3)}")
