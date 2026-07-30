import numpy as np

# Spline interpolation logic used in typical auto analyzers
# Based on logit-log or 4-parameter logistic curve?
# Actually, the user says:
# "検量線モード：折れ線(他には、スプラインや直線などがあります)"
# "折れ線" means broken-line (piecewise linear).
# Let's use numpy's interp which is exactly piecewise linear interpolation
# We saw: Orig: 23.6, Calc: 23.37
# Orig: 6.6, Calc: 6.43
# Orig: 8.3, Calc: 8.32
# The difference might be due to rounding of y (proc_val). Let's see original raw calc
