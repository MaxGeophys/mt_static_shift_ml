"""Project-wide runtime settings.

Edit these values before running the pipeline when you want to change
experiment settings shared by training, evaluation, export, and visualization.
"""

# Radius of the local neighbourhood used to build ML spatial features.
# Keep this independent from the baseline spatial-filter settings below.
# A value of 1 means a 3x3 neighbourhood.
ML_FEATURE_RADIUS_STEPS = 1


# Periods (seconds) whose predicted Rho correction coefficients are averaged
# to obtain one spatial K_Rho field applied to all periods during evaluation.
RHO_K_PERIODS = [500.0, 1000.0, 2000.0]

# Period (seconds) used as the reference period for spatial filtering.
FILTER_PERIOD = 10.0

# Spatial-filter neighbourhood radius in observation-grid steps.
# A value of 1 means a 3x3 neighbourhood: the central point plus
# four orthogonal and four diagonal neighbours (where available).
FILTER_RADIUS_STEPS = 1

# R0 in W = exp(-(|R| / R0) ** q), measured in observation-grid steps.
# With a value of 1, R0 is equal to one observation-grid step.
FILTER_HALF_WIDTH_STEPS = 2.0

# q in W = exp(-(|R| / R0) ** q).
FILTER_STEEPNESS = 4.0
