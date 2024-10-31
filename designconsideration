''' checking the nominal speed and critical speed'''

# Given critical speeds in rad/s for slopes 1 and 1.5
w1_slope_1 = np.array([489.61838543, 650.48151527, 1164.0800074, 1261.06522761, 1316.49076465,
                       1780.92776273, 1865.37284085])
w1_slope_1_5 = np.array([341.75507041, 414.41490517, 838.24910613, 863.63156107, 871.57926544,
                         1176.51563064, 1261.79053414, 1443.93244425, 1896.61819437, 2200.46054897])

# Convert nominal speed to rad/s and calculate ranges
nominal_speed_rpm = 5000
nominal_speed_rad_s = nominal_speed_rpm * 2 * np.pi / 60  # 523.33 rad/s

# Define non-acceptable ranges
low_range = 0.5 * nominal_speed_rad_s  # Lower bound
upper_range_1 = 0.75 * nominal_speed_rad_s  # Lower limit for upper bound
upper_range_2 = 1.1 * nominal_speed_rad_s  # Upper limit for upper bound

# Add 10% margin
low_range_margin = (1 - 0.1) * low_range, (1 + 0.1) * low_range
upper_range_margin = (1 - 0.1) * upper_range_1, (1 + 0.1) * upper_range_2

# Display range information for review
print("Non-acceptable ranges:")
print(f"Low range (with margin): {low_range_margin}")
print(f"Upper range (with margin): {upper_range_margin}")

# Function to check if a speed falls within the non-acceptable range
def is_non_acceptable(speed, ranges):
    return any(lower <= speed <= upper for (lower, upper) in ranges)

# Define ranges for easy comparison
non_acceptable_ranges = [low_range_margin, upper_range_margin]

# Find non-acceptable critical speeds for slope 1
non_acceptable_slope_1 = [speed for speed in w1_slope_1 if is_non_acceptable(speed, non_acceptable_ranges)]
print("Non-acceptable critical speeds for slope 1:", non_acceptable_slope_1)

# Find non-acceptable critical speeds for slope 1.5
non_acceptable_slope_1_5 = [speed for speed in w1_slope_1_5 if is_non_acceptable(speed, non_acceptable_ranges)]
print("Non-acceptable critical speeds for slope 1.5:", non_acceptable_slope_1_5)


''' adding gyroscopic effect'''
# Assuming rotor_lp (low-pressure rotor) is defined and represents the low-pressure rotor model

# Calculate nominal speed in rad/s for the low-pressure rotor
Omega_lp = nominal_speed_rpm * 2 * np.pi / 60  # Convert 5000 RPM to rad/s

# Get gyroscopic matrix for the low-pressure rotor at its nominal speed
G_lp = Omega_lp * rotor_lp.G()  # Scale by rotational speed

# Define the equation of motion with the gyroscopic effect included
def build_EOM_with_gyro(rotor, G_lp):
    """Builds the EOM with gyroscopic effects for the low-pressure rotor only."""
    # Mass and stiffness matrices for low-pressure rotor
    M = rotor.M()
    K = rotor.K()

    # Block matrices with gyroscopic matrix
    A_11 = K
    A_12 = G_lp
    A_21 = -M
    A_22 = np.zeros(M.shape)

    B_11 = G_lp
    B_12 = M
    B_21 = M
    B_22 = np.zeros(M.shape)

    # Construct the full A and B matrices
    A = np.block([[A_11, A_12], [A_21, A_22]])
    B = np.block([[B_11, B_12], [B_21, B_22]])

    return A, B

# Solve the eigenvalue problem for the low-pressure rotor with gyroscopic effect
def solve_sys_with_gyro(rotor, G_lp):
    """Solve the generalized eigenvalue problem for the low-pressure rotor with gyro effect."""
    A, B = build_EOM_with_gyro(rotor, G_lp)
    eigval, eigvec = eig(A, -B)
    
    return eigval, eigvec

# Run the analysis
eigval, eigvec = solve_sys_with_gyro(rotor_lp, G_lp)

# Convert eigenvalues to natural frequencies in Hz
omega_n = np.abs(eigval[0:20:2])  # Extract first 10 natural frequencies
freq_n = omega_n / (2 * np.pi)
print("Natural frequencies with gyroscopic effect (Hz):", freq_n)

'''check if the adjustment fix the problem of critical speed'''
import numpy as np
from scipy.linalg import eig
import ross as rs

# Define nominal speed in rad/s for low-pressure rotor
nominal_speed_rpm = 5000
Omega_lp = nominal_speed_rpm * 2 * np.pi / 60  # Convert to rad/s

# Assuming rotor_lp represents the low-pressure rotor model
G_lp = Omega_lp * rotor_lp.G()  # Gyroscopic matrix for low-pressure rotor at nominal speed

# Define the equation of motion with gyroscopic effects included
def build_EOM_with_gyro(rotor, G_lp):
    M = rotor.M()  # Mass matrix
    K = rotor.K()  # Stiffness matrix

    # Create the block matrices
    A_11 = K
    A_12 = G_lp
    A_21 = -M
    A_22 = np.zeros(M.shape)

    B_11 = G_lp
    B_12 = M
    B_21 = M
    B_22 = np.zeros(M.shape)

    # Construct the full A and B matrices
    A = np.block([[A_11, A_12], [A_21, A_22]])
    B = np.block([[B_11, B_12], [B_21, B_22]])

    return A, B

# Solve the eigenvalue problem
def solve_sys_with_gyro(rotor, G_lp):
    A, B = build_EOM_with_gyro(rotor, G_lp)
    eigval, eigvec = eig(A, -B)
    return eigval, eigvec

# Run the analysis
eigval, eigvec = solve_sys_with_gyro(rotor_lp, G_lp)

# Extract natural frequencies and convert to Hz
omega_n = np.abs(eigval[0:20:2])  # Extract first 10 natural frequencies (rad/s)
freq_n = omega_n / (2 * np.pi)    # Convert to Hz
print("Natural frequencies with gyroscopic effect (Hz):", freq_n)

