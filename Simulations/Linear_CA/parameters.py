"""
Parameters for 2D kinematic models:
- Constant Acceleration Model (CA): state = [x, vx, ax, y, vy, ay]
- Constant Velocity Model (CV): state = [x, vx, y, vy]
"""

import torch
import numpy as np

# Model dimensions
m = 6      # State dimension for 2D CA model (x, vx, ax, y, vy, ay)
delta_t_gen = 1e-1  # Time step

# Noise parameters
r2 = torch.tensor([0.1]).float()  # Measurement noise variance
q2 = torch.tensor([0.1]).float()  # Process noise variance

def create_CA_matrices():
    """Create matrices for 2D Constant Acceleration model"""
    # State transition matrix
    F = torch.zeros(m, m).float()

    # X-dimension kinematics
    F[0, 0] = 1
    F[0, 1] = delta_t_gen
    F[0, 2] = 0.5 * delta_t_gen**2
    F[1, 1] = 1
    F[1, 2] = delta_t_gen
    F[2, 2] = 1

    # Y-dimension kinematics (same structure as X)
    F[3, 3] = 1
    F[3, 4] = delta_t_gen
    F[3, 5] = 0.5 * delta_t_gen**2
    F[4, 4] = 1
    F[4, 5] = delta_t_gen
    F[5, 5] = 1

    # Process noise covariance
    Q = torch.zeros(m, m).float()

    # Common CA noise block
    ca_noise_block = q2 * torch.tensor([
        [1/20*delta_t_gen**5, 1/8*delta_t_gen**4, 1/6*delta_t_gen**3],
        [1/8*delta_t_gen**4, 1/3*delta_t_gen**3, 1/2*delta_t_gen**2],
        [1/6*delta_t_gen**3, 1/2*delta_t_gen**2, delta_t_gen]
    ]).float()

    # Apply to both X and Y dimensions
    Q[0:3, 0:3] = ca_noise_block
    Q[3:6, 3:6] = ca_noise_block

    return F, Q

def create_observation_matrices():
    """Create observation matrices for different configurations"""
    # CA model observation matrices
    H_identity = torch.eye(m)  # Full state observation

    # Position-only observation matrices for CA
    H_onlyPos_X = torch.zeros(1, m)
    H_onlyPos_X[0, 0] = 1.0  # Observe only X position

    H_onlyPos_Y = torch.zeros(1, m)
    H_onlyPos_Y[0, 3] = 1.0  # Observe only Y position

    H_onlyPos_both = torch.zeros(2, m)
    H_onlyPos_both[0, 0] = 1.0  # X position
    H_onlyPos_both[1, 3] = 1.0  # Y position

    # Position and velocity observation matrix for CA
    H_position_velocity = torch.zeros(4, m)
    H_position_velocity[0, 0] = 1.0  # X position
    H_position_velocity[1, 1] = 1.0  # X velocity
    H_position_velocity[2, 3] = 1.0  # Y position
    H_position_velocity[3, 4] = 1.0  # Y velocity

    return {
        'CA': {
            'full': H_identity,
            'pos_x': H_onlyPos_X,
            'pos_y': H_onlyPos_Y,
            'pos_both': H_onlyPos_both,
            'pos_vel': H_position_velocity
        },
    }

def create_noise_matrices():
    """Create measurement noise matrices for different configurations"""
    return {
        'R_4': r2 * torch.eye(4),      # For position and velocity in both dimensions
        'R_2': r2 * torch.eye(2),      # For position-only in both dimensions
        'R_onlyPos': r2                # For single position measurement (scalar)
    }

# Create all matrices
F_gen, Q_gen = create_CA_matrices()
H_matrices = create_observation_matrices()
R_matrices = create_noise_matrices()

# Extract individual matrices for backward compatibility
H_identity = H_matrices['CA']['full']
H_onlyPos_X = H_matrices['CA']['pos_x']
H_onlyPos_Y = H_matrices['CA']['pos_y']
H_onlyPos_both = H_matrices['CA']['pos_both']
H_position_velocity = H_matrices['CA']['pos_vel']

R_4 = R_matrices['R_4']
R_2 = R_matrices['R_2']
R_onlyPos = R_matrices['R_onlyPos']

def get_observation_params(observation_type):
    """Returns (H_obs, R_obs) tuple based on observation type."""
    observation_configs = {
        'position_x': (H_onlyPos_X, R_onlyPos),
        'position_y': (H_onlyPos_Y, R_onlyPos),
        'position_both': (H_onlyPos_both, R_2),
        'position_velocity': (H_position_velocity, R_4),
        'all_states': (H_identity, r2 * torch.eye(6))
    }
    return observation_configs.get(observation_type, (H_onlyPos_both, R_2))