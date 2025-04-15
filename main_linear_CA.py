import torch
import glob
import os
from datetime import datetime
# Override torch.Tensor.__str__ to hide device info
original_str = torch.Tensor.__str__
def clean_str(self):
    if self.numel() == 1:
        return f"{self.item():.4f}"
    return original_str(self)
torch.Tensor.__str__ = clean_str

from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
import Simulations.utils as utils
from Simulations.Linear_CA.parameters import F_gen, Q_gen, R_2, R_onlyPos, m, get_observation_params
# F_gen: State transition matrix for CA model
# Q_gen: Process noise covariance matrix for CA model
# R_2, R_onlyPos: Measurement noise covariance matrices
# m: State dimension for CA model
# get_observation_params: Function to get observation matrices and noise based on type
from Filters.KalmanFilter_test import KFTest
from KNet.KalmanNet_nn import KalmanNetNN
from Pipelines.Pipeline import Pipeline_KN as Pipeline
from trajectory_plotter import TrajectoryPlotter

################
### Get Time ###
################
# Record current time for logging and saving results
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'KNet/'  # Directory to save results

print("Pipeline Start")
####################################
### Generative Parameters For CA ###
####################################
# Get default configuration settings
args = config.general_settings()

### Dataset parameters
args.N_E = 4000  # Number of training trajectories
args.N_CV = 400  # Number of cross-validation trajectories
args.N_T = 100   # Number of test trajectories
offset = 0       # Initial condition offset for dataset
# Random initialization settings for different datasets
args.randomInit_train = True  # Use random initial states for training data
args.randomInit_cv = True   # Use random initial states for validation data
args.randomInit_test = True  # Use random initial states for test data

args.T = 100       # Length of each trajectory (time steps)
args.T_test = 50  # Length of test trajectories

### Training parameters
# If True: KalmanNet knows the random initial state during training
# If False: KalmanNet must learn to handle unknown initial states
KnownRandInit_train = True
KnownRandInit_cv = True
KnownRandInit_test = True
args.use_cuda = True  # Use GPU acceleration if available
args.n_steps = 1000   # Number of training steps/iterations default = 4000
args.n_batch = 2048     # Batch size for training
args.lr = 6e-4        # Learning rate
args.wd = 1e-3        # Weight decay (L2 regularization)

# Add early stopping parameters
args.use_early_stopping = True  # Enable early stopping
args.early_stopping_patience = 60  # Stop after 20 epochs without improvement
args.early_stopping_min_delta = 0.001  # Minimum change to qualify as improvement
# Add learning rate scheduler parameters
args.use_lr_scheduler = True  # Enable learning rate scheduling
args.lr_scheduler_gamma = 0.1  # Factor to multiply learning rate when reducing
args.lr_scheduler_patience = 15  # Epochs with no improvement after which LR is reduced
args.lr_scheduler_threshold = 0.001  # Minimum change to qualify as improvement

if args.use_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                         'cpu')
    if device.type == 'cpu':
        args.use_cuda = False
        print("No compatible GPU found. Using CPU.")
    else:
        print(f"Using {device} device")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Replace the existing initial state configuration section with:

####################################
### Initial State Configuration ####
####################################
initial_state_config = {
    # Mean values for initial state
    'means': {
        'position': {'x': 0.0, 'y': 0.0},        # Starting position (x,y)
        'velocity': {'mag': 25.0},               # Initial velocity magnitude
        'acceleration': {'mag': 2.0}             # Initial acceleration magnitude
    },
    # Standard deviations for initial state
    'std_devs': {
        'position': 1e-6,       # Very small position variance (start near origin)
        'velocity': 10.0,       # Velocity component standard deviation
        'acceleration': 2.0     # Acceleration component standard deviation
    }
}

# Create initial state covariance matrix
if (args.randomInit_train or args.randomInit_cv or args.randomInit_test):
    m2x_0_gen = torch.zeros(m, m)
    # Position components
    m2x_0_gen[0, 0] = initial_state_config['std_devs']['position'] ** 2  # x position
    m2x_0_gen[3, 3] = initial_state_config['std_devs']['position'] ** 2  # y position
    # Velocity components
    m2x_0_gen[1, 1] = initial_state_config['std_devs']['velocity'] ** 2  # x velocity
    m2x_0_gen[4, 4] = initial_state_config['std_devs']['velocity'] ** 2  # y velocity
    # Acceleration components
    m2x_0_gen[2, 2] = initial_state_config['std_devs']['acceleration'] ** 2  # x acceleration
    m2x_0_gen[5, 5] = initial_state_config['std_devs']['acceleration'] ** 2  # y acceleration
else:
    m2x_0_gen = torch.zeros(m, m)

# Set standard deviation for filter initialization based on KnownRandInit settings
if (KnownRandInit_train or KnownRandInit_cv or KnownRandInit_test):
    # Use zero uncertainty when we know the initial state exactly
    m2x_0 = torch.zeros(m, m)  # Zero covariance matrix

#############################
###  Dataset Generation   ###
#############################
### Loss calculation settings
Loss_On_AllState = False  # If False: only calculate test loss on position component
Train_Loss_On_AllState = False  # If False: only calculate training loss on position component, default = True
observation_type = 'position_velocity'
H_obs, R_obs = get_observation_params(observation_type)

# Data file paths
DatafolderName = 'Simulations/Linear_CA/data/'
DatafileName = 'decimated_dt1e-2_T100_r0_randnInit.pt'



####################
### System Model ###
####################
# Initialize state vector with zeros
m1x_0 = torch.zeros(m)

# Generation model (CA - Constant Acceleration)
sys_model_gen = SystemModel(F_gen, Q_gen, H_obs, R_obs, args.T, args.T_test)
sys_model_gen.InitSequence(m1x_0, m2x_0_gen)  # Initialize with x0 and P0

# Standard CA model for both KF and KalmanNet
sys_model_KF = SystemModel(F_gen, Q_gen, H_obs, R_obs, args.T, args.T_test)
sys_model_KN = SystemModel(F_gen, Q_gen, H_obs, R_obs, args.T, args.T_test)

# Generate synthetic dataset
print("Start Data Gen")
utils.DataGen(args, sys_model_gen, DatafolderName+DatafileName, initial_state_config)

# Load the generated data
print("Load Original Data")
[train_input, train_target, cv_input, cv_target, test_input, test_target,
 train_init, cv_init, test_init] = torch.load(DatafolderName+DatafileName,
                                             map_location=device,
                                             weights_only=False)

# Print data dimensions for verification
print("Data Shape")
print("testset state x size:", test_target.size())
print("testset observation y size:", test_input.size())
print("trainset state x size:", train_target.size())
print("trainset observation y size:", train_input.size())
print("cvset state x size:", cv_target.size())
print("cvset observation y size:", cv_input.size())

print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter")
kf_args = {
    'allStates': Loss_On_AllState
}

# Get true initial states from test_target
true_init_states = test_target[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, state_dim, 1]

# Create zero covariance for initialization
zero_cov = torch.zeros(test_target.shape[0], sys_model_KF.m, sys_model_KF.m).to(test_target.device)

# Print debug info
print("Using true initial states for KF initialization")
print("Initial state shape:", true_init_states.shape)
print("First sample initial state:", true_init_states[0].squeeze())
print("First ground truth state:", test_target[0, :, 0])

# Run KF test with true initial states
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(
    args, sys_model_KF, test_input, test_target,
    allStates=Loss_On_AllState,
    randomInit=True,  # Enable custom initialization
    test_init=true_init_states  # Pass true initial states
)

MSE_KF_dB_std = torch.std(10 * torch.log10(MSE_KF_linear_arr))
print(f"KF-MSE Test: {MSE_KF_dB_avg:.4f} [dB]")
print(f"KF-STD Test: {MSE_KF_dB_std:.4f} [dB]")

##########################
### Evaluate KalmanNet ###
##########################
# Build the KalmanNet neural network
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model_KN, args)
print("Number of trainable parameters for KNet pass 1:",
      sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))

# Set up the training and evaluation pipeline
KNet_Pipeline = Pipeline(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model_KN)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)

# Train KalmanNet with true initial states
randomInit_train = True  # Enable custom initialization
print(f"Train KNet with True Initial States")
print(f"Train Loss on All States (if false, loss on position only): {Train_Loss_On_AllState}")

# Extract true initial states for training and cross-validation
train_init_states = train_target[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, state_dim, 1]
cv_init_states = cv_target[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, state_dim, 1]

# Modify training arguments to use true initial states
train_args = {
    'MaskOnState': not Train_Loss_On_AllState,
    'randomInit': randomInit_train,
    'train_init': train_init_states,
    'cv_init': cv_init_states
}

[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = \
    KNet_Pipeline.NNTrain(sys_model_KN, cv_input, cv_target, train_input, train_target,
                          path_results, **train_args)

# Test KalmanNet with true initial states
randomInit_test = True  # Enable custom initialization
print(f"Test KNet with True Initial States")
print(f"Compute Loss on All States (if false, loss on position only): {Loss_On_AllState}")

# Modify test arguments to use true initial states
test_args = {
    'MaskOnState': not Loss_On_AllState,
    'randomInit': randomInit_test,
    'test_init': true_init_states  # Use the same true init states as for KF
}

[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = \
    KNet_Pipeline.NNTest(sys_model_KN, test_input, test_target, path_results, **test_args)

####################
### Plot results ###
####################

# Set up file paths for saving plots
PlotfolderName = "Figures/Linear_CA/"

# Initialize plotting object
Plot = TrajectoryPlotter(PlotfolderName, "KalmanNet CA Model")
print("Plot initialized")

# Number of test samples to plot
num_plots = 10  # Change this if you want fewer samples

# Create the output directory if it doesn't exist
os.makedirs(PlotfolderName, exist_ok=True)

# Clear existing figures in the folder (optional)
for file in glob.glob(f"{PlotfolderName}*.png"):
    os.remove(file)
print("Cleared existing figures")

for i in range(num_plots):
    if i >= test_target.shape[0]:
        print(
            f"Warning: Not enough test trajectories. Requested trajectory {i + 1} but only have {test_target.shape[0]}")
        break
    print(f"Generating plots for test sample {i + 1}")

    # Extract single test sample
    single_test_target = test_target[i:i + 1]
    single_KF_out = KF_out[i:i + 1]
    single_KNet_out = KNet_out[i:i + 1]

    # File prefix for this sample
    prefix = f"{i + 1}_"

    # 1. Position norm vs time plot
    Plot.plot_component_norm_vs_time(
        test_target=single_test_target,
        knet_out=single_KNet_out,
        kf_out=single_KF_out,
        file_name_prefix=f"{PlotfolderName}{prefix}",
        title_prefix=f"Test Sample {i + 1}"
    )

    # 2. XY plane trajectory
    Plot.plot_2D_trajectory(
        test_target=single_test_target,
        kf_out=single_KF_out,
        knet_out=single_KNet_out,
        file_name=f"{PlotfolderName}{prefix}2D_trajectory.png",
        title=f"Test Sample {i + 1}: 2D Trajectory",
        show_ground_truth_points=True  # Add this parameter to show ground truth points
    )

print(f"Generated plots for {min(num_plots, test_target.shape[0])} test samples")
print("Each sample has 2 plots: position norm and velocity norm, plus 2D trajectory")