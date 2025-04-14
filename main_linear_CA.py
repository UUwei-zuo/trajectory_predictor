import torch
from datetime import datetime
import os

# Override torch.Tensor.__str__ to hide device info
original_str = torch.Tensor.__str__
def clean_str(self):
    if self.numel() == 1:
        return f"{self.item():.4f}"
    return original_str(self)
torch.Tensor.__str__ = clean_str

from Simulations.Linear_sysmdl import SystemModel  # Base class for linear system models
import Simulations.config as config  # Configuration settings
import Simulations.utils as utils  # Utility functions
# Import model parameters for Constant Acceleration (CA) and Constant Velocity (CV) models
from Simulations.Linear_CA.parameters import F_gen, F_CV, Q_gen, Q_CV, R_2, R_onlyPos, m, m_cv, get_observation_params, H_onlyPos_CV
# F_gen, F_CV: State transition matrices for CA and CV models
# Q_gen, Q_CV: Process noise covariance matrices for CA and CV models
# R_2, R_onlyPos: Measurement noise covariance matrices
# m, m_cv: State dimensions for CA and CV models
# get_observation_params: Function to get observation matrices and noise based on type
# H_onlyPos_CV: Position-only observation matrix for CV model

# Import Kalman Filter implementation for testing
from Filters.KalmanFilter_test import KFTest

# Import KalmanNet neural network model
from KNet.KalmanNet_nn import KalmanNetNN

# Import training and evaluation pipeline
from Pipelines.Pipeline import Pipeline_KN as Pipeline

# Import plotting utilities
from Plot import Plot_extended as Plot

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
args.T_test = 20  # Length of test trajectories

### Training parameters
# If True: KalmanNet knows the random initial state during training
# If False: KalmanNet must learn to handle unknown initial states
KnownRandInit_train = True
KnownRandInit_cv = True
KnownRandInit_test = True
args.use_cuda = True  # Use GPU acceleration if available
args.n_steps = 1000   # Number of training steps/iterations default = 4000
args.n_batch = 1024     # Batch size for training
args.lr = 5e-4        # Learning rate
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

# Set standard deviation for initial state generation based on randomInit settings
if (args.randomInit_train or args.randomInit_cv or args.randomInit_test):
    position_std = 1e-6  # Standard deviation for initial position
    velocity_std = 10.0  # Standard deviation for initial velocity
    acceleration_std = 2.0  # Standard deviation for initial acceleration
    # Create diagonal covariance matrix with individual variances
    m2x_0_gen = torch.diag(torch.tensor([position_std, velocity_std, acceleration_std])**2)
else:
    # Zero standard deviation = deterministic initial state
    m2x_0_gen = torch.zeros(m, m)
# Set standard deviation for filter initialization based on KnownRandInit settings
if (KnownRandInit_train or KnownRandInit_cv or KnownRandInit_test):
    # If we know the initial state, no uncertainty in filter initialization
    m2x_0 = torch.zeros(m, m)
    m2x_0_cv = torch.zeros(m_cv, m_cv)
else:
    # If we don't know the initial state, add uncertainty in filter initialization
    # You can also customize these if needed
    filter_position_std = 1.0
    filter_velocity_std = 0.5
    filter_acceleration_std = 0.1
    m2x_0 = torch.diag(torch.tensor([
        filter_position_std ** 2,
        filter_velocity_std ** 2,
        filter_acceleration_std ** 2
    ]))
    # For CV model
    m2x_0_cv = torch.diag(torch.tensor([
        filter_position_std ** 2,
        filter_velocity_std ** 2
    ]))
# Define custom mean values for initial states
# These will be used as the mean for random initialization
initial_position_mean = 0.0  # Mean initial position
initial_velocity_mean = 25.0  # Mean initial velocity (e.g., 5 units/sec)
initial_acceleration_mean = 0.0  # Mean initial acceleration (e.g., 0.0 units/sec²)
# Initialize state vectors with custom means
m1x_0 = torch.tensor([initial_position_mean, initial_velocity_mean, initial_acceleration_mean])
m1x_0_cv = torch.tensor([initial_position_mean, initial_velocity_mean])  # For CV model

#############################
###  Dataset Generation   ###
#############################
### Loss calculation settings
Loss_On_AllState = False  # If False: only calculate test loss on position component
Train_Loss_On_AllState = False  # If False: only calculate training loss on position component, default = True
# Define state component weights for loss calculation (only used when Train_Loss_On_AllState=True)
position_weight = 2.0    # Weight for position component in loss
velocity_weight = 2.0    # Weight for velocity component in loss
acceleration_weight = 1.0  # Weight for acceleration component in loss
state_weights = torch.tensor([position_weight, velocity_weight, acceleration_weight])
CV_model = False  # If True: use Constant Velocity model, else: use Constant Acceleration model
observation_type = 'position_velocity'
H_obs, R_obs = get_observation_params(observation_type)

# Data file paths
DatafolderName = 'Simulations/Linear_CA/data/'
DatafileName = 'decimated_dt1e-2_T100_r0_randnInit.pt'

####################
### System Model ###
####################
# Generation model (CA - Constant Acceleration)
# This model is used to generate the synthetic data with the selected observation matrix
sys_model_gen = SystemModel(F_gen, Q_gen, H_obs, R_obs, args.T, args.T_test)
sys_model_gen.InitSequence(m1x_0, m2x_0_gen)  # Initialize with x0 and P0

if CV_model:
    if observation_type == 'position_only':
        H_obs_CV = H_onlyPos_CV  # Use the predefined matrix for position-only
        R_obs_CV = R_onlyPos  # Single measurement noise
    else:
        H_obs_CV = torch.eye(2).float()  # Observe both position and velocity
        R_obs_CV = R_2  # Use appropriate noise covariance for 2D measurements

    # Initialize system models for CV model
    sys_model_KF = SystemModel(F_CV, Q_CV, H_obs_CV, R_obs_CV, args.T, args.T_test)
    sys_model_KF.InitSequence(m1x_0_cv, m2x_0_cv)
    sys_model_KN = sys_model_KF  # Same model for KalmanNet in CV case
else:
    # Standard CA model for both KF and KalmanNet
    sys_model_KF = SystemModel(F_gen, Q_gen, H_obs, R_obs, args.T, args.T_test)
    sys_model_KF.InitSequence(m1x_0, m2x_0)
    sys_model_KN = sys_model_KF  # Same model for both

# Generate synthetic dataset
print("Start Data Gen")
utils.DataGen(args, sys_model_gen, DatafolderName+DatafileName)

# Load the generated data
print("Load Original Data")
[train_input, train_target, cv_input, cv_target, test_input, test_target,
 train_init, cv_init, test_init] = torch.load(DatafolderName+DatafileName,
                                             map_location=device,
                                             weights_only=False)
# train_input: observations (y) for training
# train_target: true states (x) for training
# cv_input: observations (y) for cross-validation
# cv_target: true states (x) for cross-validation
# test_input: observations (y) for testing
# test_target: true states (x) for testing
# *_init: initial states for each dataset

# If using CV model, truncate the state vectors to only include position and velocity
if CV_model:  # Set state as (p,v) instead of (p,v,a)
   train_target = train_target[:,0:m_cv,:]
   train_init = train_init[:,0:m_cv]
   cv_target = cv_target[:,0:m_cv,:]
   cv_init = cv_init[:,0:m_cv]
   test_target = test_target[:,0:m_cv,:]
   test_init = test_init[:,0:m_cv]

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
# Run standard Kalman Filter as baseline
print("Evaluate Kalman Filter")
kf_args = {
    'allStates': Loss_On_AllState
}
if args.randomInit_test and KnownRandInit_test:
    kf_args.update({'randomInit': True, 'test_init': test_init})

[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(
    args, sys_model_KF, test_input, test_target, **kf_args)

MSE_KF_dB_std = torch.std(10 * torch.log10(MSE_KF_linear_arr))
print(f"KF-MSE Test: {MSE_KF_dB_avg:.4f} [dB]")
print(f"KF-STD Test: {MSE_KF_dB_std:.4f} [dB]")

##########################
### Evaluate KalmanNet ###
##########################
# Build the KalmanNet neural network (with original model, it will learn to adapt)
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model_KN, args)  # Use sys_model_KN instead of sys_model
print("Number of trainable parameters for KNet pass 1:",
      sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))

# Set up the training and evaluation pipeline
KNet_Pipeline = Pipeline(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model_KN)  # Set system model - use sys_model_KN
KNet_Pipeline.setModel(KNet_model)   # Set KalmanNet model
KNet_Pipeline.setTrainingParams(args)  # Set training parameters

# Add this code to implement weighted loss for state components
if Train_Loss_On_AllState:
    # Move weights tensor to the correct device
    state_weights = state_weights.to(device)
    # Store the original loss function
    original_loss_fn = KNet_Pipeline.loss_fn

    # def weighted_loss_fn(output, target):
    #     # If not using all states, fall back to original behavior
    #     if not Train_Loss_On_AllState:
    #         return original_loss_fn(output, target)
    #     # Calculate MSE separately for each state component
    #     # This approach ensures each component is weighted properly regardless of scale
    #     position_mse = torch.mean((output[:, 0, :] - target[:, 0, :]) ** 2)
    #     velocity_mse = torch.mean((output[:, 1, :] - target[:, 1, :]) ** 2)
    #     # Handle acceleration if present (for CA model)
    #     if output.shape[1] > 2:
    #         accel_mse = torch.mean((output[:, 2, :] - target[:, 2, :]) ** 2)
    #         # Apply weights to individual components
    #         weighted_mse = (position_weight * position_mse +
    #                         velocity_weight * velocity_mse +
    #                         acceleration_weight * accel_mse)
    #         # Normalize by sum of weights to keep loss scale consistent
    #         return weighted_mse / (position_weight + velocity_weight + acceleration_weight)
    #     else:
    #         # For CV model with only position and velocity
    #         weighted_mse = position_weight * position_mse + velocity_weight * velocity_mse
    #         return weighted_mse / (position_weight + velocity_weight)

    def weighted_loss_fn(output, target):
        if not Train_Loss_On_AllState:
            return original_loss_fn(output, target)

        # Calculate component-wise MSE
        component_mse = torch.stack([
            torch.mean((output[:, i, :] - target[:, i, :]) ** 2)
            for i in range(min(output.shape[1], len(state_weights)))
        ])

        # Apply weights (only for available components)
        weights = state_weights[:component_mse.shape[0]].to(device)
        return (weights * component_mse).sum() / weights.sum()

    # Replace the loss function in the pipeline
    KNet_Pipeline.loss_fn = weighted_loss_fn
    print(f"Using weighted loss with weights: Position={position_weight}, "
          f"Velocity={velocity_weight}, Acceleration={acceleration_weight}")

# Train KalmanNet
randomInit_train = args.randomInit_train and KnownRandInit_train
print(f"Train KNet with {'Known' if randomInit_train else 'Unknown'} Random Initial State")
print(f"Train Loss on All States (if false, loss on position only): {Train_Loss_On_AllState}")

train_args = {
    'MaskOnState': not Train_Loss_On_AllState,
    'randomInit': randomInit_train
}
if randomInit_train:
    train_args.update({'cv_init': cv_init, 'train_init': train_init})

[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = \
    KNet_Pipeline.NNTrain(sys_model_KN, cv_input, cv_target, train_input, train_target,
                          path_results, **train_args)

# Test KalmanNet
randomInit_test = args.randomInit_test and KnownRandInit_test
print(f"Test KNet with {'Known' if randomInit_test else 'Unknown'} Random Initial State")
print(f"Compute Loss on All States (if false, loss on position only): {Loss_On_AllState}")

test_args = {
    'MaskOnState': not Loss_On_AllState,
    'randomInit': randomInit_test
}
if randomInit_test:
    test_args.update({'test_init': test_init})

[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_out, RunTime] = \
    KNet_Pipeline.NNTest(sys_model_KN, test_input, test_target, path_results, **test_args)

####################
### Plot results ###
####################
# Set up file paths for saving plots
PlotfolderName = "Figures/Linear_CA/"
num_plots = 10  # Number of test trajectories to plot

# Initialize plotting object
Plot = Plot(PlotfolderName, "")
print("Plot")

# The plotting loop could be simplified with a list of dimensions and names:
dimensions = [(0, "position"), (1, "velocity"), (2, "acceleration")]

for i in range(num_plots):
    if i >= test_target.shape[0]:
        print(
            f"Warning: Not enough test trajectories. Requested trajectory {i + 1} but only have {test_target.shape[0]}")
        break

    prefix = f"{i + 1}_"
    single_test_target = test_target[i:i + 1]
    single_KF_out = KF_out[i:i + 1]
    single_KNet_out = KNet_out[i:i + 1]

    for dim, name in dimensions:
        filename = f"{prefix}TrainPVA_{name}.png"
        Plot.plotTraj_CA(single_test_target, single_KF_out, single_KNet_out,
                         dim=dim, file_name=PlotfolderName + filename)

    # Also save the overall plots (all trajectories) with a special prefix
overall_prefix = "all_"
PlotfileName0 = f"{overall_prefix}TrainPVA_position.png"
PlotfileName1 = f"{overall_prefix}TrainPVA_velocity.png"
PlotfileName2 = f"{overall_prefix}TrainPVA_acceleration.png"

# Generate plots comparing ground truth, KF estimates, and KalmanNet estimates for all trajectories
# dim=0: Position
Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=0, file_name=PlotfolderName+PlotfileName0)
# dim=1: Velocity
Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=1, file_name=PlotfolderName+PlotfileName1)
# dim=2: Acceleration
Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=2, file_name=PlotfolderName+PlotfileName2)