"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

class KalmanNetNN(torch.nn.Module):
    """
    KalmanNetNN: A neural network implementation of Kalman filtering

    This class implements a learnable Kalman filter using neural networks.
    It replaces the traditional Kalman gain calculation with a neural network
    that learns to predict optimal Kalman gains from data.
    """

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        """Initialize the KalmanNetNN class"""
        super().__init__()

    def NNBuild(self, SysModel, args):
        """
        Build the KalmanNet neural network architecture

        Args:
            SysModel: System model containing dynamics and observation functions
            args: Configuration parameters
        """
        # Set device (GPU or CPU)
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Initialize system dynamics (state transition and observation functions)
        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        # Note: These commented lines show an alternative way to calculate hidden layer sizes
        # Number of neurons in the 1st hidden layer
        #H1_KNet = (SysModel.m + SysModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        #H2_KNet = (SysModel.m * SysModel.n) * 1 * (4)

        # Initialize the Kalman Gain Network with prior knowledge
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        """
        Initialize the neural network architecture for Kalman gain estimation

        Args:
            prior_Q: Prior process noise covariance matrix
            prior_Sigma: Prior state covariance matrix
            prior_S: Prior innovation covariance matrix
            args: Configuration parameters
        """
        # Sequence length for RNN inputs (1 for step-by-step processing)
        self.seq_len_input = 1  # KNet calculates time-step by time-step
        self.batch_size = args.n_batch  # Batch size for training

        # Store prior matrices on the selected device
        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        # GRU to track process noise covariance Q
        self.d_input_Q = self.m * args.in_mult_KNet  # Input dimension
        self.d_hidden_Q = self.m ** 2  # Hidden state dimension (m×m for covariance)
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track state covariance Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet  # Input dimension
        self.d_hidden_Sigma = self.m ** 2  # Hidden state dimension (m×m for covariance)
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        # GRU to track innovation covariance S
        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet  # Input dimension
        self.d_hidden_S = self.n ** 2  # Hidden state dimension (n×n for covariance)
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        # Fully connected layer 1: Maps Sigma GRU output to innovation covariance space
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU()).to(self.device)

        # Fully connected layer 2: Main layer for Kalman gain computation
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m  # Output dimension is n×m (Kalman gain matrix size)
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2)).to(self.device)
        # Enhanced FC2 with more layers and capacity
        # self.FC2 = nn.Sequential(
        #     nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.d_hidden_FC2, self.d_hidden_FC2),  # Add another layer
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.d_hidden_FC2, self.d_hidden_FC2 // 2),  # Add bottleneck layer
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.d_hidden_FC2 // 2, self.d_output_FC2)
        # ).to(self.device)

        # Fully connected layer 3: Part of the backward flow for refining estimates
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU()).to(self.device)

        # Fully connected layer 4: Updates Sigma GRU hidden state
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU()).to(self.device)

        # Fully connected layer 5: Processes state update differences
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        # Enhanced FC5 - processes state update differences
        # self.FC5 = nn.Sequential(
        #     nn.Linear(self.d_input_FC5, self.d_output_FC5 * 2),  # Double width
        #     nn.LeakyReLU(0.1),  # Better gradient flow
        #     nn.Linear(self.d_output_FC5 * 2, self.d_output_FC5 * 2),  # Add depth
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.d_output_FC5 * 2, self.d_output_FC5)
        # ).to(self.device)
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU()).to(self.device)

        # Fully connected layer 6: Processes state evolution differences
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        # Enhanced FC6 - processes state evolution differences
        # self.FC6 = nn.Sequential(
        #     nn.Linear(self.d_input_FC6, self.d_output_FC6 * 2),  # Double width
        #     nn.LeakyReLU(0.1),  # Better gradient flow
        #     nn.Linear(self.d_output_FC6 * 2, self.d_output_FC6 * 2),  # Add depth
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.d_output_FC6 * 2, self.d_output_FC6)
        # ).to(self.device)
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU()).to(self.device)

        # Fully connected layer 7: Processes observation differences
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU()).to(self.device)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        """
        Initialize the system dynamics functions and dimensions

        Args:
            f: State transition function
            h: Observation function
            m: State dimension
            n: Observation dimension
        """
        # Set State Evolution Function
        self.f = f  # State transition function
        self.m = m  # State dimension

        # Set Observation Function
        self.h = h  # Observation function
        self.n = n  # Observation dimension

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        """
        Initialize the sequence for filtering

        Args:
            M1_0 (torch.tensor): Initial state estimate (1st moment of x at time 0) [batch_size, m, 1]
            T: Sequence length
        """
        self.T = T  # Sequence length

        # Initialize state estimates
        self.m1x_posterior = M1_0.to(self.device)  # Current posterior estimate
        self.m1x_posterior_previous = self.m1x_posterior  # Previous posterior estimate
        self.m1x_prior_previous = self.m1x_posterior  # Previous prior estimate
        self.y_previous = self.h(self.m1x_posterior)  # Previous observation

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        """
        Compute prior estimates for the current time step
        """
        # Predict the 1-st moment of x (prior state estimate)
        self.m1x_prior = self.f(self.m1x_posterior)

        # Predict the 1-st moment of y (predicted observation)
        self.m1y = self.h(self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        """
        Estimate the Kalman gain using the neural network

        Args:
            y: Current observation
        """
        # Calculate observation differences
        # both in size [batch_size, n]
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)  # Observation difference
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)  # Innovation (observation - prediction)

        # Calculate state differences
        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)  # State evolution difference
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)  # State update difference


        # Normalize differences for stability
        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        # Reshape Kalman Gain to a Matrix [batch_size, m, n]
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    # Notes by Joey (5/Apr/2025):
    # Important note on variable timing in KalmanNet:
    #
    # When step_KGain_est() is called, variable timing is as follows:
    # - self.m1x_posterior: contains posterior estimate from time t-1
    # - self.m1x_posterior_previous: contains posterior estimate from time t-2
    # - self.m1x_prior: contains prior estimate for current time t (just calculated in step_prior())
    # - self.m1x_prior_previous: contains prior estimate from time t-1
    #
    # The forward update difference:
    # fw_update_diff = self.m1x_posterior - self.m1x_prior_previous
    # calculates: posterior(t-1) - prior(t-1)
    #
    # This matches the paper's definition of forward update difference (F4):
    # "The difference between posterior state estimate and prior state estimate"
    # where "for time instance t we use Δx̂t-1"
    #
    # The variable naming can be confusing because:
    # 1. Names don't indicate time steps (t, t-1, etc.)
    # 2. self.m1x_prior_previous temporarily equals self.m1x_prior at the end of each step
    # 3. "previous" in variable names doesn't consistently refer to the same time offset
    def KNet_step(self, y):
        """
        Perform one step of the KalmanNet filtering algorithm

        Args:
            y: Current observation [batch_size, n, 1]

        Returns:
            Updated state estimate (posterior) [batch_size, m, 1]
        """
        # Compute Priors (predict step)
        self.step_prior()

        # Compute Kalman Gain using neural network
        self.step_KGain_est(y)

        # Innovation: difference between actual and predicted observation
        dy = y - self.m1y  # [batch_size, n, 1]

        # Compute the posterior state estimate (update step)
        INOV = torch.bmm(self.KGain, dy)  # Apply Kalman gain to innovation
        self.m1x_posterior_previous = self.m1x_posterior  # Store previous posterior
        self.m1x_posterior = self.m1x_prior + INOV  # Update state estimate

        # Store prior for next iteration
        self.m1x_prior_previous = self.m1x_prior

        # Update previous observation
        self.y_previous = y

        # Return the updated state estimate
        return self.m1x_posterior

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        """
        Calculate Kalman gain using the neural network architecture

        Args:
            obs_diff: Observation difference
            obs_innov_diff: Innovation difference
            fw_evol_diff: State evolution difference
            fw_update_diff: State update difference

        Returns:
            Estimated Kalman gain in flattened form
        """
        # Helper function to expand dimensions for RNN input
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        # Expand dimensions for RNN inputs
        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################

        # FC 5: Process state update differences
        in_FC5 = fw_update_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU: Track process noise covariance
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        # FC 6: Process state evolution differences
        in_FC6 = fw_evol_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU: Track state covariance
        in_Sigma = torch.cat((out_Q, out_FC6), 2)  # Concatenate Q and state evolution info
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1: Map from state covariance to observation covariance space
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7: Process observation differences
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)  # Concatenate observation differences
        out_FC7 = self.FC7(in_FC7)

        # S-GRU: Track innovation covariance
        in_S = torch.cat((out_FC1, out_FC7), 2)  # Combine state and observation information
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC 2: Main Kalman gain computation
        in_FC2 = torch.cat((out_Sigma, out_S), 2)  # Combine state and innovation covariances
        out_FC2 = self.FC2(in_FC2)  # This output represents the Kalman gain

        #####################
        ### Backward Flow ###
        #####################
        # The backward flow refines the estimates and updates hidden states

        # FC 3: Process S and Kalman gain information
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4: Update Sigma hidden state
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # Update hidden state of the Sigma-GRU for next iteration
        self.h_Sigma = out_FC4

        # Return the computed Kalman gain
        return out_FC2

    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        """
        PyTorch forward method for the KalmanNet model

        Args:
            y: Current observation

        Returns:
            Updated state estimate
        """
        y = y.to(self.device)  # Move observation to the correct device
        return self.KNet_step(y)  # Perform one step of KalmanNet

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_KNet(self):
        """
        Initialize the hidden states of all GRUs with prior knowledge
        """
        # Get a reference weight to determine data type and device
        weight = next(self.parameters()).data

        # Initialize S (innovation covariance) hidden state
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        # Initialize with prior knowledge and expand for batch size
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)

        # Initialize Sigma (state covariance) hidden state
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        # Initialize with prior knowledge and expand for batch size
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)

        # Initialize Q (process noise covariance) hidden state
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        # Initialize with prior knowledge and expand for batch size
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)