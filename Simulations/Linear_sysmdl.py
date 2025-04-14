"""# **Class: System Model for Linear Cases**

1 Store system model parameters: 
    state transition matrix F, 
    observation matrix H, 
    process noise covariance matrix Q, 
    observation noise covariance matrix R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test, etc.

2 Generate dataset for linear cases

This class implements a linear state-space model of the form:
    x_t = F x_{t-1} + q_t    (State evolution equation)
    y_t = H x_t + r_t        (Observation equation)

where:
    x_t is the state vector at time t
    y_t is the observation/measurement vector at time t
    F is the state transition matrix
    H is the observation matrix
    q_t is the process noise (Gaussian with covariance Q)
    r_t is the observation noise (Gaussian with covariance R)
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class SystemModel:

    def __init__(self, F, Q, H, R, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):
        """
        Initialize the linear system model with its parameters

        Args:
            F: State transition matrix (m×m)
            Q: Process noise covariance matrix (m×m)
            H: Observation matrix (n×m)
            R: Observation noise covariance matrix (n×n)
            T: Sequence length for training/CV
            T_test: Sequence length for testing
            prior_Q: Prior process noise covariance for KalmanNet (default: identity)
            prior_Sigma: Prior state covariance for KalmanNet (default: zeros)
            prior_S: Prior innovation covariance for KalmanNet (default: identity)
        """
        ####################
        ### Motion Model ###
        ####################
        self.F = F  # State transition matrix
        self.m = self.F.size()[0]  # State dimension
        self.Q = Q  # Process noise covariance matrix

        #########################
        ### Observation Model ###
        #########################
        self.H = H  # Observation matrix
        self.n = self.H.size()[0]  # Observation dimension
        self.R = R  # Observation noise covariance matrix

        ################
        ### Sequence ###
        ################
        # Assign sequence lengths
        self.T = T  # Training/CV sequence length
        self.T_test = T_test  # Testing sequence length

        #########################
        ### Covariance Priors ###
        #########################
        # These priors are used to initialize KalmanNet's GRU hidden states
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)  # Default: identity matrix
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))  # Default: zero matrix
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)  # Default: identity matrix
        else:
            self.prior_S = prior_S


    def f(self, x):
        """
        State transition function: x_t = F * x_{t-1}
        Handles batched inputs for efficient processing

        Args:
            x: State tensor of shape [batch_size, m, 1]

        Returns:
            Next state tensor of shape [batch_size, m, 1]
        """
        # Create a batched version of F matrix to match input batch size
        batched_F = self.F.to(x.device).view(1,self.F.shape[0],self.F.shape[1]).expand(x.shape[0],-1,-1)
        # Perform batch matrix multiplication: F * x
        return torch.bmm(batched_F, x)

    def h(self, x):
        """
        Observation function: y_t = H * x_t
        Handles batched inputs for efficient processing

        Args:
            x: State tensor of shape [batch_size, m, 1]

        Returns:
            Observation tensor of shape [batch_size, n, 1]
        """
        # Create a batched version of H matrix to match input batch size
        batched_H = self.H.to(x.device).view(1,self.H.shape[0],self.H.shape[1]).expand(x.shape[0],-1,-1)
        # Perform batch matrix multiplication: H * x
        return torch.bmm(batched_H, x)

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):
        """
        Initialize a single sequence with initial state and covariance

        Args:
            m1x_0: Initial state vector (m×1)
            m2x_0: Initial state covariance matrix (m×m)
        """
        self.m1x_0 = m1x_0  # Initial state mean
        self.x_prev = m1x_0  # Set as previous state for sequence generation
        self.m2x_0 = m2x_0  # Initial state covariance

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        """
        Initialize a batch of sequences with initial states and covariances

        Args:
            m1x_0_batch: Batch of initial state vectors [batch_size, m, 1]
            m2x_0_batch: Initial state covariance matrix (m×m) - same for all in batch
        """
        self.m1x_0_batch = m1x_0_batch  # Batch of initial states
        self.x_prev = m1x_0_batch  # Set as previous states for sequence generation
        self.m2x_0_batch = m2x_0_batch  # Initial state covariance

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):
        """
        Update the process and observation noise covariance matrices

        Args:
            Q: New process noise covariance matrix
            R: New observation noise covariance matrix
        """
        self.Q = Q  # Update process noise covariance
        self.R = R  # Update observation noise covariance

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        """
        Generate a single trajectory of states and observations

        Args:
            Q_gen: Process noise covariance for generation
            R_gen: Observation noise covariance for generation
            T: Length of sequence to generate
        """
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            if torch.equal(Q_gen,torch.zeros(self.m,self.m)):
                # No noise case: x_t = F * x_{t-1}
                xt = self.F.matmul(self.x_prev)
            elif self.m == 1:
                # 1-dimensional state case
                xt = self.F.matmul(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen)  # Sample scalar noise
                # Additive Process Noise: x_t = F * x_{t-1} + q_t
                xt = torch.add(xt,eq)
            else:
                # Multi-dimensional state case
                xt = self.F.matmul(self.x_prev)
                mean = torch.zeros([self.m])
                # Sample from multivariate normal with covariance Q_gen
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()  # Sample process noise
                eq = torch.reshape(eq[:], xt.size())  # Reshape to match state dimensions
                # Additive Process Noise: x_t = F * x_{t-1} + q_t
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            if torch.equal(R_gen,torch.zeros(self.n,self.n)):
                # No noise case: y_t = H * x_t
                yt = self.H.matmul(xt)
            elif self.n == 1:
                # 1-dimensional observation case
                yt = self.H.matmul(xt)
                er = torch.normal(mean=0, std=R_gen)  # Sample scalar noise
                # Additive Observation Noise: y_t = H * x_t + r_t
                yt = torch.add(yt,er)
            else:
                # Multi-dimensional observation case
                yt = self.H.matmul(xt)
                mean = torch.zeros([self.n])
                # Sample from multivariate normal with covariance R_gen
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()  # Sample observation noise
                er = torch.reshape(er[:], yt.size())  # Reshape to match observation dimensions
                # Additive Observation Noise: y_t = H * x_t + r_t
                yt = torch.add(yt,er)

            ########################
            ### Squeeze to Array ###
            ########################
            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt,1)
            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt,1)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt  # Update previous state for next iteration

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, args, size, T, randomInit=False):
        """
        Generate a batch of trajectories efficiently

        Args:
            args: Configuration parameters
            size: Batch size (number of trajectories)
            T: Length of each trajectory
            randomInit: If True, use random initial states; otherwise use fixed initial state
        """
        if(randomInit):
            # Handle random initial conditions
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.zeros(size, self.m, 1)

            if args.distribution == 'uniform':
                ### Generate uniform random initial states
                for i in range(size):
                    # Generate random values in [0,1] and scale by variance
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                    self.m1x_0_rand[i,:,0:1] = initConditions.view(self.m,1)

            elif args.distribution == 'normal':
                ### Generate normally distributed random initial states
                for i in range(size):
                    # Create multivariate normal distribution with mean=m1x_0 and covariance=m2x_0
                    distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                    initConditions = distrib.rsample().view(self.m,1)  # Sample and reshape
                    self.m1x_0_rand[i,:,0:1] = initConditions
            else:
                raise ValueError('args.distribution not supported!')

            # Initialize batch sequence with random initial states
            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)
        else:
            # Use fixed initial state for all trajectories in batch
            initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)  # Repeat same initial state
            self.Init_batched_sequence(initConditions, self.m2x_0)

        if(args.randomLength):
            # Handle variable-length sequences with padding
            # Allocate arrays with maximum possible length
            self.Input = torch.zeros(size, self.n, args.T_max)  # Observations
            self.Target = torch.zeros(size, self.m, args.T_max)  # States
            self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool)  # Mask for valid positions

            # Generate random sequence lengths between T_min and T_max
            T_tensor = torch.round((args.T_max-args.T_min)*torch.rand(size)).int()+args.T_min

            # Generate each sequence individually
            for i in range(0, size):
                # Generate a sequence of specified length
                self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
                # Store observations (inputs)
                self.Input[i, :, 0:T_tensor[i].item()] = self.y
                # Store states (targets)
                self.Target[i, :, 0:T_tensor[i].item()] = self.x
                # Create mask to identify valid (non-padded) positions
                self.lengthMask[i, 0:T_tensor[i].item()] = True

        else:
            # Fixed-length sequences - more efficient batch generation
            # Allocate arrays for fixed length T
            self.Input = torch.empty(size, self.n, T)   # Observations [batch_size, n, T]
            self.Target = torch.empty(size, self.m, T)  # States [batch_size, m, T]

            # Set initial states
            self.x_prev = self.m1x_0_batch  # [batch_size, m, 1]
            xt = self.x_prev

            # Generate all sequences in parallel (batched computation)
            for t in range(0, T):
                ########################
                #### State Evolution ###
                ########################
                if torch.equal(self.Q,torch.zeros(self.m,self.m)):
                    # No noise case
                    xt = self.f(self.x_prev)  # Batched state transition
                elif self.m == 1:
                    # 1-dimensional state case
                    xt = self.f(self.x_prev)
                    # Generate batch of scalar noise values
                    eq = torch.normal(mean=torch.zeros(size), std=self.Q).view(size,1,1)
                    # Add noise to all states in batch
                    xt = torch.add(xt,eq)
                else:
                    # Multi-dimensional state case
                    xt = self.f(self.x_prev)
                    mean = torch.zeros([size, self.m])
                    # Sample batch of multivariate normal noise vectors
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
                    eq = distrib.rsample().view(size,self.m,1)  # Reshape to [batch_size, m, 1]
                    # Add noise to all states in batch
                    xt = torch.add(xt,eq)

                ################
                ### Emission ###
                ################
                if torch.equal(self.R,torch.zeros(self.n,self.n)):
                    # No noise case
                    yt = self.h(xt)  # Batched observation function
                elif self.n == 1:
                    # 1-dimensional observation case
                    yt = self.h(xt)
                    # Generate batch of scalar noise values
                    er = torch.normal(mean=torch.zeros(size), std=self.R).view(size,1,1)
                    # Add noise to all observations in batch
                    yt = torch.add(yt,er)
                else:
                    # Multi-dimensional observation case
                    yt = self.H.matmul(xt)  # Note: This should use self.h() for consistency
                    mean = torch.zeros([size,self.n])
                    # Sample batch of multivariate normal noise vectors
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
                    er = distrib.rsample().view(size,self.n,1)  # Reshape to [batch_size, n, 1]
                    # Add noise to all observations in batch
                    yt = torch.add(yt,er)

                ########################
                ### Store in Arrays ###
                ########################
                # Save states and observations for all sequences at time t
                self.Target[:, :, t] = torch.squeeze(xt,2)  # [batch_size, m]
                self.Input[:, :, t] = torch.squeeze(yt,2)   # [batch_size, n]

                ################################
                ### Update Previous States ###
                ################################
                self.x_prev = xt  # Store current states for next iteration


    def sampling(self, q, r, gain):
        """
        Generate perturbed covariance matrices for robustness testing

        Args:
            q: Base process noise standard deviation
            r: Base observation noise standard deviation
            gain: Perturbation factor (0 means no perturbation)

        Returns:
            [Q_gen, R_gen]: Perturbed process and observation noise covariance matrices
        """
        # Generate perturbed process noise covariance
        if (gain != 0):
            gain_q = 0.1  # Scale factor for process noise perturbation
            # Add scaled perturbation to diagonal elements
            aq = gain_q * q * torch.eye(self.m)
            # Alternative commented-out approaches:
            #aq = gain * q * np.random.randn(self.m, self.m)  # Random perturbation
            #aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # Structured perturbation
        else:
            aq = 0  # No perturbation

        # Create process noise covariance as A'A to ensure positive definiteness
        Aq = q * torch.eye(self.m) + aq
        Q_gen = torch.transpose(Aq, 0, 1) * Aq  # Equivalent to Aq.T @ Aq

        # Generate perturbed observation noise covariance
        if (gain != 0):
            gain_r = 0.5  # Scale factor for observation noise perturbation
            # Add scaled perturbation to diagonal elements
            ar = gain_r * r * torch.eye(self.n)
            # Alternative commented-out approaches:
            #ar = gain * r * np.random.randn(self.n, self.n)  # Random perturbation
            #ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # Structured perturbation
        else:
            ar = 0  # No perturbation

        # Create observation noise covariance as A'A to ensure positive definiteness
        Ar = r * torch.eye(self.n) + ar
        R_gen = torch.transpose(Ar, 0, 1) * Ar  # Equivalent to Ar.T @ Ar

        return [Q_gen, R_gen]