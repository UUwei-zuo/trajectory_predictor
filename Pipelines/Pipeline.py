"""
This file contains the class Pipeline_EKF, 
which is used to train and test KalmanNet.
"""

import torch
import torch.nn as nn
import random
import time
from trajectory_plotter import TrajectoryPlotter

class Pipeline_KN:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
            # Enable cuDNN benchmark for faster operations
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        # Rest of the method remains unchanged
        self.N_steps = args.n_steps
        self.N_B = args.n_batch  # Number of Samples in Batch
        self.learningRate = args.lr  # Learning Rate
        self.weightDecay = args.wd  # L2 Weight Regularization - Weight Decay
        self.alpha = args.alpha  # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        # Initialize learning rate scheduler if enabled
        if hasattr(args, 'use_lr_scheduler') and args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=args.lr_scheduler_gamma,
                patience=args.lr_scheduler_patience,
                threshold=args.lr_scheduler_threshold,
            )

    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, \
                MaskOnState=False, randomInit=False, cv_init=None, train_init=None, \
                train_lengthMask=None, cv_lengthMask=None):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps], device=self.device)
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps], device=self.device)
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps], device=self.device)
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps], device=self.device)

        if MaskOnState:
            # Create appropriate mask based on state dimension
            if SysModel.m == 6:  # 2D CA model
                state_mask = torch.tensor([True, False, False, True, False, False],
                                          device=self.device)  # x and y positions
            elif SysModel.m == 4:  # 2D CV model
                state_mask = torch.tensor([True, False, True, False], device=self.device)  # x and y positions
            elif SysModel.m == 2:  # 1D model
                state_mask = torch.tensor([True, False], device=self.device)  # position only
            else:
                # Default case
                state_mask = torch.tensor([True] + [False] * (SysModel.m - 1), device=self.device)

            # Also update the observation mask to handle 2D observations
            obs_mask = torch.zeros(SysModel.n, dtype=torch.bool, device=self.device)
            if SysModel.n >= 2 and (SysModel.m == 6 or SysModel.m == 4):  # For 2D models
                obs_mask[0] = True  # x position
                obs_mask[1] = True  # y position (if available)
            else:
                obs_mask[0] = True  # Only first element for 1D models

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        # Early stopping variables
        early_stopping_enabled = hasattr(self.args, 'use_early_stopping') and self.args.use_early_stopping
        if early_stopping_enabled:
            early_stopping_counter = 0
            early_stopping_patience = self.args.early_stopping_patience
            early_stopping_min_delta = self.args.early_stopping_min_delta
            print(
                f"Early stopping enabled with patience {early_stopping_patience} and min delta {early_stopping_min_delta}")

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B
            # Init Hidden State
            self.model.init_hidden_KNet()

            # Init Training Batch tensors
            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T], device=self.device)
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T], device=self.device)
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T], device=self.device)
            if self.args.randomLength:
                MSE_train_linear_LOSS = torch.zeros([self.N_B], device=self.device)
                MSE_cv_linear_LOSS = torch.zeros([self.N_CV], device=self.device)

            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E  # N_B must be smaller than N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                if self.args.randomLength:
                    y_training_batch[ii, :, train_lengthMask[index, :]] = train_input[index, :,
                                                                          train_lengthMask[index, :]]
                    train_target_batch[ii, :, train_lengthMask[index, :]] = train_target[index, :,
                                                                            train_lengthMask[index, :]]
                else:
                    y_training_batch[ii, :, :] = train_input[index]
                    train_target_batch[ii, :, :] = train_target[index]
                ii += 1

            # Init Sequence
            train_init_batch = torch.zeros([self.N_B, SysModel.m, 1], device=self.device)
            ii = 0
            for index in n_e:
                train_init_batch[ii, :, 0] = train_target[index, :, 0]
                ii += 1
            self.model.InitSequence(train_init_batch, SysModel.T)

            # Forward Computation
            # MODIFICATION: Directly assign initial state to output for t=0
            x_out_training_batch[:, :, 0] = torch.squeeze(train_init_batch, 2)

            # Start processing from t=1
            for t in range(1, SysModel.T):
                x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t], 2)))

            # Compute Training Loss
            MSE_trainbatch_linear_LOSS = 0
            if (self.args.CompositionLoss):
                y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T], device=self.device)
                for t in range(SysModel.T):
                    y_hat[:, :, t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_training_batch[:, :, t])))

                if (MaskOnState):
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:  # mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
                                x_out_training_batch[jj, state_mask, train_lengthMask[index]],
                                train_target_batch[jj, state_mask, train_lengthMask[index]]) + (
                                                                1 - self.alpha) * self.loss_fn(
                                y_hat[jj, obs_mask, train_lengthMask[index]],
                                y_training_batch[jj, obs_mask, train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch[:, state_mask, :],
                                                                               train_target_batch[:, state_mask, :]) + (
                                                             1 - self.alpha) * self.loss_fn(y_hat[:, obs_mask, :],
                                                                                            y_training_batch[:,
                                                                                            obs_mask, :])
                else:  # no mask on state
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:  # mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
                                x_out_training_batch[jj, :, train_lengthMask[index]],
                                train_target_batch[jj, :, train_lengthMask[index]]) + (1 - self.alpha) * self.loss_fn(
                                y_hat[jj, :, train_lengthMask[index]], y_training_batch[jj, :, train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch,
                                                                               train_target_batch) + (
                                                                 1 - self.alpha) * self.loss_fn(y_hat, y_training_batch)

            else:  # no composition loss
                if (MaskOnState):
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:  # mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.loss_fn(
                                x_out_training_batch[jj, state_mask, train_lengthMask[index]],
                                train_target_batch[jj, state_mask, train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch[:, state_mask, :],
                                                                  train_target_batch[:, state_mask, :])
                else:  # no mask on state
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:  # mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.loss_fn(
                                x_out_training_batch[jj, :, train_lengthMask[index]],
                                train_target_batch[jj, :, train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            MSE_trainbatch_linear_LOSS.backward(retain_graph=True)

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # Step the learning rate scheduler if enabled

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            self.model.batch_size = self.N_CV
            # Init Hidden State
            self.model.init_hidden_KNet()
            with torch.no_grad():

                SysModel.T_test = cv_input.size()[-1]  # T_test is the maximum length of the CV sequences

                x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test], device=self.device)

                # Get true initial states from cv_target
                cv_init_states = cv_target[:, :, 0].unsqueeze(-1)
                self.model.InitSequence(cv_init_states, SysModel.T_test)

                # MODIFICATION: Directly assign initial state to output for t=0
                x_out_cv_batch[:, :, 0] = torch.squeeze(cv_init_states, 2)

                # Start processing from t=1
                for t in range(1, SysModel.T_test):
                    x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:, :, t], 2)))

                # Compute CV Loss
                MSE_cvbatch_linear_LOSS = 0
                if (MaskOnState):
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(
                                x_out_cv_batch[index, state_mask, cv_lengthMask[index]],
                                cv_target[index, state_mask, cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch[:, state_mask, :],
                                                               cv_target[:, state_mask, :])
                else:
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index, :, cv_lengthMask[index]],
                                                                     cv_target[index, :, cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                # Step the learning rate scheduler if enabled
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(self.MSE_cv_dB_epoch[ti])

                # Modified code with epsilon check:
                epsilon = 1e-5  # Small value to account for floating point precision
                if (self.MSE_cv_dB_epoch[
                    ti] < self.MSE_cv_dB_opt - epsilon):  # Only count as improvement if better by epsilon
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    torch.save(self.model, path_results + 'best-model.pt')

                    # Reset early stopping counter since we found a better model
                    if early_stopping_enabled:
                        early_stopping_counter = 0
                        print(f"Found new best model at epoch {ti} with validation loss: {self.MSE_cv_dB_opt:.4f} [dB]")
                elif early_stopping_enabled:
                    # If no improvement, increment the counter
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(
                            f"Early stopping triggered after {ti + 1} epochs (no improvement for {early_stopping_patience} epochs)")
                        # Return early with the results so far
                        return [self.MSE_cv_linear_epoch[:ti + 1], self.MSE_cv_dB_epoch[:ti + 1],
                                self.MSE_train_linear_epoch[:ti + 1], self.MSE_train_dB_epoch[:ti + 1]]
                    elif ti % 10 == 0:  # Print progress every 10 epochs
                        print(
                            f"No improvement for {early_stopping_counter} epochs. Best is still {self.MSE_cv_dB_opt:.4f} [dB] at epoch {self.MSE_cv_idx_opt}")

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False, \
               randomInit=False, test_init=None, load_model=False, load_model_path=None, \
               test_lengthMask=None):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device)
        else:
            self.model = torch.load(path_results + 'best-model.pt', map_location=self.device)

        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)

        # Get true initial states from test_target
        true_init_states = test_target[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, state_dim, 1]

        # Print debug info
        print("KalmanNet using true initial states:")
        print("First sample initial state:", true_init_states[0].squeeze())
        print("First ground truth state:", test_target[0, :, 0])

        if MaskOnState:
            # Create appropriate mask based on state dimension
            if SysModel.m == 6:  # 2D CA model
                state_mask = torch.tensor([True, False, False, True, False, False],
                                          device=self.device)  # x and y positions
            elif SysModel.m == 4:  # 2D CV model
                state_mask = torch.tensor([True, False, True, False], device=self.device)  # x and y positions
            elif SysModel.m == 2:  # 1D model
                state_mask = torch.tensor([True, False], device=self.device)  # position only
            else:
                # Default case
                state_mask = torch.tensor([True] + [False] * (SysModel.m - 1), device=self.device)

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden_KNet()

        with torch.no_grad():
            # Always use true initial states
            self.model.InitSequence(true_init_states, SysModel.T_test)

            # MODIFICATION: Directly assign initial state to output for t=0
            x_out_test[:, :, 0] = torch.squeeze(true_init_states, 2)

            # Start processing from t=1
            for t in range(1, SysModel.T_test):
                x_out_test[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:, :, t], 2)))

        start = time.time()

        # Always use true initial states
        self.model.InitSequence(true_init_states, SysModel.T_test)

        # MODIFICATION: Directly assign initial state to output for t=0
        x_out_test[:, :, 0] = torch.squeeze(true_init_states, 2)

        # Start processing from t=1
        for t in range(1, SysModel.T_test):
            x_out_test[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:, :, t], 2)))

        end = time.time()

        t = end - start

        # MSE loss
        for j in range(self.N_T):  # cannot use batch due to different length and std computation
            if (MaskOnState):
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, state_mask, test_lengthMask[j]],
                                                          test_target[j, state_mask, test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, state_mask, :],
                                                          test_target[j, state_mask, :]).item()
            else:
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :, test_lengthMask[j]],
                                                          test_target[j, :, test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :, :], test_target[j, :, :]).item()

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):
        # Replace Plot_extended with TrajectoryPlotter
        self.Plot = TrajectoryPlotter(self.folderName, self.modelName)

        # The method names remain the same due to compatibility methods in TrajectoryPlotter
        self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)