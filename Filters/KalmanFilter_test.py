import torch
import torch.nn as nn
import time
from Filters.Linear_KF import KalmanFilter


def KFTest(args, SysModel, test_input, test_target, allStates=True, \
           randomInit=False, test_init=None, test_lengthMask=None):
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.zeros(args.N_T)

    # Get initial states from test_target
    true_init_states = test_target[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, state_dim, 1]

    start = time.time()

    # Create KF instance
    KF = KalmanFilter(SysModel, args)

    # Initialize with true initial states from the data
    zero_cov = torch.zeros(args.N_T, SysModel.m, SysModel.m).to(test_target.device)

    # Print debug info
    print("Initial state shape:", true_init_states.shape)
    print("First sample initial state:", true_init_states[0].squeeze())
    print("First ground truth state:", test_target[0, :, 0])

    # Always use the true initial states from test_target
    KF.Init_batched_sequence(true_init_states, zero_cov)

    # Generate batch and get output
    KF.GenerateBatch(test_input)
    KF_out = KF.x

    end = time.time()
    t = end - start

    if not allStates:
        # Create appropriate mask based on state dimension
        if SysModel.m == 6:  # 2D CA model
            loc = torch.tensor([True, False, False, True, False, False])  # x and y positions
        elif SysModel.m == 4:  # 2D CV model
            loc = torch.tensor([True, False, True, False])  # x and y positions
        elif SysModel.m == 2:  # 1D model
            loc = torch.tensor([True, False])  # position only
        else:
            loc = torch.tensor([True] + [False] * (SysModel.m - 1))

    # MSE loss calculation
    for j in range(args.N_T):
        if (allStates):
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j, :, test_lengthMask[j]],
                                               test_target[j, :, test_lengthMask[j]]).item()
            else:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j, :, :], test_target[j, :, :]).item()
        else:  # mask on state
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j, loc, test_lengthMask[j]],
                                               test_target[j, loc, test_lengthMask[j]]).item()
            else:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j, loc, :], test_target[j, loc, :]).item()

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)

    # Confidence interval
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out]