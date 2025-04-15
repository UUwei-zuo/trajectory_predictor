"""
The file contains utility functions for the simulations.
"""
import torch
import numpy as np

# def DataGen(args, SysModel_data, fileName):
#
#     ##################################
#     ### Generate Training Sequence ###
#     ##################################
#     SysModel_data.GenerateBatch(args, args.N_E, args.T, randomInit=args.randomInit_train)
#     train_input = SysModel_data.Input
#     train_target = SysModel_data.Target
#     ### init conditions ###
#     train_init = SysModel_data.m1x_0_batch #size: N_E x m x 1
#     ### length mask ###
#     if args.randomLength:
#         train_lengthMask = SysModel_data.lengthMask
#
#     ####################################
#     ### Generate Validation Sequence ###
#     ####################################
#     SysModel_data.GenerateBatch(args, args.N_CV, args.T, randomInit=args.randomInit_cv)
#     cv_input = SysModel_data.Input
#     cv_target = SysModel_data.Target
#     cv_init = SysModel_data.m1x_0_batch #size: N_CV x m x 1
#     ### length mask ###
#     if args.randomLength:
#         cv_lengthMask = SysModel_data.lengthMask
#
#     ##############################
#     ### Generate Test Sequence ###
#     ##############################
#     SysModel_data.GenerateBatch(args, args.N_T, args.T_test, randomInit=args.randomInit_test)
#     test_input = SysModel_data.Input
#     test_target = SysModel_data.Target
#     test_init = SysModel_data.m1x_0_batch #size: N_T x m x 1
#     ### length mask ###
#     if args.randomLength:
#         test_lengthMask = SysModel_data.lengthMask
#
#     #################
#     ### Save Data ###
#     #################
#     if(args.randomLength):
#         torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask], fileName)
#     else:
#         torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init], fileName)

def DataGen(args, SysModel, fileName, initial_state_config):
    """
    Generate synthetic data for training, validation, and testing.
    Args:
        args: Configuration arguments
        SysModel: System model for data generation
        fileName: Path to save generated data
        initial_state_config: Dictionary containing initial state configuration
    """
    # Get dimensions
    m = SysModel.m  # State dimension
    n = SysModel.n  # Observation dimension

    def generate_initial_conditions(num_samples, is_random_init):
        """Helper function to generate initial conditions"""
        init_conditions = torch.zeros(num_samples, m, 1)

        for i in range(num_samples):
            # Generate random heading angle (0-360 degrees)
            heading_angle = 2 * np.pi * torch.rand(1).item()

            # Get configuration values
            vel_magnitude = initial_state_config['means']['velocity']['mag']
            accel_magnitude = initial_state_config['means']['acceleration']['mag']
            pos_x_mean = initial_state_config['means']['position']['x']
            pos_y_mean = initial_state_config['means']['position']['y']

            # Calculate components based on heading angle
            vel_x = vel_magnitude * torch.cos(torch.tensor(heading_angle))
            vel_y = vel_magnitude * torch.sin(torch.tensor(heading_angle))
            accel_x = accel_magnitude * torch.cos(torch.tensor(heading_angle))
            accel_y = accel_magnitude * torch.sin(torch.tensor(heading_angle))

            # Create initial state with means
            init_state = torch.tensor([
                pos_x_mean, vel_x, accel_x,
                pos_y_mean, vel_y, accel_y
            ])

            # Add random perturbations based on standard deviations
            if is_random_init:
                pos_std = initial_state_config['std_devs']['position']
                vel_std = initial_state_config['std_devs']['velocity']
                acc_std = initial_state_config['std_devs']['acceleration']

                perturbation = torch.zeros(6)
                # Position perturbation
                perturbation[0] = torch.randn(1) * pos_std
                perturbation[3] = torch.randn(1) * pos_std
                # Velocity perturbation
                perturbation[1] = torch.randn(1) * vel_std
                perturbation[4] = torch.randn(1) * vel_std
                # Acceleration perturbation
                perturbation[2] = torch.randn(1) * acc_std
                perturbation[5] = torch.randn(1) * acc_std

                init_state += perturbation

            init_conditions[i, :, 0] = init_state

        return init_conditions

    # Generate training data
    train_init = generate_initial_conditions(args.N_E, args.randomInit_train)
    SysModel.GenerateBatch(args, args.N_E, args.T, randomInit=args.randomInit_train)
    train_input = SysModel.Input
    train_target = SysModel.Target

    # Generate validation data
    cv_init = generate_initial_conditions(args.N_CV, args.randomInit_cv)
    SysModel.GenerateBatch(args, args.N_CV, args.T, randomInit=args.randomInit_cv)
    cv_input = SysModel.Input
    cv_target = SysModel.Target

    # Generate test data
    test_init = generate_initial_conditions(args.N_T, args.randomInit_test)

    # Use the same initial conditions for trajectory generation
    SysModel.m1x_0_batch = test_init  # Set the initial state
    SysModel.GenerateBatch(args, args.N_T, args.T_test, randomInit=args.randomInit_test)
    test_input = SysModel.Input
    test_target = SysModel.Target

    # Save all data
    torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,
                train_init, cv_init, test_init], fileName)