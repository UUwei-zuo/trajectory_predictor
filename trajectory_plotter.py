import torch
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 1E4
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


class TrajectoryPlotter:
    """Unified class for all trajectory plotting functionality"""

    def __init__(self, folder_name, model_name):
        """
        Initialize the plotter

        Args:
            folder_name: Directory to save plots
            model_name: Name of the model being plotted
        """
        self.folder_name = folder_name
        self.model_name = model_name

        # Color and legend definitions
        self.colors = {
            'train': '-ro',
            'validation': 'darkorange',
            'test': 'k-',
            'kf': 'b-',
            'rts': 'g-',
            'true': 'k-'
        }

        self.legends = {
            'kf': ["KalmanNet - Train", "KalmanNet - Validation", "KalmanNet - Test", "Kalman Filter"],
            'rts': ["RTSNet - Train", "RTSNet - Validation", "RTSNet - Test", "RTS Smoother", "Kalman Filter"],
            'ekf': ["RTSNet - Train", "RTSNet - Validation", "RTSNet - Test", "RTS", "EKF"]
        }

    def plot_histogram(self, data_dict, title_suffix="Histogram [dB]", filename="plt_hist_dB"):
        """
        Create a histogram plot for multiple data series

        Args:
            data_dict: Dictionary of {label: data_array}
            title_suffix: Suffix for the plot title
            filename: Name for the saved file (without path)
        """
        file_name = f"{self.folder_name}{filename}"

        # Convert to dB if not already
        for key, value in data_dict.items():
            if torch.min(value) > 0:  # Assuming positive values need conversion to dB
                data_dict[key] = 10 * torch.log10(value)

        # Create the plot
        ax = sns.displot(
            data_dict,
            kind="kde",
            common_norm=False,
            palette=["blue", "orange", "g"],
            linewidth=1,
        )

        plt.title(f"{self.model_name}: {title_suffix}")
        plt.xlabel('MSE Loss Value [dB]')
        plt.ylabel('Percentage')
        sns.move_legend(ax, "upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    def plot_training_progress(self, epochs, train_data, val_data, test_avg,
                               baseline_data=None, baseline_label=None,
                               second_baseline=None, second_label=None,
                               legend_type='kf', title_suffix="MSE Loss [dB] - per Epoch"):
        """
        Plot training progress over epochs

        Args:
            epochs: Number of epochs to plot
            train_data: Training loss values
            val_data: Validation loss values
            test_avg: Average test loss (will be plotted as horizontal line)
            baseline_data: Optional baseline comparison (e.g., Kalman Filter)
            baseline_label: Label for the baseline
            second_baseline: Optional second baseline (e.g., RTS Smoother)
            second_label: Label for the second baseline
            legend_type: Type of legends to use ('kf', 'rts', or 'ekf')
            title_suffix: Suffix for the plot title
        """
        legends = self.legends.get(legend_type, self.legends['kf'])
        file_name = f"{self.folder_name}plt_epochs_dB"
        font_size = 32

        # Figure
        plt.figure(figsize=(25, 10))

        # x_axis
        x_plt = range(0, epochs)

        # Train
        plt.plot(x_plt, train_data[:epochs], self.colors['train'], label=legends[0])

        # CV
        plt.plot(x_plt, val_data[:epochs], self.colors['validation'], label=legends[1])

        # Test
        plt.plot(x_plt, test_avg * torch.ones(epochs), self.colors['test'], label=legends[2])

        # First baseline (e.g., RTS or KF)
        if baseline_data is not None:
            plt.plot(x_plt, baseline_data * torch.ones(epochs), self.colors['kf'],
                     label=baseline_label if baseline_label else legends[3])

        # Second baseline (e.g., KF when RTS is first)
        if second_baseline is not None:
            plt.plot(x_plt, second_baseline * torch.ones(epochs), self.colors['rts'],
                     label=second_label if second_label else legends[4])

        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.xlabel('Number of Training Epochs', fontsize=font_size)
        plt.ylabel('MSE Loss Value [dB]', fontsize=font_size)
        plt.grid(True)
        plt.title(f"{self.model_name}: {title_suffix}", fontsize=font_size)
        plt.savefig(file_name)
        plt.close()

    def plot_component_vs_time(self, test_target, knet_out, kf_out, dim, file_name,
                               knet_label="KalmanNet", kf_label="Kalman Filter", truth_label="Ground Truth"):
        """
        Plot single component (position/velocity/acceleration) over time

        Args:
            test_target: Ground truth trajectory data
            knet_out: KalmanNet output
            kf_out: Kalman Filter output
            dim: Dimension to plot (0=position, 1=velocity, 2=acceleration)
            file_name: Name for the saved file (with path)
            knet_label: Label for KalmanNet
            kf_label: Label for Kalman Filter
            truth_label: Label for ground truth
        """
        legend = [knet_label, truth_label, kf_label]
        font_size = 14
        T_test = knet_out[0].size()[1]
        x_plt = range(0, T_test)

        dim_labels = ["position", "velocity", "acceleration"]
        if dim < 0 or dim >= len(dim_labels):
            print("Invalid dimension")
            return

        # Extract data and ensure it's numpy arrays
        try:
            # Try to move tensors to CPU if they're on GPU
            knet_data = knet_out[0][dim, :].detach().cpu().numpy()
            truth_data = test_target[0][dim, :].detach().cpu().numpy()
            kf_data = kf_out[0][dim, :].detach().cpu().numpy()
        except:
            # Fallback if .cpu() fails (data already on CPU)
            knet_data = knet_out[0][dim, :].detach().numpy()
            truth_data = test_target[0][dim, :].detach().numpy()
            kf_data = kf_out[0][dim, :].detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(x_plt, knet_data, 'r-', label=legend[0])
        plt.plot(x_plt, truth_data, 'k-', label=legend[1])
        plt.plot(x_plt, kf_data, 'b-', label=legend[2])
        plt.legend(fontsize=font_size)
        plt.xlabel('Time Step', fontsize=font_size)
        plt.ylabel(dim_labels[dim], fontsize=font_size)
        plt.grid(True)
        plt.savefig(file_name)
        plt.close()

    def plot_2D_trajectory(self, test_target, kf_out, knet_out, file_name,
                           title="2D Trajectory Comparison",
                           x_index=0, y_index=3,
                           show_markers=True, show_covariance=False,
                           show_ground_truth_points=False):
        """
        Plot 2D trajectory in x-y plane

        Args:
            test_target: Ground truth trajectory data [batch, state_dim, time]
            kf_out: Kalman Filter output [batch, state_dim, time]
            knet_out: KalmanNet output [batch, state_dim, time]
            file_name: Name for the saved file (with path)
            title: Plot title
            x_index: Index of x position in state vector (default 0)
            y_index: Index of y position in state vector (default 3)
            show_markers: Whether to show markers at regular intervals
            show_covariance: Whether to show covariance ellipses (if available)
        """
        # Create figure
        plt.figure(figsize=(12, 10))

        # Extract trajectory data for the first batch item
        try:
            # Try to move tensors to CPU if they're on GPU
            x_true = test_target[0][x_index, :].detach().cpu().numpy()
            y_true = test_target[0][y_index, :].detach().cpu().numpy()

            x_kf = kf_out[0][x_index, :].detach().cpu().numpy()
            y_kf = kf_out[0][y_index, :].detach().cpu().numpy()

            x_knet = knet_out[0][x_index, :].detach().cpu().numpy()
            y_knet = knet_out[0][y_index, :].detach().cpu().numpy()
        except:
            # Fallback if .cpu() fails (data already on CPU)
            x_true = test_target[0][x_index, :].detach().numpy()
            y_true = test_target[0][y_index, :].detach().numpy()

            x_kf = kf_out[0][x_index, :].detach().numpy()
            y_kf = kf_out[0][y_index, :].detach().numpy()

            x_knet = knet_out[0][x_index, :].detach().numpy()
            y_knet = knet_out[0][y_index, :].detach().numpy()

        # Plot trajectories
        plt.plot(x_true, y_true, 'k-', linewidth=2, label='Ground Truth')

        # Add ground truth points if requested
        if show_ground_truth_points:
            plt.plot(x_true, y_true, 'ko', markersize=3, alpha=0.5)

        # Add markers at regular intervals if requested
        if show_markers:
            # Calculate marker interval based on trajectory length
            marker_interval = max(1, len(x_true) // 20)  # Show ~20 markers

            plt.plot(x_kf[::marker_interval], y_kf[::marker_interval], 'bs', markersize=6, alpha=0.7)
            plt.plot(x_kf, y_kf, 'b-', linewidth=1.5, label='Kalman Filter')

            plt.plot(x_knet[::marker_interval], y_knet[::marker_interval], 'ro', markersize=6, alpha=0.7)
            plt.plot(x_knet, y_knet, 'r-', linewidth=1.5, label='KalmanNet')
        else:
            plt.plot(x_kf, y_kf, 'b-', linewidth=1.5, label='Kalman Filter')
            plt.plot(x_knet, y_knet, 'r-', linewidth=1.5, label='KalmanNet')

        # Mark start and end points
        plt.plot(x_true[0], y_true[0], 'ko', markersize=10, label='Start')
        plt.plot(x_true[-1], y_true[-1], 'kx', markersize=10, label='End')

        # Add labels and legend
        plt.xlabel('X Position', fontsize=14)
        plt.ylabel('Y Position', fontsize=14)
        plt.title(title, fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)

        # Make axes equal to preserve trajectory shape
        plt.axis('equal')

        # Save figure
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_state_components(self, test_target, kf_out, knet_out, file_name_prefix,
                              state_labels=None):
        """
        Plot each state component over time for comparison

        Args:
            test_target: Ground truth trajectory data [batch, state_dim, time]
            kf_out: Kalman Filter output [batch, state_dim, time]
            knet_out: KalmanNet output [batch, state_dim, time]
            file_name_prefix: Prefix for saved files (with path)
            state_labels: Labels for each state component (default: auto-generated)
        """
        # Get state dimension and time steps
        state_dim = test_target[0].shape[0]
        T = test_target[0].shape[1]

        # Generate default state labels if not provided
        if state_labels is None:
            if state_dim == 6:  # CA model with x,y
                state_labels = ['x position', 'x velocity', 'x acceleration',
                                'y position', 'y velocity', 'y acceleration']
            elif state_dim == 4:  # CV model with x,y
                state_labels = ['x position', 'x velocity', 'y position', 'y velocity']
            else:
                state_labels = [f'State {i}' for i in range(state_dim)]

        # Time axis
        t = np.arange(T)

        # Extract data
        try:
            # Try to move tensors to CPU if they're on GPU
            true_data = test_target[0].detach().cpu().numpy()
            kf_data = kf_out[0].detach().cpu().numpy()
            knet_data = knet_out[0].detach().cpu().numpy()
        except:
            # Fallback if .cpu() fails (data already on CPU)
            true_data = test_target[0].detach().numpy()
            kf_data = kf_out[0].detach().numpy()
            knet_data = knet_out[0].detach().numpy()

        # Plot each state component
        for i in range(state_dim):
            plt.figure(figsize=(12, 6))

            plt.plot(t, true_data[i, :], 'k-', linewidth=2, label='Ground Truth')
            plt.plot(t, kf_data[i, :], 'b-', linewidth=1.5, label='Kalman Filter')
            plt.plot(t, knet_data[i, :], 'r-', linewidth=1.5, label='KalmanNet')

            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel(state_labels[i], fontsize=14)
            plt.title(f'{state_labels[i]} over Time', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=12)

            # Save figure
            component_filename = f"{file_name_prefix}_{i}_{state_labels[i].replace(' ', '_')}.png"
            plt.savefig(component_filename, bbox_inches='tight', dpi=300)
            plt.close()

        # Create combined plots for CA model
        if state_dim == 6:  # CA model
            self._plot_combined_components(t, true_data, kf_data, knet_data, file_name_prefix)

    def _plot_combined_components(self, t, true_data, kf_data, knet_data, file_name_prefix):
        """Helper method to create combined component plots for CA model"""
        # Position plot
        plt.figure(figsize=(12, 6))
        plt.plot(t, true_data[0, :], 'k-', linewidth=2, label='True x')
        plt.plot(t, true_data[3, :], 'k--', linewidth=2, label='True y')
        plt.plot(t, kf_data[0, :], 'b-', linewidth=1.5, label='KF x')
        plt.plot(t, kf_data[3, :], 'b--', linewidth=1.5, label='KF y')
        plt.plot(t, knet_data[0, :], 'r-', linewidth=1.5, label='KNet x')
        plt.plot(t, knet_data[3, :], 'r--', linewidth=1.5, label='KNet y')
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Position', fontsize=14)
        plt.title('Position Components over Time', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.savefig(f"{file_name_prefix}_position_combined.png", bbox_inches='tight', dpi=300)
        plt.close()

        # Velocity plot
        plt.figure(figsize=(12, 6))
        plt.plot(t, true_data[1, :], 'k-', linewidth=2, label='True vx')
        plt.plot(t, true_data[4, :], 'k--', linewidth=2, label='True vy')
        plt.plot(t, kf_data[1, :], 'b-', linewidth=1.5, label='KF vx')
        plt.plot(t, kf_data[4, :], 'b--', linewidth=1.5, label='KF vy')
        plt.plot(t, knet_data[1, :], 'r-', linewidth=1.5, label='KNet vx')
        plt.plot(t, knet_data[4, :], 'r--', linewidth=1.5, label='KNet vy')
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Velocity', fontsize=14)
        plt.title('Velocity Components over Time', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.savefig(f"{file_name_prefix}_velocity_combined.png", bbox_inches='tight', dpi=300)
        plt.close()

        # Acceleration plot
        plt.figure(figsize=(12, 6))
        plt.plot(t, true_data[2, :], 'k-', linewidth=2, label='True ax')
        plt.plot(t, true_data[5, :], 'k--', linewidth=2, label='True ay')
        plt.plot(t, kf_data[2, :], 'b-', linewidth=1.5, label='KF ax')
        plt.plot(t, kf_data[5, :], 'b--', linewidth=1.5, label='KF ay')
        plt.plot(t, knet_data[2, :], 'r-', linewidth=1.5, label='KNet ax')
        plt.plot(t, knet_data[5, :], 'r--', linewidth=1.5, label='KNet ay')
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Acceleration', fontsize=14)
        plt.title('Acceleration Components over Time', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.savefig(f"{file_name_prefix}_acceleration_combined.png", bbox_inches='tight', dpi=300)
        plt.close()

    #######################################################
    # Compatibility methods for backward compatibility
    #######################################################

    def plotTraj_CA(self, test_target, kf_out, knet_out, dim, file_name):
        """Compatibility method for plotTraj_CA"""
        self.plot_component_vs_time(test_target, knet_out, kf_out, dim, file_name)

    def NNPlot_epochs(self, N_Epochs_plt, MSE_KF_dB_avg, MSE_test_dB_avg,
                      MSE_cv_dB_epoch, MSE_train_dB_epoch):
        """Compatibility method for KalmanNet training plots"""
        self.plot_training_progress(
            epochs=N_Epochs_plt,
            train_data=MSE_train_dB_epoch,
            val_data=MSE_cv_dB_epoch,
            test_avg=MSE_test_dB_avg,
            baseline_data=MSE_KF_dB_avg,
            legend_type='kf'
        )

    def NNPlot_Hist(self, MSE_KF_linear_arr, MSE_RTS_data_linear_arr=None, MSE_Net_linear_arr=None):
        """Compatibility method for histogram plots"""
        data_dict = {'Kalman Filter': MSE_KF_linear_arr}

        if MSE_Net_linear_arr is not None:
            data_dict[self.model_name] = MSE_Net_linear_arr

        if MSE_RTS_data_linear_arr is not None:
            data_dict['RTS Smoother'] = MSE_RTS_data_linear_arr

        self.plot_histogram(data_dict)

    def plot_component_norm_vs_time(self, test_target, knet_out, kf_out, file_name_prefix,
                                    title_prefix="Trajectory Component Norms"):
        """
        Plot L2 norm of position, velocity, and acceleration components over time

        Args:
            test_target: Ground truth trajectory data [batch, state_dim, time]
            knet_out: KalmanNet output [batch, state_dim, time]
            kf_out: Kalman Filter output [batch, state_dim, time]
            file_name_prefix: Prefix for saved files (with path)
            title_prefix: Prefix for plot titles
        """
        # Get state dimension and time steps
        state_dim = test_target[0].shape[0]
        T = test_target[0].shape[1]

        # Time axis
        t = np.arange(T)

        # Extract data
        try:
            # Try to move tensors to CPU if they're on GPU
            true_data = test_target[0].detach().cpu().numpy()
            kf_data = kf_out[0].detach().cpu().numpy()
            knet_data = knet_out[0].detach().cpu().numpy()
        except:
            # Fallback if .cpu() fails (data already on CPU)
            true_data = test_target[0].detach().numpy()
            kf_data = kf_out[0].detach().numpy()
            knet_data = knet_out[0].detach().numpy()

        if state_dim == 6:  # CA model with x,y
            # Calculate L2 norms
            # Position norm (indices 0 and 3)
            pos_true_norm = np.sqrt(true_data[0, :] ** 2 + true_data[3, :] ** 2)
            pos_kf_norm = np.sqrt(kf_data[0, :] ** 2 + kf_data[3, :] ** 2)
            pos_knet_norm = np.sqrt(knet_data[0, :] ** 2 + knet_data[3, :] ** 2)

            # Velocity norm (indices 1 and 4)
            vel_true_norm = np.sqrt(true_data[1, :] ** 2 + true_data[4, :] ** 2)
            vel_kf_norm = np.sqrt(kf_data[1, :] ** 2 + kf_data[4, :] ** 2)
            vel_knet_norm = np.sqrt(knet_data[1, :] ** 2 + knet_data[4, :] ** 2)

            # Plot position norm
            plt.figure(figsize=(12, 6))
            plt.plot(t, pos_true_norm, 'k-', linewidth=2, label='Ground Truth')
            plt.plot(t, pos_kf_norm, 'b-', linewidth=1.5, label='Kalman Filter')
            plt.plot(t, pos_knet_norm, 'r-', linewidth=1.5, label='KalmanNet')
            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel('Position Norm', fontsize=14)
            plt.title(f'{title_prefix}: Position Magnitude', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(f"{file_name_prefix}_position_norm.png", bbox_inches='tight', dpi=300)
            plt.close()

            # Plot velocity norm
            plt.figure(figsize=(12, 6))
            plt.plot(t, vel_true_norm, 'k-', linewidth=2, label='Ground Truth')
            plt.plot(t, vel_kf_norm, 'b-', linewidth=1.5, label='Kalman Filter')
            plt.plot(t, vel_knet_norm, 'r-', linewidth=1.5, label='KalmanNet')
            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel('Velocity Norm', fontsize=14)
            plt.title(f'{title_prefix}: Velocity Magnitude', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(f"{file_name_prefix}_velocity_norm.png", bbox_inches='tight', dpi=300)
            plt.close()

        elif state_dim == 4:  # CV model with x,y
            # Calculate L2 norms
            # Position norm (indices 0 and 2)
            pos_true_norm = np.sqrt(true_data[0, :] ** 2 + true_data[2, :] ** 2)
            pos_kf_norm = np.sqrt(kf_data[0, :] ** 2 + kf_data[2, :] ** 2)
            pos_knet_norm = np.sqrt(knet_data[0, :] ** 2 + knet_data[2, :] ** 2)

            # Velocity norm (indices 1 and 3)
            vel_true_norm = np.sqrt(true_data[1, :] ** 2 + true_data[3, :] ** 2)
            vel_kf_norm = np.sqrt(kf_data[1, :] ** 2 + kf_data[3, :] ** 2)
            vel_knet_norm = np.sqrt(knet_data[1, :] ** 2 + knet_data[3, :] ** 2)

            # Plot position norm
            plt.figure(figsize=(12, 6))
            plt.plot(t, pos_true_norm, 'k-', linewidth=2, label='Ground Truth')
            plt.plot(t, pos_kf_norm, 'b-', linewidth=1.5, label='Kalman Filter')
            plt.plot(t, pos_knet_norm, 'r-', linewidth=1.5, label='KalmanNet')
            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel('Position Norm', fontsize=14)
            plt.title(f'{title_prefix}: Position Magnitude', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(f"{file_name_prefix}_position_norm.png", bbox_inches='tight', dpi=300)
            plt.close()

            # Plot velocity norm
            plt.figure(figsize=(12, 6))
            plt.plot(t, vel_true_norm, 'k-', linewidth=2, label='Ground Truth')
            plt.plot(t, vel_kf_norm, 'b-', linewidth=1.5, label='Kalman Filter')
            plt.plot(t, vel_knet_norm, 'r-', linewidth=1.5, label='KalmanNet')
            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel('Velocity Norm', fontsize=14)
            plt.title(f'{title_prefix}: Velocity Magnitude', fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(f"{file_name_prefix}_velocity_norm.png", bbox_inches='tight', dpi=300)
            plt.close()

        else:
            print(f"Unsupported state dimension: {state_dim}. Expected 4 (CV model) or 6 (CA model).")

    def plotTraj_CA_norm(self, test_target, kf_out, knet_out, file_name_prefix):
        """
        Compatibility method for plotting trajectory component norms (L2 norms)

        Args:
            test_target: Ground truth trajectory data
            kf_out: Kalman Filter output
            knet_out: KalmanNet output
            file_name_prefix: Prefix for saved files (with path)
        """
        self.plot_component_norm_vs_time(
            test_target=test_target,
            knet_out=knet_out,
            kf_out=kf_out,
            file_name_prefix=file_name_prefix
        )