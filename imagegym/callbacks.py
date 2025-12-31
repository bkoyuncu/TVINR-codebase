from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Callback, LightningModule, Trainer
from imagegym.config import cfg
import plotly.express as px
import torch
from torch import Tensor
from PIL import Image
from torch.optim import Optimizer

import numpy as np
import time
import math
from itertools import cycle
from tqdm import tqdm

import matplotlib
# matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import wandb

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
# from source.datasets import *
# from source.models.model_neat import *
COLORS_ = px.colors.qualitative.Bold
# Prepare colors for Plotly
COLORS = []
COLORS_PRED = []
for i in range(len(COLORS_)):
    COLORS.append(COLORS_[i].replace("rgb", "rgba").replace(")", ", 0.6)"))
    COLORS_PRED.append(COLORS_[i*-1].replace("rgb", "rgba").replace(")", ", 0.4)"))
MARKERS = ['rgba(255,0,0,0.7)', 'rgba(0,255,0,0.7)', 'rgba(0,0,255,0.7)']

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Option 2: Grid of all batches
def plot_all_batches(tensor):
    """Plot all batches as a grid of heatmaps"""
    batch_size = tensor.shape[0]
    
    # Calculate grid dimensions
    grid_cols = min(batch_size, 3)
    grid_rows = (batch_size + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3*grid_rows,15))
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(batch_size):
        data = tensor[i, :, :, 0].numpy()
        im = axes[i].imshow(data, cmap='viridis')
        axes[i].set_title(f'Batch {i}')
        
        # Add colorbar for each subplot
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
    
    # Hide unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def swap_axes(x: Tensor) -> Tensor:
    '''
    input shape [Bs, dim_x, T]
    output shape [Bs, T, dim_x]

    '''

    if len(x.shape) == 3:
        return x.permute(0, 2, 1)
    elif len(x.shape) == 4:
        return x.permute(0, 2, 3, 1)
    else:
        raise ValueError("The input tensor should have 3 or 4 dimensions.")

class plotPredictions_mean_var(Callback):
    """
    #add docs
    """

    def __init__(
        self,
        samples=1,
        reconstruct=False
    ) -> None:
        """
        Args:
            log_steps: interval of steps for logging
        """

        super().__init__()
        self.samples = samples
        self.reconstruct = reconstruct #false
    
    @torch.no_grad()
    def compute(self, x_hat_mu_z, x_hat_L, x, mode, epoch, T0, T, wandb_logger, plot_mu = False, observed_mask_all = None, tau = None, split = None, window_len=None, sparsity=None):
        '''
        x_hat_mu_z : [bs, ch, h, w]
        x_hat_L : [bs, ch, h, w, L]
        x : [bs, ch, h, w]
        w is T.
        '''
        x_hat_mu_z = swap_axes(x_hat_mu_z)
        x = swap_axes(x)
        x_hat_L = x_hat_L.permute(0, 4, 2, 3, 1)

        x_hat_mu_z, x = x_hat_mu_z.detach().cpu(), x.detach().cpu()
        bs = x_hat_mu_z.shape[0]
        
        #check the dimensions
        # assert x_hat_mu_z.shape==x.shape

        L = x_hat_L.shape[1]
        x_hat_mean = torch.mean(x_hat_L,dim=1).cpu()
        x_hat_std = torch.std(x_hat_L,dim=1).cpu()
        nan_mask  = torch.isnan(x)
        non_nan_mask = ~torch.isnan(x)
        observed_mask_all = observed_mask_all.cpu()
        
        if torch.sum(nan_mask) > 0:
            non_nan_mask[~nan_mask] = observed_mask_all.flatten()[~nan_mask.flatten()]

        non_nan_mask = non_nan_mask.reshape(*x.shape)

        if len(x.shape) == 4:
            x_hat_mean = x_hat_mean.reshape(*x.shape)
            x_hat_std = x_hat_std.reshape(*x.shape)
            x_hat_mu_z = x_hat_mu_z.reshape(*x.shape)
            x = x.squeeze(-1).permute(0, 2, 1).numpy()
            x_hat_mean = x_hat_mean.squeeze(-1).permute(0, 2, 1).numpy()
            x_hat_std = x_hat_std.squeeze(-1).permute(0, 2, 1).numpy()
            x_hat_mu_z = x_hat_mu_z.squeeze(-1).permute(0, 2, 1).numpy()
            non_nan_mask = non_nan_mask.squeeze(-1).permute(0, 2, 1).numpy()

            # torch.zeros_like(x, dtype=torch.bool).numpy()
            # observed_mask_all # [bs, ch, #points_full]
            # observed_mask_all = observed_mask_all.reshape(-1, x.shape[-1], x.shape[-2])


        x_dim = x_hat_mu_z.shape[-1]
        t_dim = x_hat_mu_z.shape[1]

        #PLOT PRED GT
        
        # fig, axs = plt.subplots(int(bs//4), 4, figsize=(25,25))
        T = torch.ones(bs, dtype=int)*t_dim
        T0 = torch.zeros(bs, dtype=int)*t_dim
        name = f"mean_variance_L_{L}_bs_{bs}"

        if plot_mu:
            x_hat_mean = x_hat_mu_z.numpy()
            x_hat_std = torch.zeros_like(x_hat_mu_z).numpy()
            name = f"mean_L_{L}_bs_{bs}"

        if observed_mask_all is not None:
            observed_mask_all = observed_mask_all.detach().cpu().numpy()
            name = name + f"_tau_{tau}"
            if window_len is not None:
                name = name + f"_window_{window_len}"

        name = name + f"_{split}" + f"_{sparsity}"

        # Set the number of rows and columns for subplots
        # batch size can be anything consider edge cases
        cols = 1
        rows = bs
        if cols == 0:
            cols = 1
            rows = bs

        if cfg.dataset.name in ['P12','P12_new']:

            
            plot_comparison_grid_P12(
                x=x,
                x_hat_mean=x_hat_mean,
                non_nan_mask=non_nan_mask,
                tau=tau,
                name=name,
                wandb_logger=wandb_logger
            )
           # Additionally, plot using Seaborn to compare
            #change nan to np.nan

            # # Create a new figure
            # fig = plt.figure(figsize=(12, 5))

            # # Plot arr1 using Seaborn
            # plt.subplot(1, 2, 1)
            # #check if str has '-1' or '-2' in it

            # if tau == -1 or tau== -2:
            #     x[~non_nan_mask.squeeze(-1).permute((0,2,1))] = np.nan
            # ax1 = sns.heatmap(x[0], cmap="OrRd")
            # ax1.collections[0].cmap.set_bad('blue')  # set NaN values to blue
            # plt.title('Ground Truth')

            # # Plot arr2 using Seaborn
            # plt.subplot(1, 2, 2)
            # ax2 = sns.heatmap(x_hat_mean[0], cmap="OrRd")  # fresh colormap for the second plot
            # ax2.collections[0].cmap.set_bad('blue')  # set NaN values to blue again
            # plt.title('Predictions')
                        
            # # Save the figure to a file
            # save_path = f"{wandb_logger.experiment.dir}/{name}.png"
            # plt.savefig(save_path,)  # saves the figure as a PNG file
            # wandb_logger.experiment.log({name: wandb.Image(save_path)})
            # plt.close()





        else:
            # Create subplots
            fig = make_subplots(rows=rows, cols=cols)
            # Plotting logic
            for i in range(bs):
                row = i // cols + 1
                col = i % cols + 1
                for d in range(x_dim):
                    showlegend = True if i == 0 else False #and d == 0 else False
                    observed_mask = observed_mask_all.reshape(bs, x.shape[-1], x.shape[-2])[i, d, :]
                    # observed_mask = non_nan_mask[i, d, : , 0]
                    if self.reconstruct:
                        x_axis_ticks = np.arange(torch.max(T[i]).detach().cpu().numpy())
                        legendgroup=f'group_{d}_lines'
                        fig.add_trace(go.Scatter(x=x_axis_ticks, y=x[i, :, d], mode='lines', name=f'GT_{d}', line=dict(color=COLORS[d]), showlegend=showlegend, legendgroup=legendgroup),
                                    row=row, col=col)
                        if observed_mask is not None:
                            legendgroup=f'group_{d}_markers'
                            fig.add_trace(go.Scatter(x=x_axis_ticks[observed_mask], y=x[i, [observed_mask], d], mode='markers', name=f'Pred_{d}',
                                                marker=dict(color=MARKERS[d]), showlegend=showlegend, legendgroup=legendgroup),
                                    row=row, col=col)
                        legendgroup=f'group_{d}_pred'
                        fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_mean[i, :, d], mode='lines', name=f'Pred_{d}',
                                                error_y=dict(type='data', array=x_hat_std[i, :, d], visible=True, color=COLORS_PRED[d]),
                                                line=dict(color=COLORS_PRED[d]), showlegend=showlegend, legendgroup=legendgroup),
                                    row=row, col=col)
                    else:
                        x_axis_ticks = np.arange(torch.max(T[i]).detach().cpu().numpy() - torch.max(T0[i]).detach().cpu().numpy())
                        legendgroup=f'group_{d}_lines'
                        fig.add_trace(go.Scatter(x=x_axis_ticks, y=x[i, T0[i]:T[i], d], mode='lines', name=f'GT_{d}', line=dict(color=COLORS[d]), showlegend=showlegend, legendgroup=legendgroup),
                                    row=row, col=col)
                        if observed_mask is not None:
                            legendgroup=f'group_{d}_markers'
                            fig.add_trace(go.Scatter(x=x_axis_ticks[observed_mask], y=x[i, observed_mask, d], mode='markers', name=f'Observed_{d}',
                                                marker=dict(color=MARKERS[d]), showlegend=showlegend, legendgroup=legendgroup),
                                    row=row, col=col)
                        legendgroup=f'group_{d}_pred'
                        fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_mean[i, T0[i]:T[i], d], mode='lines', name=f'Pred_{d}',
                                                error_y=dict(type='data', array=x_hat_std[i, T0[i]:T[i], d], visible=True, color=COLORS_PRED[d]),
                                                line=dict(color=COLORS_PRED[d]), showlegend=showlegend, legendgroup=legendgroup),
                                    row=row, col=col)

            # Update axis labels
            fig.update_xaxes(title_text='time')
            fig.update_yaxes(title_text='value')

            # Update layout
            fig.update_layout(showlegend=True, height=rows*500, width=cols*500, title_text=name)

            # Save the figure as a PNG file and log to WandB
            fig.write_image(f"{wandb_logger.experiment.dir}/{name}.png")
            wandb_logger.experiment.log({name: fig})
            plt.close()



class plotPredictions_mean_var_plotly_dims(Callback):
    """
    Plot the mean and variance of the predictions and the ground truth for the first 12 dimensions of the data.
    """

    def __init__(
        self,
        samples=1,
        reconstruct=False
    ) -> None:
        """
        Args:
            samples (int): Number of samples used to calculate the std and mean of the predictions.
            reconstruct (bool): If the model is reconstructing the history sequence or not.
        """

        super().__init__()
        self.samples = samples
        self.reconstruct = reconstruct
    
    @torch.no_grad()
    def compute(self, x_hat_mu_z, x_hat_L, x, mode, epoch, T0, T, wandb_logger, plot_mu = False, observed_mask = None, tau = None, split = None):
        # // BUG this is loading twice we need to use the existing one. #
        #put them as from bs dim T to bs x T x x_dim
        x_hat_mu_z = swap_axes(x_hat_mu_z)
        x = swap_axes(x)
        x_hat_L = x_hat_L.permute(0, 3, 2, 1)

        x_hat_mu_z, x = x_hat_mu_z.detach().cpu(), x.detach().cpu()
        bs = x_hat_mu_z.shape[0]
        x_dim = x_hat_mu_z.shape[-1]
        rows =  min(x_dim, 12)
        num_plots = min(x_dim, 12)
        t_dim = x_hat_mu_z.shape[1]
        old_bs = x.shape[0]
        
        #check the dimensions
        assert x_hat_mu_z.shape==x.shape

        L = x_hat_L.shape[1]
        x_hat_mean = torch.mean(x_hat_L,dim=1).cpu().numpy()
        x_hat_std = torch.std(x_hat_L,dim=1).cpu().numpy()

        T = torch.ones(bs, dtype=int)*t_dim
        T0 = torch.zeros(bs, dtype=int)*t_dim
        # name = f"mean_variance_L_{L}_bs_{bs}"

        cols = 1
        #PLOT PRED GT
        fig = make_subplots(rows=rows, cols=cols)

        for dim in range(num_plots):
            row = dim // cols + 1
            col = dim % cols + 1
            showlegend = True if dim == 0 else False         
            if self.reconstruct:
                x_axis_ticks = np.arange(0, T[0].item())
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x[0, :T[0], dim], mode='lines', name=f'GT', line=dict(color=COLORS[0]), showlegend=showlegend),
                                            row = row,
                                            col = col)
                if observed_mask is not None:
                        fig.add_trace(go.Scatter(x=x_axis_ticks[observed_mask], y=x[i, observed_mask, dim], mode='markers', name=f'Observed', marker=dict(color=MARKERS[0]), showlegend=showlegend),
                                            row = row,
                                            col = col)


                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_mean[0, :T[0], dim], mode='lines', name=f'Pred',
                                        error_y=dict(type='data', array=x_hat_std[0, :T[0], dim], visible=True, color=COLORS_PRED[0]),
                                        line=dict(color=COLORS_PRED[0]), showlegend=showlegend),
                                            row = row,
                                            col = col)

            else:
                x_axis_ticks = np.arange(T0[0].item(), T[0].item())
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x[0, T0[0]:T[0], dim], mode='lines', name=f'GT', line=dict(color=COLORS[0]), showlegend=showlegend),
                                            row = row,
                                            col = col)

                if observed_mask is not None:
                        fig.add_trace(go.Scatter(x=x_axis_ticks[observed_mask], y=x[0, observed_mask, dim], mode='markers', name=f'Observed', marker=dict(color=MARKERS[0]), showlegend=showlegend),
                                            row = row,
                                            col = col)


                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_mean[0, T0[0]:T[0], dim], mode='lines', name=f'Pred',
                                        error_y=dict(type='data', array=x_hat_std[0, T0[0]:T[0], dim], visible=True, color=COLORS_PRED[0]),
                                        line=dict(color=COLORS_PRED[0]), showlegend=showlegend),
                                            row = row,
                                            col = col)


    
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='value')
        
        name = f"mean_variance_L_{L}_dim"
        if observed_mask is not None:
            name = name + f"_tau_{tau}"
        name = name + f"_{split}"

        fig.update_layout(showlegend=True, height=5000, width=2500, title_text=name)
        # fig.show()

        # Save the figure as a PNG file
        fig.write_image(f"{wandb_logger.experiment.dir}/{name}.png")
        # plt.savefig(f"{wandb_logger.experiment.dir}/{name}.png")
        wandb_logger.experiment.log({name: fig})
        plt.close()

        
        #--------- plot the sum over all dimensions ------------ #
        rows = min(old_bs, 12)
        num_plots = min(old_bs, 12)
        
        # sum along the last dimension
        x_hat_mu_z_sum = torch.sum(x_hat_mu_z, dim=-1) # [bs, Tm]
        x_hat_L_sum = torch.sum(x_hat_L, dim=-1) # [bs, L, Tm]
        x_sum = torch.sum(x, dim=-1) # [bs, Tm]
        x_hat_mu_z_sum, x_hat_L_sum, x_sum = x_hat_mu_z_sum.detach().cpu(), x_hat_L_sum.detach().cpu(), x_sum.detach().cpu()
        
        x_hat_sum_mean = torch.mean(x_hat_L_sum,dim=1).cpu().numpy() # [bs, Tm] 
        x_hat_sum_std = torch.std(x_hat_L_sum,dim=1).cpu().numpy() # [bs, Tm]

        #check the dimensions
        assert x_hat_mu_z_sum.shape==x_sum.shape

        L = x_hat_L_sum.shape[1]
        cols = 1
        fig = make_subplots(rows=rows, cols=1)
        
        for i in range(num_plots):
            row = i // cols + 1
            col = i % cols + 1
            showlegend = True if i == 0 else False
            if self.reconstruct:
                x_axis_ticks = np.arange(0, T[i].item())
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_sum[i, :T[i]], mode='lines', name=f'GT_sum_dims_{x_dim}', line=dict(color=COLORS[0]), showlegend=showlegend),
                                            row = row, 
                                            col = col)
                if observed_mask is not None:
                        fig.add_trace(go.Scatter(x=x_axis_ticks[observed_mask], y=x_sum[i, observed_mask], mode='markers', name=f'Observed_sum', marker=dict(color=MARKERS[0]), showlegend=showlegend),
                                            row = row, 
                                            col = col)

                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_sum_mean[i, :T[i]], mode='lines', name=f'Pred_sum_dims_{x_dim}',
                                        error_y=dict(type='data', array=x_hat_sum_std[i, :T[i]], visible=True, color=COLORS_PRED[0]),
                                        line=dict(color=COLORS_PRED[0]), showlegend=showlegend),
                                            row = row, 
                                            col = col)
            else:
                x_axis_ticks = np.arange(T0[i].item(), T[i].item())
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_sum[i, T0[i]:T[i]], mode='lines', name=f'GT_sum_dims_{x_dim}', line=dict(color=COLORS[0]), showlegend=showlegend),
                                            row = row, 
                                            col = col)
                if observed_mask is not None:
                        fig.add_trace(go.Scatter(x=x_axis_ticks[observed_mask], y=x_sum[i, observed_mask], mode='markers', name=f'Observed_sum', marker=dict(color=MARKERS[0]), showlegend=showlegend),
                                            row = row, 
                                            col = col)

                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_sum_mean[i, T0[i]:T[i]], mode='lines', name=f'Pred_sum_dims_{x_dim}',
                                        error_y=dict(type='data', array=x_hat_sum_std[i, T0[i]:T[i]], visible=True, color=COLORS_PRED[0]),
                                        line=dict(color=COLORS_PRED[0]), showlegend=showlegend),
                                            row = row, 
                                            col = col)

    
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='value')
        
        name = f"mean_variance_sum_dim_{x_dim}_L_{L}_bs_{old_bs}"
        if observed_mask is not None:
            name = name + f"_tau_{tau}"
        name = name + f"_{split}"
        fig.update_layout(showlegend=True, height=5000, width=2500, title_text=name)
        # fig.show()]
        # Save the figure as a PNG file
        fig.write_image(f"{wandb_logger.experiment.dir}/{name}.png")
        # plt.savefig(f"{wandb_logger.experiment.dir}/{name}.png")
        wandb_logger.experiment.log({name: fig})
        plt.close()

def plot_comparison_grid_P12(x, x_hat_mean, non_nan_mask=None, tau=None, name="comparison", wandb_logger=None, max_rows=4, max_cols=4):
    """
    Plot comparison grid of ground truth vs predictions.
    
    Args:
        x: Ground truth values
        x_hat_mean: Predicted values
        non_nan_mask: Mask for non-NaN values
        tau: Tau value for special handling of -1 or -2 cases
        name: Name for saving the plot
        wandb_logger: WandB logger instance
        max_rows: Maximum number of rows (default 4)
        max_cols: Maximum number of columns (default 4)
    """
    # Determine number of elements to plot (minimum of len(x) and 8)
    n_elements = min(x.shape[0], 8)  # Assuming x is batched with shape [batch, seq_len, ...]
    n_rows = (n_elements + 1) // 2  # Each row needs 2 elements (x and x_hat)
    n_rows = min(n_rows, max_rows)
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(16, 3 * n_rows))
    
    # Handle NaN values if tau is -1 or -2
    # if tau in [-1, -2, -3] and non_nan_mask is not None:
    x_plot = x.copy()
    x_hat_mean_plot = x_hat_mean.copy()  # Create a copy to avoid modifying original data
    #change them to numpy arrays
    # x_plot = x_plot.numpy()
    # x_hat_mean = x_hat_mean.numpy()
    #check if nan

    if 'full' not in name:
        x_hat_mean_plot[np.isnan(x_plot)] = np.nan
    x_plot[~non_nan_mask] = np.nan
    # else:
    #     x_plot = x
    
    # Plot each pair of ground truth and prediction
    for i in range(n_elements):
        # Calculate subplot position
        row = i // 2
        col = (i % 2) * 2  # Multiply by 2 because each row has x and x_hat
        
        if row < max_rows:
            # Plot ground truth
            plt.subplot(n_rows, max_cols, row * max_cols + col + 1)
            ax1 = sns.heatmap(x_plot[i], 
                            cmap="OrRd")
            ax1.collections[0].cmap.set_bad('blue')
            plt.title(f'Ground Truth x[{i}]')
            
            # Plot prediction
            plt.subplot(n_rows, max_cols, row * max_cols + col + 2)
            ax2 = sns.heatmap(x_hat_mean_plot[i], 
                            cmap="OrRd")
            ax2.collections[0].cmap.set_bad('blue')
            plt.title(f'Prediction x_hat_mean[{i}]')
    
    # Adjust layout
    plt.tight_layout()
    # plt.subplots_adjust(hspace=1, wspace=1)  # Adjust these values as needed

    # Save and log if wandb_logger is provided
    if wandb_logger is not None:
        save_path = f"{wandb_logger.experiment.dir}/{name}.png"
        plt.savefig(save_path, bbox_inches='tight')
        wandb_logger.experiment.log({name: wandb.Image(save_path)})

    
    plt.close()
