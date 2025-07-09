"""
Author: Mélanie Gaillochet
"""
import matplotlib.pyplot as plt
from skimage.measure import find_contours


def plot_data_pred_volume(data, target, pred, slice_num, weak_label=None, plot_type='contour', title='', vmin=0, vmax=None):
    """ We plot the data, target and pred of a volume at a given slice

    Args:
        data (tensor of shape [BS, num_slices in volume, H,  W]): 
        target (tensor of shape [BS, num_slices in volume, H,  W]): 
        pred (tensor of shape [BS, num_slices in volume, H,  W]): 
        plot_ype (str, optional): 'contour' or 'image'
        title (str, optional): Indice of volume we plot
    """

    fig = plt.figure(figsize=(10, 10), layout="constrained")

    cur_data = data[0, slice_num, :, :].detach().cpu().numpy()
    cur_target = target[0, slice_num, :, :].detach().cpu().numpy()
    cur_pred = pred[0, slice_num, :, :].detach().cpu().numpy()

    if plot_type == 'contour':
        ax = _plot_contour(fig, 1, 1, cur_data, cur_target, cur_pred)
    elif plot_type == 'image':
        ax = _plot_image(fig, 1, 3, cur_data, cur_target, cur_pred, vmin=vmin, vmax=vmax)
    elif plot_type == 'image_contour':
        ax = _plot_imagecontour(fig, 1, 1, cur_data, cur_target, cur_pred, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)

    if weak_label is not None: 
        contour_target = find_contours(weak_label.T.detach().cpu().numpy(), 0.5)
        for contour in contour_target:
            ax.plot(contour[:, 0], contour[:, 1], color='red', linestyle='dotted', lw=5)
            plt.axis('off')
    
    return plt


def _plot_imagecontour(fig, nrows, ncols, data, target, pred=None, vmin=0, vmax=None):
    """
    Function to generate 1 subplot: image overlapped with contour of target (blue) and pred (red)
    """
    # Computing the Active Contour for the given image
    contour_target = find_contours(target.T, 0.5)

    ## Image
    ax = fig.add_subplot(nrows, ncols, 1)
    ax.imshow(data, 'gray')
    plt.axis('off')
    
    # Contours of target and pred
    for contour in contour_target:
        ax.plot(contour[:, 0], contour[:, 1], '-r', lw=5)
        plt.axis('off')

    # We also plot prediction contour if it is available
    if pred is not None:
        ax.imshow(pred, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
        plt.axis('off')
    return ax


def _plot_contour(fig, nrows, ncols, data, target, pred=None):
    """
    Function to generate 1 subplot: image overlapped with contour of target (blue) and pred (red)
    """
    # Computing the Active Contour for the given image
    contour_target = find_contours(target.T, 0.5)

    ## Image
    ax = fig.add_subplot(nrows, ncols, 1)
    ax.imshow(data, 'gray')
    plt.axis('off')
    
    # Contours of target and pred
    for contour in contour_target:
        ax.plot(contour[:, 0], contour[:, 1], '-r', lw=5)
        plt.axis('off')

    # We also plot prediction contour if it is available
    if pred is not None:
        contour_pred = find_contours(pred.T, 0.5)
        for contour in contour_pred:
            ax.plot(contour[:, 0], contour[:, 1], '-b', lw=5)
            plt.axis('off')
    return ax


def _plot_image(fig, nrows, ncols, data, target, pred=None, vmin=0, vmax=None):
    """
    Function to generate 3 subplots: image, target and pred
    """
    # Image
    i = 1
    ax = fig.add_subplot(nrows, ncols, i)
    ax.imshow(data, 'gray')
    plt.axis('off')

    # Target
    i = 2
    ax = fig.add_subplot(nrows, ncols, i)
    ax.imshow(data, 'gray', interpolation='none')
    plt.axis('off')
    ax.imshow(target, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
    plt.axis('off')

    # Prediction
    i = 3
    ax = fig.add_subplot(nrows, ncols, i)
    ax.imshow(data, 'gray', interpolation='none')
    plt.axis('off')
    ax.imshow(pred, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
    plt.axis('off')

    return ax
