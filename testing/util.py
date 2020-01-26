
import matplotlib.pyplot as plt
import numpy as np

def plot_image_grid(x,samples=64,grid_cols=8,show=True,save=None,normalize=False):
    n=x.shape[0]
    initial_sample=0
    samples=min(samples,n)
    skip= n // samples


    grid_rows=samples // grid_cols
    if samples % grid_cols >0:
        grid_rows+=1

    f,axes=plt.subplots(grid_rows,grid_cols,dpi=200,squeeze=False,figsize=(grid_cols,grid_rows))
    for axr in axes:
        for ax in axr:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    for i in range(samples):
        i_sample=i*skip+initial_sample
        row=i // grid_cols
        col=i % grid_cols
        ax = axes[row, col]

        if x.shape[3]==1:
            ax.imshow(x[i_sample,:,:,0], cmap='gray')
        else:
            sample=x[i_sample, :, :,:]
            if normalize:
                mn,mx=sample.min(axis=(0,1)),sample.max(axis=(0,1))
                sample = (sample - mn)
                sample /= (mx-mn)
            ax.imshow(sample)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if show:
        plt.show()
    if not save is None:
        plt.savefig(save)
    plt.close(f)




