
import matplotlib.pyplot as plt
import numpy as np

def plot_image_grid(x,y,samples=64,show=True,save=None):


    initial_sample=0
    samples=min(samples,len(y))
    skip= y.shape[0] // samples

    grid_cols=8
    grid_rows=samples // grid_cols
    if samples % grid_cols >0:
        grid_rows+=1

    f,axes=plt.subplots(grid_rows,grid_cols,dpi=100)
    for axr in axes:
        for ax in axr:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    for i in range(samples):
        i_sample=i*skip+initial_sample
        klass = y[i_sample]
        row=i // grid_cols
        col=i % grid_cols
        ax=axes[row,col]
        if x.shape[3]==1:
            ax.imshow(x[i_sample,:,:,0], cmap='gray')
        else:
            sample=x[i_sample, :, :,:]
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




