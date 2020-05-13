
import numpy as np
import cv2
import datasets

def expand_channels(dataset:datasets.ClassificationDataset,c:int):
    if dataset.dataformat=="NHWC":
        axis=3
    else:
        axis=1

    dataset.x_train = np.repeat(dataset.x_train,c,axis=axis)
    dataset.x_test = np.repeat(dataset.x_test, c, axis=axis)


def collapse_channels(dataset:datasets.ClassificationDataset):
    if dataset.dataformat=="NHWC":
        axis=3
    else:
        axis=1
    dataset.x_train = dataset.x_train.mean(axis=axis,keepdims=True)
    dataset.x_test  = dataset.x_test.mean(axis=axis,keepdims=True)


def resize(dataset:datasets.ClassificationDataset,h:int,w:int,c:int):

    if dataset.dataformat=="NCHW":
        dataset.x_train=np.transpose(dataset.x_train,axes=(0,2,3,1))
        dataset.x_test = np.transpose(dataset.x_test, axes=(0, 2, 3, 1))

    subsets = [dataset.x_train, dataset.x_test]
    new_subsets=[np.zeros((s.shape[0],h,w,c)) for s in subsets]

    for (subset,new_subset) in zip(subsets,new_subsets):
        for i in range(subset.shape[0]):
            img=subset[i, :]
            if c==1:
                #remove channel axis, resize, put again
                img=img[:,:,0]
                img= cv2.resize(img, dsize=(h, w))
                img = img[:, :, np.newaxis]
            else:
                #resize
                img = cv2.resize(img, dsize=(h, w))

            new_subset[i,:]=img

    dataset.x_train = new_subsets[0]
    dataset.x_test = new_subsets[1]

    if dataset.dataformat=="NCHW":
        dataset.x_train = np.transpose(dataset.x_train,axes=(0,3,1,2))
        dataset.x_test = np.transpose(dataset.x_test, axes=(0, 3, 1, 2))

def adapt_dataset(dataset:datasets.ClassificationDataset, dataset_template:str):
    dataset_template = datasets.get(dataset_template)
    h,w,c= dataset_template.input_shape
    del dataset_template
    oh,ow,oc=dataset.input_shape

    # fix channels
    if c !=oc and oc==1:
        expand_channels(dataset,c)

    elif c != oc and c ==1:
        collapse_channels(dataset)
    else:
        raise ValueError(f"Cannot transform image with {oc} channels into image with {c} channels.")

    #fix size
    if h!=oh or w!=ow:
        resize(dataset,h,w,c)

    dataset.input_shape=(h,w,c)
