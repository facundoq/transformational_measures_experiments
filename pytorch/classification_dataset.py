from torch.utils.data import DataLoader

from variance_measure.pytorch_activations_iterator import ImageDataset
from pytorch.numpy_dataset import NumpyDataset
def get_data_generator(x,y,batch_size):

    dataset=NumpyDataset(x,y)

    image_dataset=ImageDataset(dataset)
    dataloader=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

    image_rotated_dataset = ImageDataset(dataset, rotation=180)
    rotated_dataloader= DataLoader(image_rotated_dataset , batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader,rotated_dataloader
