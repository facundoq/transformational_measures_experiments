from torch.utils.data import DataLoader

from variance_measure.pytorch_activations_iterator import ImageDataset

def get_data_generator(x,y,batch_size):

    image_dataset=ImageDataset(x,y)
    dataset=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

    image_rotated_dataset = ImageDataset(x, y, rotation=180)
    rotated_dataset = DataLoader(image_rotated_dataset , batch_size=batch_size, shuffle=True, num_workers=1)

    return dataset,rotated_dataset
