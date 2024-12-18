import torchvision

def brightness_transform(brightness_factor:float):
    #return lambda x: torchvision.transforms.functional.adjust_brightness(x,brightness_factor)
    return lambda x: x*brightness_factor

def rotation_transform(angle:float):
    #return lambda x: torchvision.transforms.functional.adjust_brightness(x,brightness_factor)
    
    return lambda x: torchvision.transforms.functional.rotate(x,angle=angle)


transformations = {
    "brightness":[brightness_transform(f) for f in [0.25,0.5,0.75,1,1.25,1.50,1.75,2.0]],
    "rotation":[rotation_transform(f) for f in [0,45,90,135,180,225,270,315]],
}