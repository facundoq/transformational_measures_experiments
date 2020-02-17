from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys

from pathlib import Path

# fontname='Pillow/Tests/fonts/FreeMono.ttf'
fontname='/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-regular.otf'


def generate_text_image(text,color,fontsize):
    font = ImageFont.truetype(fontname, fontsize)
    width, height = font.getsize(text)
    text_image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_image)
    draw.text((0, 0), text, color, font=font)
    return text_image

def draw_text_rotated(image,text,position,angle,color=(0,0,0),fontsize=13):
    x, y = position
    text_image = generate_text_image(text,color,fontsize)
    text_image = text_image.rotate(angle,expand=True)
    #h,w=text_image.height,text_image.width
    image.paste(text_image, (x,y))
    return image

def draw_text(draw,text,position,fontsize=13):
    x,y=position
    font = ImageFont.truetype(fontname, fontsize)
    w,h=font.getsize(text)

    draw.rectangle((x, y, x + w, y + h), fill='white')
    draw.text(position, text, (0, 0, 0), font=font)

def modify_accuracy(image):
    draw = ImageDraw.Draw(img)
    label_size = 16
    for x in [109,298,540,731]:
        draw_text(draw,"Época",(x,245),fontsize=label_size)

    legend_size=11
    for x in [163,592]:
        draw_text(draw,"entren.",(x,193),fontsize=legend_size)
        draw_text(draw, "prueba", (x, 207),fontsize=legend_size)

    for x in [353,784]:
        draw_text(draw,"entren.",(x,20),fontsize=legend_size)
        draw_text(draw, "prueba", (x, 34),fontsize=legend_size)

    for x in [17,448]:
        image = draw_text_rotated(image, "Tasa de aciertos",(x,70),90,fontsize=label_size)
    for x in [203,633]:
        image = draw_text_rotated(image, "Error",(x,100),90,fontsize=label_size)
    return image

def modify_training(image):
    draw = ImageDraw.Draw(img)
    label_size = 17
    labels=["Entrenamiento normal","Entrenamiento rotado"]
    for x,l in zip([63, 265],labels):
        draw_text(draw, l, (x, 241), fontsize=label_size)

    image = draw_text_rotated(image, "Tasa de aciertos", (19, 60), 90, fontsize=label_size+5)

    legend_size=12
    for y,l in zip([11,26],["Prueba normal","Prueba rotado"]):
        draw_text(draw,l,(219,y),fontsize=legend_size)


    return image
def modify_retraining(image):
    draw = ImageDraw.Draw(img)
    label_size = 65
    draw_text(draw, "Capas reentrenadas", (1200, 1065), fontsize=label_size)

    image = draw_text_rotated(image, "Tasa de aciertos", (6,300), 90, fontsize=label_size)

    legend_size = 43
    for y, l in zip([32,82], ["Prueba normal", "Prueba rotado"]):
        draw_text(draw, l, (1442, y), fontsize=legend_size)

    return image

folderpath=Path("~/Dropbox/tesis/img/da").expanduser()
output_folderpath=folderpath / "es"

output_folderpath.mkdir(exist_ok=True)

image_files = [f for f in folderpath.iterdir() if f.suffix==".jpg"]

accuracy_image_files = [f for f in image_files if f.stem.endswith("accuracy")]

training_image_files = [f for f in image_files if f.stem.endswith("training")]
retraining_image_files = [f for f in image_files if f.stem.startswith("retrain")]

sets = [accuracy_image_files,training_image_files,retraining_image_files]
functions = [modify_accuracy,modify_training,modify_retraining]


for s,f in zip(sets,functions):
    for filepath in s:
        img = Image.open(filepath)
        img = f(img)
        output_filepath = output_folderpath / filepath.name
        img.save(output_filepath)


