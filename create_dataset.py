from PIL import Image, ImageFont, ImageDraw ,ImageOps
from pathlib import Path
import time
import numpy as np
      


def get_all_font_names():

    return [
'C:\\Windows\\fonts\\arialbd.ttf',
'C:\\Windows\\fonts\\arial.ttf',
'C:\\Windows\\fonts\\arialbi.ttf',
'C:\\Windows\\fonts\\ariali.ttf',
'C:\\Windows\\fonts\\calibri.ttf',
'C:\\Windows\\fonts\\calibrib.ttf',
'C:\\Windows\\fonts\\calibrii.ttf',
'C:\\Windows\\fonts\\calibril.ttf',
'C:\\Windows\\fonts\\calibrili.ttf',
'C:\\Windows\\fonts\\calibriz.ttf',
'C:\\Windows\\fonts\\Candara.ttf',
'C:\\Windows\\fonts\\Candarab.ttf',
'C:\\Windows\\fonts\\Candarai.ttf',
'C:\\Windows\\fonts\\Candaral.ttf',
'C:\\Windows\\fonts\\Candarali.ttf',
'C:\\Windows\\fonts\\Candaraz.ttf',
'C:\\Windows\\fonts\\comic.ttf',
'C:\\Windows\\fonts\\comicbd.ttf',
'C:\\Windows\\fonts\\comici.ttf',
'C:\\Windows\\fonts\\comicz.ttf',
'C:\\Windows\\fonts\\consola.ttf',
'C:\\Windows\\fonts\\consolab.ttf',
'C:\\Windows\\fonts\\consolai.ttf',
'C:\\Windows\\fonts\\consolaz.ttf',
'C:\\Windows\\fonts\\georgia.ttf',
'C:\\Windows\\fonts\\georgiab.ttf',
'C:\\Windows\\fonts\\georgiai.ttf',
'C:\\Windows\\fonts\\georgiaz.ttf',
'C:\\Windows\\fonts\\GOTHIC.TTF',
'C:\\Windows\\fonts\\verdana.ttf',
'C:\\Windows\\fonts\\verdanab.ttf',
'C:\\Windows\\fonts\\verdanai.ttf',
'C:\\Windows\\fonts\\verdanaz.ttf',
"C:\\Windows\\fonts\\tahoma.ttf",
"C:\\Windows\\fonts\\impact.ttf",
'C:\\Windows\\fonts\\times.ttf',
'C:\\Windows\\fonts\\timesbd.ttf',
'C:\\Windows\\fonts\\timesbi.ttf',
'C:\\Windows\\fonts\\timesi.ttf',
'C:\\Windows\\fonts\\trebuc.ttf',
'C:\\Windows\\fonts\\swiss.ttf',
'C:\\Windows\\fonts\\swissb.ttf',
'C:\\Windows\\fonts\\swissbi.ttf',
'C:\\Windows\\fonts\\swissbo.ttf',
'C:\\Windows\\fonts\\swissc.ttf',
]

if __name__ == "__main__":
    data={"x_train":[],"y_train":[]}
    for f in get_all_font_names():
        for num in range(1,10):
            for size in [25,24,23,22,26]:
                for x_offset in [0,1,-1]:
                    for y_offset in [0,1,-1]:
                        image = Image.new("L", (28, 28), color=255)
                        draw = ImageDraw.Draw(image)
                        font = ImageFont.truetype(f, size)  
                        text = str(num)
                        draw.text((14+x_offset,14+y_offset),text,font=font,anchor="mm")
                        image = ImageOps.invert(image)
                        #image.save(f'C:\\misc\\python_projects\\simple_neural_network_mnist\\images\\{Path(f).name}{time.time()}.png')
                        data["x_train"].append(np.array(image))
                        data['y_train'].append(num)

    np.savez("new_data.npz",x_train=data['x_train'],y_train=data['y_train'])
                        




        
