import glob, pathlib, os
from PIL import Image

input_dir = "data_ready/"
output_dir = "data/data_ready/"
image_size = 64

if not pathlib.Path(output_dir).exists():
    os.makedirs(output_dir)
    
img_list = glob.glob(input_dir + "*.JPG")
for i,im_name in enumerate(img_list):
    im_name = im_name.replace("\\", "/")
    im = Image.open(im_name)
    im = im.convert("RGB") if im.mode != "RGB" else im
    im = im.resize((image_size, image_size))
    im.save(output_dir + im_name.split("/")[-1][:-3] + "jpg", "JPEG")
    if i % 100 == 0:
        print( i, "/", len(img_list))