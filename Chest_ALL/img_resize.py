from PIL import Image, ImageDraw, ImageFont
import os


im = Image.open('./trainAndTest/test/1.2.156.112536.2.560.7050106199148.1364040662164.2786.jpg').convert("RGB")
w = im.width
h = im.height
im = im.resize((512, 512), Image.ANTIALIAS)
im.w
im = im.resize((1024, 1024), Image.ANTIALIAS)
