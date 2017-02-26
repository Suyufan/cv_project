from PIL import Image
import random

img = Image.open("clear_6.jpg")

y, x = img.size
print(x)
print(y)


for i in range(1, 20):
    y1 = random.randint(0, y-64-1)
    x1 = random.randint(0, x-64-1)

    box = (x1, y1, x1+64, y1+64)
    region = img.crop(box)
    s = "easy_neg/" + str(i) + ".jpg"
    region.save(s)