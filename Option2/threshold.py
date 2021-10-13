import numpy as np
from PIL import Image

def threshold_LUT(t):
    LUT = []
    for i in range(t):
        LUT.append(0)
    for i in range(t,256):
        LUT.append(255)
    return LUT


if __name__ == '__main__':
    img = Image.open("Images/img6.bmp")
    img_a = np.array(img)
    h = img.height
    w = img.width
    LUT_1 = threshold_LUT(140)
    for i in range(h):
        for j in range(w):
            img_a[i, j] = LUT_1[img_a[i, j]]

    img2 = Image.fromarray(np.uint8(img_a))
    img2.show()
    # img2.save('2.png',"png")









