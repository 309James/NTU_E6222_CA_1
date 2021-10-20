import numpy as np
from PIL import Image

# Look up Table
def threshold_LUT(t):
    LUT = []
    for i in range(t):
        LUT.append(0)
    for i in range(t,256):
        LUT.append(255)
    return LUT


if __name__ == '__main__':
    # load the image file
    img_name = 'img1'
    img = Image.open("Images/"+img_name+".bmp")
    # transform image into grey level matrix
    img_a = np.array(img)
    h = img.height
    w = img.width
    v = 180
    LUT_1 = threshold_LUT(v)
    # threshold process
    for i in range(h):
        for j in range(w):
            img_a[i, j] = LUT_1[img_a[i, j]]

    img2 = Image.fromarray(np.uint8(img_a))
    img2.show()
    # output
    img2.save(img_name+'_'+str(v)+'.png', "png")









