import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def threshold_LUT(t):
    LUT = []
    for i in range(t):
        LUT.append(0)
    for i in range(t, 256):
        LUT.append(255)
    return LUT


if __name__ == '__main__':
    img_name = 'img6'
    img = Image.open("Images/"+img_name+".bmp")
    img_a = np.array(img)
    h = img.height
    w = img.width
    v = 1
    LUT_1 = threshold_LUT(v)
    for i in range(h):
        for j in range(w):
            img_a[i, j] = LUT_1[img_a[i, j]]
            if img_a[i, j] == 0:
                img_a[i, j] = 1
            else:
                img_a[i, j] = 0
    
    indicator_1 = 0
    indicator_2 = 0
    height = h
    width = w

    while indicator_1 == 0 or indicator_2 == 0:
        # step 1
        indicator_1 = 1
        remove_r1 = []
        remove_c1 = []
        for i in range(1, height-1):
            for j in range(1, width-1):
                if img_a[i, j] == 1:
                    p1 = img_a[i, j]
                    neighbors = [img_a[i-1, j], img_a[i-1, j+1], img_a[i, j+1], img_a[i+1, j+1],
                                 img_a[i+1, j], img_a[i+1, j-1], img_a[i, j-1], img_a[i-1, j-1],
                                 img_a[i-1, j]]
                    a = 0
                    b = 0
                    satisfy = 0
                    for k in range(8):
                        if neighbors[k] == 1:
                            b = b+1
                    if 2 <= b <= 6:
                        satisfy = satisfy+1

                    for k in range(8):
                        if neighbors[k] == 0 and neighbors[k+1] == 1:
                            a = a+1
                    if a == 1:
                        satisfy = satisfy+1

                    if neighbors[0] == 0 or neighbors[2] == 0 or neighbors[4] == 0:
                        satisfy = satisfy+1
                    if neighbors[2] == 0 or neighbors[4] == 0 or neighbors[6] == 0:
                        satisfy = satisfy+1

                    if satisfy == 4:
                        remove_r1.append(i)
                        remove_c1.append(j)
                        indicator_1 = 0
        if indicator_1 == 0:
            for i in range(len(remove_c1)):
                img_a[remove_r1[i], remove_c1[i]] = 0

        # step 2
        indicator_2 = 1
        remove_r2 = []
        remove_c2 = []
        for i in range(1, height-1):
            for j in range(1, width-1):
                if img_a[i, j] == 1:
                    p1 = img_a[i, j]
                    neighbors = [img_a[i-1, j], img_a[i-1, j+1], img_a[i, j+1], img_a[i+1, j+1],
                                 img_a[i+1, j], img_a[i+1, j-1], img_a[i, j-1], img_a[i-1, j-1],
                                 img_a[i-1, j]]
                    a = 0
                    b = 0
                    satisfy = 0
                    for k in range(8):
                        if neighbors[k] == 1:
                            b = b+1
                    if 2 <= b <= 6:
                        satisfy = satisfy+1
                    for k in range(8):
                        if neighbors[k] == 0 and neighbors[k+1] == 1:
                            a = a+1
                    if a == 1:
                        satisfy = satisfy+1
                    if neighbors[0] == 0 or neighbors[2] == 0 or neighbors[6] == 0:
                        satisfy = satisfy+1
                    if neighbors[0] == 0 or neighbors[4] == 0 or neighbors[6] == 0:
                        satisfy = satisfy+1

                    if satisfy == 4:
                        remove_r2.append(i)
                        remove_c2.append(j)
                        indicator_2 = 0
        if indicator_2 == 0:
            for i in range(len(remove_c2)):
                img_a[remove_r2[i], remove_c2[i]] = 0

    for i in range(h):
        for j in range(w):
            if img_a[i, j] == 0:
                img_a[i, j] = 255
            else:
                img_a[i, j] = 0
    img2 = Image.fromarray(np.uint8(img_a))
    img2.show()
    img2.save(img_name + '_thinning' + '.png', "png")
