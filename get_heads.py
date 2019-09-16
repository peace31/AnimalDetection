import cv2
import numpy as np
from scipy.ndimage import label,gaussian_filter
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import os

def find_contour(img,x,y,r):
    height, width = img.shape[:2]
    x=int(x)
    y=int(y)
    r=int(r)
    s=3.14*r**2
    num=0
    for i in range(max([x-r,0]),min([x+r,width])):
        for j in range(max(y-r,0),min(y+r,height)):
            if((i-x)**2+(j-y)**2>r**2):
                continue
            if(img[j][i]>0):
                num+=1
    if(num/s<0.5):
        return False
    return True
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)
def process(img):
    from scipy.ndimage import label
    points=[]
    shifted = cv2.pyrMeanShiftFiltering(img, 10, 21)
    img=shifted
    gamma = 0.4  # change the value here to get different result
    shifted = adjust_gamma(img, gamma=gamma)
    b, g, r = cv2.split(shifted)
    for i in range(len(b)):
        for j in range(len(b[0])):
            bb=int(b[i][j])
            gg=int(g[i][j])
            rr = int(r[i][j])
            # if (i == 291 and j == 292):
            #     print(bb)
            #     print(gg)
            #     print(rr)
            if(rr>gg and gg>bb and bb>40 and bb<110 and rr>70 and rr<150 and np.abs(bb-gg)<25):
                continue
            else:
                b[i][j]=0
                g[i][j]=0
                r[i][j]=0

        # get b,g,r
    shifted = cv2.merge([r, g, b])  # switch it to rgb
    # cv2.imshow('shifted', shifted)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    #
    gray = gaussian_filter(gray, sigma=1)
    # cv2.imshow('gray',gray)
    # cv2.waitKey(0)
    # canny operation for binary image
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow('bw',thresh)
    # cv2.waitKey(0)
    kernel = np.ones((3, 3), np.uint8)
    # image dilation for finding each player's card group
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    # cv2.imshow('bw', thresh)
    # cv2.waitKey(0)
    lab_array, num_features = label(thresh)
    unique, counts = np.unique(lab_array, return_counts=True)
    for i in range(len(counts)):
        if (counts[i] < 300):
            continue
        th3 = thresh.copy()
        img1 = img.copy()
        th3[lab_array != unique[i]] = 0
        img1[lab_array != unique[i]] = 0
        ind = np.where(lab_array == unique[i])
        xm = np.min(ind[1])
        ym = np.min(ind[0])
        xn = np.max(ind[1])
        yn = np.max(ind[0])
        th31 = th3[ym:yn, xm:xn]
        if (np.sum(th31) == 0): continue
        if (xm < 100 or xm > 850):
            continue
        points.append((xm, xn, ym, yn))
        # cv2.rectangle(img, (xm, ym), (xn, yn), (0, 255, 0), 3)
    return points
# folder = 'Images/Color'
# for filename in os.listdir(folder):
#     im_path = os.path.join(folder, filename)
#     # im_path='Images/Color/Color_20180109_092453_699.jpg'
#     img=cv2.imread(im_path)
#     img=process(img)
#     cv2.imshow('output', img)
#     cv2.waitKey(0)
