# Import the required modules
from scipy.ndimage import gaussian_filter
import cv2
import os
import numpy as np
import sys
sys.setrecursionlimit(5000)
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
def detect_pecking(obj_pos):
    output=[]
    for i in range(len(obj_pos)):
        p1=obj_pos[i]
        for j in range(len(obj_pos)):
            if(i==j):
                continue
            p2=obj_pos[j]
            xm=np.min([p1[0],p2[0]])
            xn = np.max([p1[1], p2[1]])
            ym = np.min([p1[2], p2[2]])
            yn = np.max([p1[3], p2[3]])
            if(xn-xm<p1[1]-p1[0]+p2[1]-p2[0] and yn-ym<p1[3]-p1[2]+p2[3]-p2[2]):
                output.append((i,j))
    return output

def neighbour_obj(obj_pos,pos):
    xp=int((pos[0]+pos[1])/2)
    yp = int((pos[2] + pos[3]) / 2)
    M=100000
    index=-1
    for i in range(len(obj_pos)):
        pos1=obj_pos[i]
        xp1 = int((pos1[0] + pos1[1]) / 2)
        yp1 = int((pos1[2] + pos1[3]) / 2)
        d=np.sqrt((xp-xp1)**2+(yp-yp1)**2)
        if(d<M):
            M=d
            index=i
    return index,M
def empyt_posnum(pos_num):
    for i in range(22):
        flag=0
        for j in range(len(pos_num)):
            if(pos_num[j]==i):
                flag=1
                break
        if(flag==1):
            continue
        else:
            return i
def run(text):

    # Create the tracker object
    folder = text
    num=0
    obj_pos=[]
    obj_num=[]
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        # Read frame from device or file
        img = cv2.imread(image_path)
        cobj_pos=[]
        cobj_num=[]
        points = process(img)
        for i in range(len(points)):
            pos=points[i]
            if(num==0):
                obj_pos.append(pos)
                obj_num.append(i)
            else:
                index, M=neighbour_obj(obj_pos,pos)
                if(M<100):
                    cobj_pos.append(pos)
                    cobj_num.append(obj_num[index])
                else:
                    cobj_pos.append(pos)
                    cobj_num.append(-1)


        if(num>0):
            for i in range(len(cobj_num)):
                if (cobj_num[i] == -1):
                    index = empyt_posnum(cobj_num)
                    cobj_num[i] = index
            obj_pos=cobj_pos
            obj_num=cobj_num
        for i in range(len(obj_pos)):
            pos = obj_pos[i]
            xp = int((pos[0] + pos[1]) / 2)
            yp = int((pos[2] + pos[3]) / 2)
            loc = (xp, yp)
            txt = "{}".format(obj_num[i] + 1)
            cv2.putText(img, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
            pt1 = (pos[0], pos[2])
            pt2 = (pos[1], pos[3])
            cv2.rectangle(img, pt1, pt2, (255, 0, 255), 2)
            print("Object {} tracked at [{}, {}] \r".format(obj_num[i], pt1, pt2))
        output=detect_pecking(obj_pos)
        if(len(output)>0):
            for i in range(len(output)):
                txt='pecking detection: object '+str(obj_num[output[i][0]])+'and object '+str(obj_num[output[i][1]])
                cv2.putText(img, txt, (100,100+i*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            txt = 'No pecking'
            cv2.putText(img, txt, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        num += 1
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    text = input("file folder:")
    run(text)