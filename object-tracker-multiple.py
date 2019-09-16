# Import the required modules
import dlib
import cv2
import argparse as ap
# import get_points
import os
import Get_points as gp
import obj_points as op

def run():

    # Create the tracker object
    folder = 'Images/Color'
    img=cv2.imread(folder+"/Color_20180109_092452_601.jpg")
    points=gp.get_points(img)
    # points=op.run(img)
    tracker = [dlib.correlation_tracker() for _ in range(len(points))]
    # Provide the tracker the initial position of the object
    [tracker[i].start_track(img, dlib.rectangle(*rect)) for i, rect in enumerate(points)]

    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        # Read frame from device or file
        img = cv2.imread(image_path)
        # retval, img = cam.read()
        # if not retval:
        #     print ("Cannot capture frame device | CODE TERMINATION :( ")
        #     exit()
        # Update the tracker  
        for i in range(len(tracker)):
            tracker[i].update(img)
            # Get the position of th object, draw a 
            # bounding box around it and display it.
            rect = tracker[i].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(img, pt1, pt2, (255, 0, 255), 2)
            print ("Object {} tracked at [{}, {}] \r".format(i, pt1, pt2))
            loc = (int(rect.left()+10), int(rect.top()+10))
            txt = "{}".format(i+1)
            cv2.putText(img, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 2)
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break

    # Relase the VideoCapture object
    # cam.release()

if __name__ == "__main__":
    # # Parse command line arguments
    # parser = ap.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('-d', "--deviceID", help="Device ID")
    # group.add_argument('-v', "--videoFile", help="Path to Video File")
    # parser.add_argument('-l', "--dispLoc", dest="dispLoc", action="store_true")
    # args = vars(parser.parse_args())
    #
    # # Get the source of video
    # if args["videoFile"]:
    #     source = args["videoFile"]
    # else:
    #     source = int(args["deviceID"])
    # run(source, args["dispLoc"])
    run()
