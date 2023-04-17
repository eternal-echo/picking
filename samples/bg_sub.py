from __future__ import print_function
import cv2 as cv
import argparse
from detect.parts_segment import BackgroundModel

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--history', type=int, help='Length of the history.', default=500)
parser.add_argument('--var-threshold', type=float, help='Threshold on the squared Mahalanobis distance between \
                    the pixel and the model to decide whether a pixel is well described by the background model. \
                    This parameter does not affect the background update.', default=16)
parser.add_argument('--no-shadows', dest='shadows', action='store_false', help='Highlight shadows in the foreground mask.')
parser.set_defaults(shadows=True)
parser.add_argument('--save', type=str, help='Path to save the output video.', default=None)
args = parser.parse_args()

## [create]
# create Background Subtractor objects
print('Using ' + args.algo + ' algorithm')
print('History: ' + str(args.history))
print('VarThreshold: ' + str(args.var_threshold))
print('DetectShadows: ' + str(args.shadows))
back_model = BackgroundModel(algo=args.algo, history=args.history, varThreshold=args.var_threshold, detectShadows=args.shadows)
## [create]

## [capture]
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

# 如果指定了保存路径，则创建VideoWriter对象
if args.save is not None:
    fps = int(capture.get(cv.CAP_PROP_FPS))
    frame_size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
    out = cv.VideoWriter(args.save, cv.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    ## [apply]
    #update the background model
    fgMask = back_model.apply(frame)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ## [show]
    #show the current frame and the fg masks
    cv.namedWindow('Frame', cv.WINDOW_NORMAL)
    cv.imshow('Frame', frame)
    cv.namedWindow('FG Mask', cv.WINDOW_NORMAL)
    cv.imshow('FG Mask', fgMask)
    ## [show]

    # 如果指定了保存路径，则将当前帧写入视频文件中
    if args.save is not None:
        out.write(fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
