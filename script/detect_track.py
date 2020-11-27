#Quick fix
import sys
sys.path.insert(0, '..')

from pathlib import Path

import collections
import time
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# Detection
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, \
    set_logging, increment_path,scale_coords
from utils.torch_utils import select_device, time_synchronized

# Tracking
from sort import Sort

import numpy as np

class DebuggingTools:
    def __init__(self, file,img_size=640):
        self.img_size=img_size

        self.vid_cap = cv2.VideoCapture(file)  # video capture object

        # source, view_img, save_txt, imgsz = opt.source, opt.view_img, opt.save_txt, opt.img_size

        vid_path, vid_writer = None, None

        save_path = 'test2.mp4'

        if vid_path != save_path:  # new video
            vid_path = save_path
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer

            fourcc = 'mp4v'  # output video codec
            fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        
        _,first_img = self.vid_cap.read()
        # check for common shapes
        s = np.stack([self.letterbox(first_img, new_shape=self.img_size)[0].shape], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def save_frame_into_video(self,im0):
        self.vid_writer.write(im0)
    
    def get_frame_read(self):
        ret, frame = self.vid_cap.read()

        # Padded resize
        if ret:
            img = self.letterbox(frame, new_shape=self.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            return ret,img,frame

        return ret,[],frame
    
    def release_video_writer(self):
        self.vid_writer.release()
    
    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
        


class Detection:
    def __init__(self):
        print("Initialize Detection Model")
        weights, imgsz = opt.weights, opt.img_size

        # Initialize
        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if self.half:
            model.half()  # to FP16

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

        # TODO: restructure this
        self.model = model
        self.augment = opt.augment
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms

    def detect(self,img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        # t2 = time_synchronized()

        # Print time (inference + NMS)
        # print('%sDone. (%.3fs)' % (s, t2 - t1))
  
        return pred 

class Tracker:
    def __init__(self):
        print("Initialize Tracker")
        self.tracker_ = Sort()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='../runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.weights = '../runs/train/exp8_trolley_bed/weights/best.pt'
    opt.conf_thres = 0.5

    print(opt)
    
    # Alex trajectory visualization start from here
    track_trajectory = dict()
    collection_length = 20
    moving_direction_threshold = 5
    min_trajectory_length = 15
    # Alex trajectory visualization start from here
    
    total_time = 0.0

    tracker = Tracker()
    dataset = DebuggingTools('../dataset/trolley_bed/5.mp4')
    detector = Detection()
    
    loading_video = True
    while(loading_video):
        try:
            status , input_image , im0 = dataset.get_frame_read()

            if not status:
                print("Error from get_frame_read")
                loading_video = False
                dataset.release_video_writer()
                sys.exit(0)
                continue
            
        except Exception as e:
            print(e)
            print("maybe no more video liao")
            loading_video = False
            dataset.release_video_writer()
            sys.exit(0)

        start_time = time.time()

        # Detection
        pred = detector.detect(input_image)
        pred_ = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords([384,640], det[:, :4], im0.shape).round()
                pred_.append(det)
        
        if pred_ !=[]:
            trackers = tracker.tracker_.update(pred_[0].cpu().numpy())
            cycle_time = time.time() - start_time
            total_time += cycle_time

            current_track_ids = []

            input_image = np.moveaxis(input_image,0,-1)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

            # print(f"Tracker {trackers.shape}")
            for d in trackers:
                if d[4] in track_trajectory:
                    track_trajectory[d[4]].append([int((d[2]+d[0])/2),int((d[3]+d[1])/2)])
                    size = len(track_trajectory[d[4]])
                    current_track_ids.append(d[4])

                else:
                    trajectory_point_history = collections.deque(maxlen=collection_length)
                    trajectory_point_history.append([int((d[2]+d[0])/2),int((d[3]+d[1])/2)])
                    track_trajectory[d[4]]= trajectory_point_history
                    current_track_ids.append(d[4])
                d = d.astype(np.int32)
                cv2.rectangle(im0, (d[0],d[1]),(d[2],d[3]), color=(255, 0, 0) , thickness=2) 

            for key in track_trajectory:
                if key not in current_track_ids:
                    continue

                # Determine direction
                direction = 0
                moving_direction = ''
                if len(track_trajectory[key])>min_trajectory_length:
                    for index , i in enumerate(track_trajectory[key]):
                        if index==0:
                            last_x = i[0]
                            continue
                        direction_ = 1 if i[0] - last_x > 0 else -1
                        direction+=direction_
                        if direction >moving_direction_threshold:
                            moving_direction = 'right'
                        elif direction < -moving_direction_threshold:
                            moving_direction = 'left'

                        last_x = i[0]
            
            end_time = time.time()

            for key in track_trajectory:
                if key not in current_track_ids:
                    continue
                
                for i in track_trajectory[key]:
                    cv2.circle(im0, (i[0], i[1]), radius=5, color=(255, 0, 0), thickness=-1)

                centre_x = i[0]-200 if i[0]-200>0 else 10
                centre_y = i[1]-100 if i[1]-100>0 else 10

                cv2.putText(im0, str(key)+"_"+str(moving_direction), (centre_x, centre_y), cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA) 

            # Draw Confidence
            for i in pred_[0].cpu().numpy():
                centre_x = i[0]-200 if i[0]-200>0 else 10
                centre_y = i[1]-100 if i[1]-100>0 else 10
                cv2.putText(im0, str(int(i[4]*100))+"_confidence", (int(centre_x), int(centre_y)), cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA) 

            print(f"FPS: {1/(end_time-start_time)}")

        dataset.save_frame_into_video(im0)



        
        

