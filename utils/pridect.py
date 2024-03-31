# 模型预测相关
import cv2
import time
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams
from ultralytics.data.augment import LetterBox
from queue import Queue


class Pridect(LoadStreams):
    def __init__(
            self,
            weight,
            imgsz=640,
            sources="file.streams",
            group_scale: int=4,  # 每组4路视频
            show_w: int=1920,
            show_h: int=1080,
            vid_stride: int=1,
            buffer=False
            ):
        super().__init__(sources, vid_stride, buffer)

        self.weight = weight
        self.sources = sources
        self.imgsz = imgsz
        self.group_scale = group_scale
        self.show_w = show_w
        self.show_h = show_h
        self.vid_stride = vid_stride

        self.group_index = 0  # 当前显示的组索引

        self.scale = int(np.ceil(np.sqrt(group_scale)))  # 横纵方向的视频数量
        self.grid_w = int(self.show_w / self.scale)  # 每个视频方格的宽度
        self.grid_h = int(self.show_h / self.scale)  # 每个视频方格的高度

        self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)

        # self.run = True  # 运行标志  使用LoadStreams的running属性替代
        self.first_run = True  # 第一次运行标志

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # keep a <=30-image buffer
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        print('摄像头无响应')
                        cap.open(stream)  # re-open stream if signal was lost
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer is empty
    
    def start(self):
        print(self.shape)


if __name__ == '__main__':
    weight = r'E:\Projects\weight\yolo\v8\detect\coco\yolov8s.pt'
    sources = 'list.streams'
    imgsz = 640
    predicter = Pridect(weight, imgsz, sources, vid_stride=5, buffer=True)
    # predicter.start()
    # print(predicter.shape)