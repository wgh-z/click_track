#!/usr/bin/env python3

import cv2
import os
import yaml
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from utils.toolbox import Interpolator, Timer
from utils.regional_judgment import point_in_rect
from moviepy.editor import VideoFileClip


show_all = False
hide_all = True
pause = False
first_puase = True
prior_frame = None
l_point, r_point = None, None

def on_mouse(event, x, y, flags, param):
    global img, l_point, r_point
    # img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        l_point = (x,y)
        # cv2.circle(img2, point1, 10, (0,255,0), 5)
        # cv2.imshow('image', img2)
        print(l_point)
    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
        r_point = (x,y)
        # cv2.circle(img2, point2, 10, (255,0,0), 5)
        # cv2.imshow('image', img2)
        # print(point2)

def main(video_name='测试.mp4'):
    global img, l_point, r_point, show_all, hide_all, pause, first_puase, prior_frame
    show_id = {}

    timer = Timer(30)
    
    with open('assets/coco_chinese.yaml', 'r', encoding='utf-8') as f:
        names = yaml.load(f, Loader=yaml.FullLoader)
    
    model = YOLO(r"weights/yolov8m.pt")
    
    video_path = f"videos/{video_name}"
    video_name_without_suffix = os.path.splitext(video_name)[0]
    temp_path = f"results/{video_name_without_suffix}_temp.mp4"
    save_path = f"results/{video_name_without_suffix}_out.mp4"
    
    ori_video = VideoFileClip(video_path)
    audio = ori_video.audio
    
    img = np.zeros((600, 600, 4), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    while True:
        if not pause:
            ret, frame = cap.read()
            prior_frame = frame
        else:
            frame = prior_frame
            ret = True

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[0, 2, 5, 7],  # 0: person, 2: car
            # tracker="botsort.yaml",  # 12fps
            tracker="bytetrack.yaml",  # 20fps
            # imgsz=(384, 640),
            half=True,
            verbose=False,
            )

        # maintain show_id
        try:
            id_set = set(results[0].boxes.id.int().cpu().tolist())
        except AttributeError:
            id_set = set()
        show_id = timer(id_set, show_id)

        annotated_frame = frame.copy()
        annotator = Annotator(annotated_frame, line_width=2, font_size=20, pil=True, example=str(names[0]))
        det = results[0].boxes.data.cpu().numpy()
        if len(det) and len(det[0]) == 7:  # 有目标，且有id元素
            for *xyxy, id, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    id = int(id)  # integer id

                    if l_point is not None and id not in show_id:
                        if point_in_rect(l_point, xyxy):
                            # show_id.append(id)
                            show_id = timer.add_delay(show_id, id)
                            l_point = None
                            show_all = False
                            hide_all = False

                    if r_point is not None:
                        if point_in_rect(r_point, xyxy):
                            try:
                                # show_id.remove(id)
                                del show_id[id]
                            except:
                                pass
                            r_point = None
                            show_all = False
                            hide_all = False

                    if show_all: # 显示所有目标
                        # label = f"{id} {results[0].names[c]} {conf:.2f}"
                        label = f"{names[c]}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    elif hide_all:
                            pass
                    elif id in show_id.keys():  # 显示指定id的目标
                        label = "可疑目标事件对象"
                        annotator.box_label(xyxy, label, color=colors(0, True))

        annotated_frame = annotator.result()

        cv2.imshow('image', annotated_frame)
        if not pause:
            video_writer.write(annotated_frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 97:  # a键显示所有目标
            show_all = True
            hide_all = False
            show_id = {}
        elif key == 115:  # s键清除所有目标
            show_all = False
            hide_all = True
            show_id = {}
        elif key == 32:  # 空格键暂停
            # cv2.waitKey(0)
            pause = not pause
            if first_puase:
                first_puase = False
                show_all = True
                hide_all = False
                show_id = {}

    cap.release()
    video_writer.release()

    new_video = VideoFileClip(temp_path)
    new_video = new_video.set_audio(audio)
    new_video.write_videofile(save_path)
    
    os.remove(temp_path)


if __name__ == '__main__':
    # 读取videos文件夹下的所有视频文件
    video_list = os.listdir('videos')
    for video_name in video_list:
        if video_name.endswith('.mp4'):
            print(f"开始处理 {video_name}")
            main(video_name)
