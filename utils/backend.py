# 自定义模型预测相关
import cv2
import time
import threading
import numpy as np
import yaml
from pathlib import Path
from queue import Queue

# from ultralytics import YOLO
# from ultralytics.data.loaders import LoadStreams
# from ultralytics.data.augment import LetterBox

from utils.toolbox import SquareSplice
from utils.draw import create_void_img, DelayDraw
from utils.video_io import VideoDisplayManage, ReadVideo
from models.track import Track


class SmartBackend:
    def __init__(self, config_path='./cfg/track.yaml',) -> None:
        self.config_path = config_path
        self.running = False  # 运行标志

    # config文件解析
    def parse_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # show键
        self.group_scale = cfg['show']['group_scale']  # 每组显示的视频数量
        self.show_w, self.show_h = cfg['show']['show_shape']  # 显示窗口大小
    
        # source键
        self.n = len(cfg['source'])  # 视频总数量
        self.source_dict = cfg['source']

    def initialize(self):
        # 参数计算
        self.scale = int(np.ceil(np.sqrt(self.group_scale)))  # 横纵方向的视频数量
        self.groups_num = int(np.ceil(self.n / self.group_scale))  # 组数

        # 按组排序
        group_dict = {i: [] for i in range(self.groups_num)}
        for source in self.source_dict.keys():
            group_dict[self.source_dict[source]['group_num']].append(source)

        self.source_list = []
        # self.group_index_dict = {i: None for i in range(self.n)}
        index = 0
        for i in group_dict.keys():
            self.source_list += group_dict[i]
            # self.group_index_dict[i] = [i+index for i in range(len(group_dict[i]))]
            # index += len(group_dict[i])

            # self.group_index_dict[i] = len(group_dict[i])

        # 初始化共享变量
        self.im_show = create_void_img((self.show_w, self.show_h), '正在加载')
        self.no_im = create_void_img((self.show_w, self.show_h), '未添加视频')
        # self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        self.video_reader_list = [None] * self.n
        self.tracker_thread_list = [None] * self.n
        self.q_in_list = [Queue(30) for _ in range(self.n)]
        # self.q_out_list = [Queue(30) for _ in range(n)]
        self.frame_list = [None] * self.n  # 用于存储每路视频的帧
        self.l_rate = None
        self.r_rate = None

        # 工具类
        self.splicer = SquareSplice(self.scale, show_shape=(self.show_w, self.show_h))
        self.display_manager = VideoDisplayManage(self.group_scale, self.groups_num, self.scale)

    def start(self):
        if not self.running:  # 防止重复启动
            self.running = True
            self.parse_config()
            self.initialize()

            # 更新结果线程
            show_thread = threading.Thread(target=self.update_results, daemon=False)
            show_thread.start()

            # 追踪检测线程
            for i, source in enumerate(self.source_list):
                # self.video_reader_list[i] = ReadVideo(source)

                tracker_thread = threading.Thread(
                    target=self.run_in_thread,
                    args=(self.source_dict[source],
                          self.q_in_list[i],
                          source, i),
                    daemon=False
                )
                self.tracker_thread_list[i] = tracker_thread
                tracker_thread.start()

    def stop(self):
        # self.im_show = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        self.im_show = create_void_img((self.show_w, self.show_h), '正在加载')

        self.running = False
        self.wait_thread()
        self.clear_up()
        print('模型已结束')

    def wait_thread(self):
        for tracker_thread in self.tracker_thread_list:
            tracker_thread.join()

    def clear_up(self):
        for video_reader in self.video_reader_list:
            video_reader.release()

        for q in self.q_in_list:
            q.queue.clear()

        # 清理显示管理器
        self.display_manager.reset()

        # 清理显存
        # if torch.cuda.is_available():
        #     with torch.cuda.device('cuda:0'):
        #         torch.cuda.empty_cache()
        #         torch.cuda.ipc_collect()

    def get_results(self):
        return self.im_show
    
    def click(self, l_rate):
        if self.display_manager.intragroup_index != -1:  # 单路显示
            self.l_rate = l_rate
            print('左键点击')
            result = '单击选中'
        else:
            result = '尚未进入单路显示模式'
        return result

    def dclick(self, r_rate):
        if self.display_manager.intragroup_index == -1:  # 宫格显示
            result = self.display_manager.select_intragroup(r_rate)
        else:
            self.r_rate = r_rate
            print('右键点击')
            result = '双击取消选中'
        return result
            

    def update_results(self):
        avg_fps = 0
        target_fps = 25
        while self.running:
            start_time = time.time()
            # group = [None] * self.group_scale
            for i, q in enumerate(self.q_in_list):
                # if not q.empty():  # 异步获取结果，防止忙等待
                    self.frame_list[i] = q.get()
                # else:
                #     time.sleep(wait_time/self.n)
                #     print('等待结果')
                #     if not q.empty():
                #         self.frame_list[i] = q.get()

            if self.display_manager.intragroup_index == -1:  # 宫格显示
                    start = self.display_manager.intergroup_index * self.group_scale
                    # videos = self.frame_list[start:start+self.group_len_dict[i]]
                    videos = self.frame_list[start:start+self.group_scale]
                    # no_videos = [self.no_im] * (self.group_scale - self.group_len_dict[i])
                    im_show = self.splicer(videos)
                # group = [None] * self.group_scale
            else:  # 单路显示
                im_show = self.frame_list[self.display_manager.intragroup_index]
                im_show = cv2.resize(im_show, (self.show_w, self.show_h))

            self.im_show = im_show.copy()
            self.im_show = cv2.putText(self.im_show, f"FPS={avg_fps:.2f}", (0, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps

            self.im_show = cv2.putText(self.im_show, f"{self.display_manager.get_display_index()}",
                                       (self.show_w-100, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # 显示组数和路数

            # end_time = time.time()
            if avg_fps > target_fps:  # 帧数稳定
                # print('显示空等待')
                time.sleep(1/target_fps - 1/avg_fps)
                # print('显示空等待')
            avg_fps = (avg_fps + (1 / (time.time() - start_time))) / 2
        print('collect_results结束')

    def read_frames(self, sources, q_out_list):
        pass

    # 需要抽象为类，每路加载不同的配置文件
    def run_in_thread(self, cfg_dict: dict, q: Queue, source, index):
        """
        """
        tracker = Track(
            weight=f"./weights/{cfg_dict['weight']}",
            imgsz=cfg_dict['detect_size'],
            classes=cfg_dict['classes'],
            tracker=cfg_dict['tracker'],
            vid_stride=cfg_dict['video_stride']
            )
        _, _ = tracker(self.im_show, {})  # warmup

        video_reader = ReadVideo(source)
        point_drawer = DelayDraw()
        show_id = dict()

        wait_time = 1 / 25
        while self.running:
            t1 = time.time()
            # print(f'第{index}路:{self.run}')
            frame, success = video_reader()
            if success:
                t2 = time.time()
                annotated_frame, show_id = tracker(frame, show_id, self.l_rate, self.r_rate)
                t3 = time.time()
            else:
                annotated_frame = frame

            if index == self.display_manager.intragroup_index:
                annotated_frame, self.l_rate, self.r_rate = point_drawer(annotated_frame, self.l_rate, self.r_rate)

            # if t3 - t1 < wait_time:  # 帧数稳定
            #     time.sleep(wait_time - (t2 - t1))
            #     print('检测空等待')
            # if index == 0:
            #     print(f'第{index}路检测时间:{t2-t1},{t3-t2}')  # 0.014,0.4/0.03
            # print(f'第{index}路检测:{tracker.count}')  # 0.014,0.291
            if q.full():
                q.get()
            q.put(annotated_frame)
        print(f"第{index}路已停止")
        video_reader.release()
