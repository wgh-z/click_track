# 工具箱
import cv2
import time
import yaml
import copy
import numpy as np
from utils.regional_judgment import point_in_rect
from utils.draw import create_void_img
import pprint
import json
from operator import itemgetter


def fps(func):
    """
    这是一个计算帧率的装饰器
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('fps:', 1 / (time.time() - start))
        return result

    return wrapper


def display_avg_fps_decorator(func):
    def wrapper(*args, **kwargs):
        avg_fps = 0
        t1 = time.time()
        result = func(*args, **kwargs)
        self = args[0]  # 获取self参数
        avg_fps = (avg_fps + (self.vid_stride / (time.time() - t1))) / 2
        self.im_show = cv2.putText(self.im_show, f"FPS={avg_fps:.2f}", (0, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 显示fps
        return result

    return wrapper


def interpolate_bbox(bbox1, bbox2, n=1):
    # bbox转np.array
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    # 计算插值后的 bbox 坐标
    bbox_n = bbox2 + (bbox2 - bbox1) * n

    # 返回插值后的 bbox
    return bbox_n


def create_model_config_flie(
        config_dict: dict,
        save_path: str = './cfg/track.yaml',
        default_cfg_path = './cfg/default.yaml'):

    if not isinstance(config_dict, dict):
        return '参数必须为一个字典'
    # else:
    #     if 'source' not in config_dict.keys():
    #         return '字典必须包含show键和source键'

    with open(default_cfg_path, 'r', encoding='utf-8') as default:
        default_cfg: dict = yaml.load(default, Loader=yaml.FullLoader)

    # 值检查
    # for k, v in config_dict['show'].items():
    #     if v is None:
    #         config_dict['show'][k] = default_cfg['show'][k]        

    # 检查show键
    for k, v in config_dict['show'].items():
        if v is None:
            config_dict['show'][k] = default_cfg['show'][k]

    # 检查source键
    temp_dict = copy.deepcopy(config_dict)
    for i, source in enumerate(config_dict['source'].keys()):
        for k, v in config_dict['source'][source].items():
            if v is None:
                if k == 'group_num':
                    temp_dict['source'][source]['group_num'] = i // 4  # 默认分组
                else:
                    temp_dict['source'][source][k] = default_cfg['source'][k]
    config_dict = copy.deepcopy(temp_dict)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True)
    return '配置文件更新成功'


def create_model_config_flie2(
        config_dict: dict,
        save_path: str = './cfg/track.yaml',
        default_cfg_path = './cfg/default.yaml'):

    if not isinstance(config_dict, dict):
        return '参数必须为一个字典'
    else:
        if 'source' not in config_dict.keys():
            return '参数必须包含source键，且属性不能为空'

    with open(default_cfg_path, 'r', encoding='utf-8') as default:
        default_cfg: dict = yaml.load(default, Loader=yaml.FullLoader)

    if 'show' not in config_dict.keys():  # 如果没有show键，则使用默认配置
        config_dict['show'] = default_cfg['show']
    else:  # 如果有show键，则检查属性是否为空
        for k, v in config_dict['show'].items():
            if v is None:  # 如果属性为空，则使用默认配置
                config_dict['show'][k] = default_cfg['show'][k]
    
    for source, v in config_dict['source'].items():
        if v is None:  # 如果属性为空，则使用默认配置
            config_dict['source'][source] = default_cfg['source']

    config_str = json.dumps(config_dict)
    new_dict = json.loads(config_str)

    temp_dict = copy.deepcopy(config_dict)
    for i, source in enumerate(config_dict['source'].keys()):
        for k, v in config_dict['source'][source].items():
            if v is None:
                if k == 'group_num':
                    temp_dict['source'][source]['group_num'] = i // 4  # 默认分组
                else:
                    temp_dict['source'][source][k] = default_cfg['source'][k]
    config_dict = copy.deepcopy(temp_dict)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_dict, f, allow_unicode=True)
    with open('./cfg/test.json', 'w', encoding='utf-8') as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)
    return '配置文件更新成功'


class SquareSplice:
    """
    这是一个拼接图片类，将一组图片按平方宫格拼接成一张大图
    """

    def __init__(self,
                 scale: int = 2,
                 show_shape: tuple = (1920, 1080),
                 line_color: tuple = (0, 0, 255)
                 ):
        self.scale = scale
        self.show_num = scale ** 2  # 总计能显示的视频数量
        self.line_width = int(4 / self.scale)
        self.line_color = line_color
        self.show_w, self.show_h = show_shape
        self.grid_w = int(self.show_w / self.scale)
        self.grid_h = int(self.show_h / self.scale)

        self.void_img = create_void_img((1280, 720))

    def __call__(self, im_lists: list, divider=True):
        im_list = im_lists.copy()
        im = np.zeros((self.show_h, self.show_w, 3), dtype=np.uint8)
        for i in range(self.show_num - len(im_list)):
            im_list.append(self.void_img)
        for i, im0 in enumerate(im_list):
            if im0 is None:
                im0 = self.void_img
            im0 = cv2.resize(im0, (self.grid_w, self.grid_h))
            im[self.grid_h * (i // self.scale):self.grid_h * (1 + (i // self.scale)),
                self.grid_w * (i % self.scale):self.grid_w * (1 + (i % self.scale))] = im0
        if divider:
            im = self.add_divider(im)
        return im

    # 添加分割线
    def add_divider(self, im):
        for line_num in range(self.scale+1):
            hline_y = line_num * self.grid_h  # 水平线的y坐标
            vline_x = line_num * self.grid_w  # 垂直线的x坐标
            im = cv2.line(im, (0, hline_y), (self.show_w - 1, hline_y),
                          self.line_color, self.line_width)  # 横线
            im = cv2.line(im, (vline_x, 0), (vline_x, self.show_h - 1),
                          self.line_color, self.line_width)  # 竖线
        return im


class Timer:
    """
    这是一个计时器类，用于将(data_dict.keys() - data_dict.keys() ∩ data_set)的元素延迟delay_count次后删除
    """

    def __init__(self, delay_count: int = 30):
        self.delay_count = delay_count

    def add_delay(self, data_dict: dict, id: int):
        data_dict[id] = self.delay_count
        return data_dict

    def __call__(self, data_set: set, data_dict: dict):
        temp_dict = data_dict.copy()
        for element in data_dict.keys():
            if element not in data_set:
                temp_dict[element] -= 1
                if temp_dict[element] == 0:
                    del temp_dict[element]
            else:
                data_dict[element] = self.delay_count  # 重置计时器
        return temp_dict


class Interpolator:
    """
    这是一个插值器类，用于对检测结果进行插值
    参数：
        vid_stride: int, 视频检测间隔
        mode: str, 插值模式： copy, linear, quadratic, cubic
    """

    def __init__(self, vid_stride: int = 2, mode: str = 'copy'):
        self.vid_stride = vid_stride
        self.stride_counter = vid_stride
        self.mode = mode
        self.prior_det = None

    def __call__(self, current_det):
        if self.vid_stride == 1:  # vid_stride为1时，不进行插值
            return current_det
        if self.mode == 'copy':
            return self.copy(current_det)

    def copy(self, current_det):
        """
        超快速插值模式，即不插值，直接返回上一帧检测结果
        """
        if self.stride_counter == self.vid_stride:
            # self.prior_det = current_det[:, :4]
            self.prior_det = current_det
            self.stride_counter = 0
        else:
            self.stride_counter += 1
            # current_det[:, :4] = interpolate_bbox(self.prior_det, current_det[:, :4], self.stride_counter)
            current_det = self.prior_det
        return current_det

    def balanced(self, current_det):
        """
        平衡插值模式，即每隔vid_stride帧进行一次插值
        """
        if self.stride_counter == self.vid_stride:
            if self.prior_det is not None:
                current_det = interpolate_bbox(self.prior_det, current_det, self.stride_counter)
            self.prior_det = current_det
            self.stride_counter = 0
        else:
            self.stride_counter += 1
        return current_det

    def slow(self, current_det):
        """
        慢速插值模式，即每帧都进行插值
        """
        if self.prior_det is not None:
            current_det = interpolate_bbox(self.prior_det, current_det, self.stride_counter)
        self.prior_det = current_det
        return current_det


class ClickFilterDet:
    """
    这是一个点击过滤器类，用于使用点击坐标过滤或回恢复yolo检测结果
    """

    def __init__(self, frame):
        self.frame = frame
        self.click_point = None

        # 30帧清空离场id
        self.timer = Timer(30)

    def __call__(self, det, l_point=None, r_point=None):
        in_show = []
        for i, *xyxy in enumerate(reversed(det[:, :4])):
            if point_in_rect(l_point, xyxy):
                # show_id.append(id)
                show_id = self.timer.add_delay(show_id, id)
                l_point = None

            if point_in_rect(r_point, xyxy):
                try:
                    # show_id.remove(id)
                    del show_id[id]
                except:
                    pass
                r_point = None
        new_det = np.take(det, in_show, axis=1)
        return l_point, r_point
