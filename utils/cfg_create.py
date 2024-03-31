class ModelCFG:
    '''
    用于生成模型的配置文件
    '''
    def __init__(self, model_name, model_path, cfg_path, cfg_name):
        self.model_name = model_name
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.cfg_name = cfg_name

    def __call__(self):
        with open(self.cfg_path, 'w', encoding='utf-8') as f:
            f.write(f'model_name: {self.model_name}\n')
            f.write(f'model_path: {self.model_path}\n')
            f.write(f'cfg_name: {self.cfg_name}\n')
            f.write(f'cfg_path: {self.cfg_path}\n')
            f.write(f'weight: {self.model_path}\n')
            f.write(f'stream: list.streams\n')
            f.write(f'imgsz: 640\n')
            f.write(f'vid_stride: 1\n')
            f.write(f'ip:')