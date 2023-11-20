import time, os, re, subprocess, glob, math, urllib, requests, cv2, torch, yaml
import numpy as np
from pathlib import Path
from threading import Thread
import torch.nn as nn
import pandas as pd

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes

class LoadVideos:
  def __init__(self, path, img_size=640, stride=32, auto=True):
    p = str(Path(path).resolve())  # os-agnostic absolute path
    if '*' in p:
      files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
      files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
      files = [p]  # files
    else:
      raise Exception(f'ERROR: {p} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
    ni, nv = len(images), len(videos)
    
    self.count = 0

    self.img_size = img_size
    self.stride = stride
    self.files = images + videos
    self.nf = ni + nv  # number of files
    self.video_flag = [False] * ni + [True] * nv
    self.mode = 'image'
    self.auto = auto
    if any(videos):
      self.new_video(videos[0])  # new video
    else:
      self.cap = None
    assert self.nf > 0, f'No images or videos found in {p}. ' \
                        f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

  def __iter__(self):
    self.count = 0
    return self

  def next(self):
    if self.count == self.nf:
      raise StopIteration
    path = self.files[self.count]

    if self.video_flag[self.count]:
      # Read video
      self.mode = 'video'
      ret_val, img0 = self.cap.read()
      while not ret_val:
        self.count += 1
        self.cap.release()
        if self.count == self.nf:  # last video
          raise StopIteration
        else:
          path = self.files[self.count]
          self.new_video(path)
          ret_val, img0 = self.cap.read()

      self.frame += 1
      s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
    else:
      # Read image
      self.count += 1
      img0 = cv2.imread(path)  # BGR
      assert img0 is not None, f'Image Not Found {path}'
      s = f'image {self.count}/{self.nf} {path}: '

    # Padded resize
    img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img, img0

  def new_video(self, path):
    self.frame = 0
    self.cap = cv2.VideoCapture(path)
    self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

  def __len__(self):
    return self.nf  # number of files
  
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
  # Resize and pad image while meeting stride-multiple constraints
  shape = im.shape[:2]  # current shape [height, width]
  if isinstance(new_shape, int):
      new_shape = (new_shape, new_shape)

  # Scale ratio (new / old)
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  if not scaleup:  # only scale down, do not scale up (for better val mAP)
    r = min(r, 1.0)

  # Compute padding
  ratio = r, r  # width, height ratios
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
  if auto:  # minimum rectangle
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
  elif scaleFill:  # stretch
    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
    ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

  dw /= 2  # divide padding into 2 sides
  dh /= 2

  if shape[::-1] != new_unpad:  # resize
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
  return im, ratio, (dw, dh)


class LoadStreams:
  def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
    self.mode = 'stream'
    self.img_size = img_size
    self.stride = stride
    self.count=0
    if os.path.isfile(sources):
      with open(sources) as f:
        sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
    else:
      sources = [sources]

    n = len(sources)
    self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
    self.sources = [clean_str(x) for x in sources]  # clean source names for later
    self.auto = auto
    for i, s in enumerate(sources):  # index, source
      # Start thread to read frames from video stream
      st = f'{i + 1}/{n}: {s}... '
      s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
      cap = cv2.VideoCapture(s)
      assert cap.isOpened(), f'{st}Failed to open {s}'
      w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
      self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
      self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
      _, self.imgs[i] = cap.read()  # guarantee first frame
      self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
      self.threads[i].start()
    # check for common shapes
    s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
    self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

  def update(self, i, cap, stream):
    # Read stream `i` frames in daemon thread
    n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
    while cap.isOpened() and n < f:
      n += 1
      cap.grab()
      if n % read == 0:
        success, im = cap.retrieve()
        if success:
          self.imgs[i] = im
        else:
          self.imgs[i] = np.zeros_like(self.imgs[i])
          cap.open(stream)  # re-open stream if signal was lost
      time.sleep(1 / self.fps[i])  # wait time

  def __iter__(self):
    self.count = -1
    return self

  def __next__(self):
    self.count += 1
    if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
      cv2.destroyAllWindows()
      raise StopIteration
    
    # Letterbox
    img0 = self.imgs.copy()
    img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

    # Stack
    img = np.stack(img, 0)

    # Convert
    img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    img = np.ascontiguousarray(img)

    return self.sources, img, img0, None, ''

  def __len__(self):
    return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years

def clean_str(s):
  # Cleans a string by replacing special characters with underscore _
  return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)



########## BACKEND 


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False, data=None):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.jit or self.onnx or self.engine:  # warmup types
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs

def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def export_formats():
    # YOLOv5 export formats
    x = [['PyTorch', '-', '.pt'],
         ['TorchScript', 'torchscript', '.torchscript'],
         ['ONNX', 'onnx', '.onnx'],
         ['OpenVINO', 'openvino', '_openvino_model'],
         ['TensorRT', 'engine', '.engine'],
         ['CoreML', 'coreml', '.mlmodel'],
         ['TensorFlow SavedModel', 'saved_model', '_saved_model'],
         ['TensorFlow GraphDef', 'pb', '.pb'],
         ['TensorFlow Lite', 'tflite', '.tflite'],
         ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite'],
         ['TensorFlow.js', 'tfjs', '_web_model']]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix'])



class Ensemble(nn.ModuleList):
  # Ensemble of models
  def __init__(self):
    super().__init__()

  def forward(self, x, augment=False, profile=False, visualize=False):
    y = []
    for module in self:
      y.append(module(x, augment, profile, visualize)[0])
    y = torch.cat(y, 1)  # nms ensemble
    return y, None  # inference, train output
  
class Conv(nn.Module):
  # Standard convolution
  def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
    super().__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
    self.bn = nn.BatchNorm2d(c2)
    self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

  def forward(self, x):
    return self.act(self.bn(self.conv(x)))

  def forward_fuse(self, x):
    return self.act(self.conv(x))

def autopad(k, p=None):  # kernel, padding
  # Pad to 'same'
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p
    
def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from yolo.models.yolo import Detect, Model
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble


def attempt_download(file, repo='ultralytics/yolov5'):  # from utils.downloads import *; attempt_download()
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                print(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
            assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response['tag_name']  # i.e. 'v1.0'
        except Exception:  # fallback plan
            assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
                      'yolov5n6.pt', 'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except Exception:
                tag = 'v6.0'  # current release

        if name in assets:
            safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

    return str(file)




def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print('')
