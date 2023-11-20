import time, os, re, subprocess, datetime
import numpy as np
import cv2, torch, torchvision
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


#### utils general 
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):

  nc = prediction.shape[2] - 5  # number of classes
  xc = prediction[..., 4] > conf_thres  # candidates

  # Checks
  assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
  assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

  # Settings
  min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
  max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
  time_limit = 10.0  # seconds to quit after
  redundant = True  # require redundant detections
  multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
  merge = False  # use merge-NMS

  t = time.time()
  output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
  for xi, x in enumerate(prediction):  # image index, image inference
    # Apply constraints
    x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
    x = x[xc[xi]]  # confidence

    # Cat apriori labels if autolabelling
    if labels and len(labels[xi]):
      lb = labels[xi]
      v = torch.zeros((len(lb), nc + 5), device=x.device)
      v[:, :4] = lb[:, 1:5]  # box
      v[:, 4] = 1.0  # conf
      v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
      x = torch.cat((x, v), 0)

    # If none remain process next image
    if not x.shape[0]:
      continue

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
      i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
      x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    else:  # best class only
      conf, j = x[:, 5:].max(1, keepdim=True)
      x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

    # Filter by class
    if classes is not None:
      x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
      continue
    elif n > max_nms:  # excess boxes
      x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
      i = i[:max_det]
    if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
      iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
      weights = iou * scores[None]  # box weights
      x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
      if redundant:
        i = i[iou.sum(1) > 1]  # require redundancy

    output[xi] = x[i]
    if (time.time() - t) > time_limit:
        break  # time limit exceeded

  return output

def xywh2xyxy(x):
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
  return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
  if ratio_pad is None:  # calculate from img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
  else:
    gain = ratio_pad[0][0]
    pad = ratio_pad[1]

  coords[:, [0, 2]] -= pad[0]  # x padding
  coords[:, [1, 3]] -= pad[1]  # y padding
  coords[:, :4] /= gain
  clip_coords(coords, img0_shape)
  return coords


def clip_coords(boxes, shape):
  if isinstance(boxes, torch.Tensor):  # faster individually
    boxes[:, 0].clamp_(0, shape[1])  # x1
    boxes[:, 1].clamp_(0, shape[0])  # y1
    boxes[:, 2].clamp_(0, shape[1])  # x2
    boxes[:, 3].clamp_(0, shape[0])  # y2
  else:  # np.array (faster grouped)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def box_iou(box1, box2):
  def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])
  area1 = box_area(box1.T)
  area2 = box_area(box2.T)
  inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
  return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


### utils plots

RANK = int(os.getenv('RANK', -1))

def check_pil_font(font='Arial.ttf', size=10):
  # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
  font = Path(font)
  font = font 
  try:
    return ImageFont.truetype(str(font) if font.exists() else font.name, size)
  except Exception:  # download if missing
    try:
      return ImageFont.truetype(str(font), size)
    except:
      pass

class Annotator:
  if RANK in (-1, 0):
    check_pil_font()  # download TTF if necessary

  def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
    self.pil = pil or not is_ascii(example) or is_chinese(example)
    if self.pil:  # use PIL
      self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
      self.draw = ImageDraw.Draw(self.im)
      self.font = check_pil_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                  size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
    else:  # use cv2
      self.im = im
    self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

  def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # Add one xyxy box to image with label
    if self.pil or not is_ascii(label):
      self.draw.rectangle(box, width=self.lw, outline=color)  # box
      if label:
        w, h = self.font.getsize(label)  # text width, height
        outside = box[1] - h >= 0  # label fits outside box
        self.draw.rectangle((box[0],
                              box[1] - h if outside else box[1],
                              box[0] + w + 1,
                              box[1] + 1 if outside else box[1] + h + 1), fill=color)
        # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
        self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
    else:  # cv2
      p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
      cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
      if label:
          tf = max(self.lw - 1, 1)  # font thickness
          w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
          outside = p1[1] - h - 3 >= 0  # label fits outside box
          p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
          cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
          cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                      thickness=tf, lineType=cv2.LINE_AA)

  def rectangle(self, xy, fill=None, outline=None, width=1):
    # Add rectangle to image (PIL-only)
    self.draw.rectangle(xy, fill, outline, width)

  def text(self, xy, text, txt_color=(255, 255, 255)):
    # Add text to image (PIL-only)
    w, h = self.font.getsize(text)  # text width, height
    self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

  def result(self):
    # Return annotated image as array
    return np.asarray(self.im)

  def draw_trk(self,thickness,centroids):
    [cv2.line(self.im, (int(centroids.centroids[i][0]),int(centroids.centroids[i][1])), 
              (int(centroids.centroids[i+1][0]),int(centroids.centroids[i+1][1])),
              (255,144,30), thickness=thickness) for i,_ in  enumerate(centroids.centroids)
              if i < len(centroids.centroids)-1 ]


  def draw_id(self, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
      x1, y1, x2, y2 = [int(i) for i in box]
      x1 += offset[0]
      x2 += offset[0]
      y1 += offset[1]
      y2 += offset[1]
      cat = int(categories[i]) if categories is not None else 0
      id = int(identities[i]) if identities is not None else 0
      data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
      label = str(id)
      (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
      cv2.rectangle(self.im, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
      cv2.putText(self.im, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
      cv2.circle(self.im, data, 4, (255,0,255),-1)

def is_ascii(s=''):
  # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
  s = str(s)  # convert list, tuple, None, etc. to str
  return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
  # Is string composed of any Chinese characters?
  return True if re.search('[\u4e00-\u9fff]', str(s)) else False

def git_describe(path=Path(__file__).parent):  # path must be a directory
  # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
  s = f'git -C {path} describe --tags --long --always'
  try:
    return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
  except subprocess.CalledProcessError:
    return ''  # not a git repository

def date_modified(path=__file__):
  # return human-readable file modification date, i.e. '2021-3-26'
  t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
  return f'{t.year}-{t.month}-{t.day}'

def select_device(device='', batch_size=0, newline=True):
  # device = 'cpu' or '0' or '0,1,2,3'
  s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
  device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
  cpu = device == 'cpu'
  if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
  elif device:  # non-cpu device requested
    os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
    assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
        f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

  cuda = not cpu and torch.cuda.is_available()
  if cuda:
    devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
    n = len(devices)  # device count
    if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
        assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
    space = ' ' * (len(s) + 1)
    for i, d in enumerate(devices):
        p = torch.cuda.get_device_properties(i)
        s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"  # bytes to MB
  else:
    s += 'CPU\n'

  if not newline:
    s = s.rstrip()
  return torch.device('cuda:0' if cuda else 'cpu')



########## CvTools


def processIm(im, device):
  im = torch.from_numpy(im).to(device)
  # im = im.half() # uint8 to fp16/32
  im = im.float()  
  im /= 255  
  if len(im.shape) == 3:
      im = im[None]  # expand for batch dim
  return im

def containsInBounds(point, bound):
  px, py = point
  x0, y0, x1, y1 = bound
  if (px > x0 and px < x1 and py > y0 and py < y1):
    return True
  return False

def getMidPoint(bound):
  x0, y0, x1, y1 = bound

  x = int(x0 + (x1-x0)/2)
  y = int(y0 + (y1-y0)/2)
  return x, y

def addCameraNumber(im0, camID):
  dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  cv2.rectangle(im0, (2, 1), (235, 65), (23,23,23), -1)
  if len(str(camID)) == 1:
      camID = '0'+str(camID)
  cv2.putText(im0, 'Cam: '+ str(camID), (15,40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
              fontScale = 1.5, color = (220,220,220), thickness = 2 )
  cv2.putText(im0, dt, (15,60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
              fontScale = 0.55, color = (220,220,220), thickness = 1 )
  return im0

def displayAlert(img, x_offset, y_offset, message=''):
    alert_img = cv2.imread("yolo/images/alert.png", -1)
    
    alert_img = cv2.resize(alert_img, (150,150), interpolation = cv2.INTER_AREA)

    y1, y2 = y_offset, y_offset + alert_img.shape[0]
    x1, x2 = x_offset, x_offset + alert_img.shape[1]

    alpha_s = alert_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * alert_img[:, :, c] +
                                alpha_l * img[y1:y2, x1:x2, c])
    if len(message):
        cv2.putText(img, message, (x_offset-25, y_offset+180), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1, color = (10,10,10), thickness = 4 )
    return img

def containsWithin(xyxy1, xyxy2):
    return xyxy1[0] <= xyxy2[0] and xyxy1[1] <= xyxy2[1] and xyxy1[2] >= xyxy2[2] and xyxy1[3] >= xyxy2[3]

