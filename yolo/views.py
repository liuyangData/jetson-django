import torch.nn.functional as F
from yolo.models.tools import non_max_suppression, scale_coords, Annotator, processIm, addCameraNumber, containsInBounds, getMidPoint, displayAlert, select_device
from yolo.models.loading import DetectMultiBackend, letterbox
import cv2, datetime, numpy as np, time
from django.http import StreamingHttpResponse
from yolo.database import Alerts

class Person:
	
	xyxt = [0,0,0,0]
	currPPE = 'ALL PPE'
	nextPPE = ''
	posCounter = 0 
	instance = 1

	writingVideo = False
	output = ''
	vidFrame = 0
	vidURL = ''
	vid = ''

	helmet = 0 
	mask = 0
	vest = 0
	maskUnsure = 0
	helmetUnsure = 0

	complianceScore = 0
	maxPPEScore = 100
	percentageScore = 0
	color = (0, 51, 255)

	alertHistory = []
	currStatus = ['helmetUnsure', 'maskUnsure','vestOff', '', '100', ''] 

	vType = ''
	videoName = ''
	imgName = ''
	resultFileName = ''

	dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

	def __init__(self, personId):
		self.personId = personId
		self.alertId = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
		self.setFileNames()

	def setFileNames(self):
		self.videoName = f'yolo/outputs/alert-{self.alertId}.mp4'
		self.imgName = f'yolo/outputs/thumbnail-{self.alertId}.jpg'
		self.resultFileName = f"alert-{self.alertId}"

	def updatePPE(self, det):
		if det == 'helmet':
			self.helmet = 1
			self.currStatus[0] = 'helmetOn'
		elif det == 'vest':
			self.vest = 1
			self.currStatus[2] = 'vestOn'
		elif det == 'mask':
			self.mask = 1
			self.currStatus[1] = 'maskOn'
		elif det == 'noHelmet':
			self.helmet = 0
			self.helmetUnsure = 0
			self.currStatus[0] = 'helmetOff'
		elif det == 'noMask':
			self.mask = 0
			self.maskUnsure = 0
			self.currStatus[1] = 'maskOff'

		self.currStatus[3] = str(datetime.datetime.now())
		self.complianceScore = max(self.helmet,self.helmetUnsure)*60 + self.vest*30 + max(self.mask, self.maskUnsure)*10
		self.currStatus[4] = str(self.complianceScore)

	def getPersonColor(self):
		if self.maxPPEScore:
			self.percentageScore = int(self.complianceScore/self.maxPPEScore * 100) 
		else:
			self.percentageScore = 100

		if self.percentageScore == 100:
			self.color = (0, 230, 0)
		elif self.percentageScore >= 90:
			self.color = (30, 216, 200)
		elif self.percentageScore >= 70:
			self.color = (45, 180, 235)
		elif self.percentageScore >= 60:
			self.color = (90, 165, 235)
		elif self.percentageScore >= 40:
			self.color = (90, 90, 255)
		elif self.percentageScore >= 30:
			self.color = (45, 45, 255)
		elif self.percentageScore >= 10:
			self.color = (10, 10, 255)
		else:
				self.color = (0, 0, 255)

		if self.maskUnsure and not self.mask:
			self.color = (150, 150, 150)
		if self.helmetUnsure and not self.helmet:
			self.color = (150, 150, 150)
		return self.color

	def getVtype(self):
		result = 'NO CLASS'
		if self.complianceScore == 0:
			result = 'NO PPE'
		elif self.complianceScore == 10:
			result = 'MASK ONLY'
		elif self.complianceScore == 30:
			result = 'VEST ONLY'
		elif self.complianceScore == 40:
			result = 'MASK VEST ONLY'
		elif self.complianceScore == 60:
			result = 'HELMET ONLY'
		elif self.complianceScore == 70:
			result = 'HELMET MASK ONLY'
		elif self.complianceScore == 90:
			result = 'HELMET VEST ONLY'
		elif self.complianceScore == 100:
			result = 'ALL PPE'

		self.vType = result 
		return result

	def updateMaxScore(self, score):
		self.maxPPEScore = score

	def getLabel(self):
		questionMark = ''
		if self.maskUnsure and not self.mask:
			questionMark = ' ?'
		if self.helmetUnsure and not self.helmet:
			questionMark = ' ?'
		return f'ID-{self.personId}: PPE Score {self.percentageScore }% - {self.vType}{questionMark}'

	def resetPPE(self):
		if self.helmet:
			self.helmetUnsure = 1 
			self.currStatus[0] = 'helmetUnsure'
		if self.mask:
			self.maskUnsure = 1
			self.currStatus[1] = 'maskUnsure'

		self.helmet = 0
		self.vest = 0
		self.mask = 0
		self.currStatus[2] = 'vestOff'

	def frameIncrement(self, xyxy, im, camName):
		self.camName = camName
		self.xyxy = xyxy
		vType = self.getVtype()

		if vType != self.currPPE:
			if self.nextPPE == '':
				self.nextPPE = vType
			else:
				if self.nextPPE == vType:
					self.posCounter =  self.posCounter + 1
				else:
					self.posCounter = 0
					self.nextPPE = ''
		else:
			self.posCounter = 0

		# if different from currPPE for threshold duration
		if self.posCounter > 100 and self.percentageScore < 100:
			# if videoing, write frame, else initiate Video
			if self.writingVideo:
				self.output.write(im)
				self.vidFrame = self.vidFrame + 1
			else:
				self.output = cv2.VideoWriter(self.videoName, cv2.VideoWriter_fourcc(*'mp4v'), 3.0, (im.shape[1],im.shape[0]))
				self.writingVideo = 1

			# if video exceed max duration,
			if self.vidFrame > 30:
				# release and push video
				
				self.output.release()
				print('video released')
				self.writingVideo = False
				self.vidFrame = 0
				self.saveThumbnail(im)
				self.instance = self.instance + 1
				self.sendNotification()
				self.alertHistory.append(self.currStatus.copy())
				self.currPPE = self.nextPPE
				self.nextPPE = '' 
				self.setFileNames()

	def sendNotification(self):
		violations = []
		if self.vType == 'NO PPE':
			violations = ['No Helmet', 'No Mask', 'No Vest']
		elif self.vType == 'MASK ONLY':
			violations = ['No Helmet', 'No Vest']
		elif self.vType == 'VEST ONLY':
			violations = ['No Helmet', 'No Mask']
		elif self.vType == 'MASK VEST ONLY':
			violations = ['No Helmet']
		elif self.vType == 'HELMET ONLY':
			violations = ['No Vest', 'No Mask']
		elif self.vType == 'HELMET MASK ONLY':
			violations = [ 'No Vest']
		elif self.vType == 'HELMET VEST ONLY':
			violations = [ 'No Mask']

		for violation in violations:
			alert = Alerts(datetime=str(self.dt), camera=self.camName, violation=violation)
			alert.save()
      
		self.alertId = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ") 
		self.resultFileName = f"alert-{self.alertId}"

	def resetPerson(self):
		self.currPPE = 'ALL PPE'
		self.nextPPE = ''
		self.posCounter = 0 

		self.writingVideo = False
		self.vidFrame = 0

	def saveThumbnail(self, im):
		x0 = int(self.xyxy[0].cpu().numpy())
		y0 = int(self.xyxy[1].cpu().numpy())
		x1 = int(self.xyxy[2].cpu().numpy())
		y1 = int(self.xyxy[3].cpu().numpy())
		cv2.imwrite(self.imgName , im[y0:y1, x0:x1] )
		self.currStatus[5] = self.imgName

personList = [Person(i) for i in range(100)]

def detectPPE(im, im0s, device, modelPPE, modelYOLO, camName):
  im = processIm(im, device) 
  im0 = im0s[0].copy()
  im0 = cv2.resize(im0, (1280,720), interpolation = cv2.INTER_AREA)

  annotator = Annotator(im0, line_width=1)

  # YOLO model 
  names = modelYOLO.names
  try:
    det = non_max_suppression(modelYOLO(im))[0]
  except:
    im = F.interpolate(im, size=(640, 640))
    det = non_max_suppression(modelYOLO(im))[0]

  # PPE model
  namesPPE = ['helmet', 'mask','noHelmet', 'noMask', 'vest']
  colorsPPE = [(51,183,51), (204,153, 0), (35,0, 224), (0, 75, 240), (204, 51, 51) ]
  detPPE = non_max_suppression(modelPPE(im, False, False), 0.25, 0.45, None, False, 1000)[0]

  personId = 0

  detPerson = []

  if len(det):
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round() # Rescale to im0 size

    for *xyxy, conf, cls in reversed(det):
      if conf > 0.6 and names[int(cls)] == 'person':
        if(xyxy[1] > 40):
          xyxy[1] = xyxy[1] - 40
        else:
          xyxy[1] = xyxy[1] - xyxy[1]

        if len(detPerson) == 0:
          detPerson.append([xyxy, conf, cls])
        else:
          for i in range(len(detPerson)):
            if xyxy[0] < detPerson[i][0][0]:
              detPerson.insert(i, [xyxy, conf, cls]) 
              break
            elif i == len(detPerson) - 1:
              detPerson.append([xyxy, conf, cls])
              break

    for det in detPerson:
      xyxy = det[0]
      conf = det[1]
      cls = det[2]

      personId = personId + 1 

      personTracker = personList[personId]
      personTracker.updateMaxScore(100)
      personTracker.resetPPE()
      frame = im0
      
      detPersonPPE = []
      if len(detPPE):
        detPPE[:, :4] = scale_coords(im.shape[2:], detPPE[:, :4], im0.shape).round() # Rescale to im0 size
        for *xyxyPPE, conf, cls in reversed(detPPE):
          if conf > 0.55:
            ppePoint = getMidPoint(xyxyPPE)
            ppe = namesPPE[int(cls)]

            if containsInBounds(ppePoint, xyxy):
              annotator.box_label(xyxyPPE, ppe, color=colorsPPE[int(cls)])
              if ppe == 'helmet' or ppe == 'mask':
                detPersonPPE.insert(0, ppe)
              else:
                detPersonPPE.append(ppe)
            frame = annotator.result()
              
      for ppe in detPersonPPE:
        personTracker.updatePPE(ppe)

        # save alert history
        personTracker.frameIncrement(xyxy, frame, camName)
        label =personTracker.getLabel()
        color = personTracker.getPersonColor()
        annotator.box_label(xyxy, label, color=color)
                      
  frame = annotator.result()

  frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_AREA)

  # add icons
  for i, det in enumerate(detPerson):
    xyxy = det[0]
    person = personList[i+1]
    try:
      frame = displayPPEIcons(frame, xyxy, person)
    except:
      pass


  cv2.rectangle(frame, (900, 3), (1260, 50), (107,65,123), -1)
  cv2.putText(frame, 'PPE Compliance', (910,35), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
              fontScale = 1, color = (255,255,255), thickness = 2 )
  frame = addCameraNumber(frame, 0)    

  return frame

def displayPPEIcons(img, xyxy, person):
  x_offset = int(xyxy[0].cpu().numpy())*2 + 10
  y_offset = int(xyxy[1].cpu().numpy())*2 + 10
  icons = []

  if person.helmet:
    icons.append('helmetOn') 
  elif person.helmetUnsure:
    icons.append('helmetUnsure')
  else : 
    icons.append('helmetOff')

  if person.mask:
    icons.append('maskOn') 
  elif person.maskUnsure:
    icons.append('maskUnsure')
  else : 
    icons.append('maskOff')

  icons.append('vestOn') if person.vest else icons.append('vestOff')

  for icon in icons:
    alert_img = cv2.imread(f"yolo/images/{icon}.png", -1)
    alert_img = cv2.resize(alert_img, (60,60), interpolation = cv2.INTER_AREA)

    y1, y2 = y_offset, y_offset + alert_img.shape[0]
    x1, x2 = x_offset, x_offset + alert_img.shape[1]

    alpha_s = alert_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
      img[y1:y2, x1:x2, c] = (alpha_s * alert_img[:, :, c] +
                              alpha_l * img[y1:y2, x1:x2, c])
    
    y_offset = y_offset + 70

  return img

def detectYolov5_PPE(rtsp):
	if rtsp == '0': rtsp = int(rtsp)
	cam_name = 'Camera RTSP'
	device = select_device()     
	modelPPE = DetectMultiBackend('yolo/weights/ppe.pt' , device=device)
	modelPPE.model.float()
	modelYOLO = DetectMultiBackend('yolo/weights/yolox.pt', device=device)
	modelYOLO.model.float()
	cap = cv2.VideoCapture(rtsp)
	
	while True:
		_, frame = cap.read()
		img0 = [frame.copy()]
		im = [letterbox(x, 640, stride=32, auto=True)[0] for x in img0]
		im = np.stack(im, 0)
		im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
		im = np.ascontiguousarray(im)
		im0 = detectPPE(im, img0, device, modelPPE, modelYOLO, cam_name)
		results = cv2.imencode('.jpg', im0)[1].tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + results + b'\r\n')


class Camera:
    violations = []
    active = 0
    currViolation = None
    noRepeatdt = datetime.datetime.now() + datetime.timedelta(minutes=-1000)

    def __init__(self, camId):
        self.camId = camId

    def addViolation(self, violationType):
        instance = len(self.violations)
        self.currViolation = Violation(self.camId, instance , violationType)
        self.violations.append(self.currViolation)
        self.active = 1

class Violation:

    frame = 0
    posCount = 0
    negCount = 0
    active = 1
    video = 0
    output = 0
    videoName = ''

    def __init__(self, camId, instance, violationType):
        self.camId = camId
        self.camName = 'Camera-00' + str(camId) if camId < 10 else 'Camera-0' + str(camId) 
        self.instance = instance 
        self.violationType = violationType
        self.alertId = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.videoName = f'yolo/outputs/alert-{self.alertId}.mp4'
        self.imgName = f'yolo/outputs/thumbnail-{self.alertId}.jpg'

    def frameIncrement(self, positive, frame):
        completedFlag = False
        if self.video:
            self.frame = self.frame + 1
        if positive:
            self.posCount = self.posCount + 1
            if self.posCount > 10:
                self.negCount = 0
                if self.video == 0 :
                    self.output = cv2.VideoWriter(self.videoName, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (frame.shape[1],frame.shape[0]))

                    self.video = 1
                    cv2.imwrite(self.imgName, frame)
                if self.video:
                    self.output.write(frame)
            
            if self.posCount > 100 :
                self.output.release()
                self.sendNotification()
                self.video = 0
                completedFlag = True

        else:
            self.negCount = self.negCount + 1
            if self.negCount > 15:
                self.posCount = 0
                if self.output:
                    self.output.release()
                    self.sendNotification()
                    completedFlag = True
                self.video = 0
            if self.video:
                self.output.write(frame)
        return completedFlag

    def sendNotification(self):
        alertMsg = f'There is a new Violation Alert'
        payload = {"dt": self.dt, "duration": "00:00:10", "violation": self.violationType, "camera": self.camName}
        print(payload)

camList = [ Camera(i) for i in range(100) ]

def smartNotify(c, violationType, frame):
  if len(violationType) > 0:
    if c.noRepeatdt < datetime.datetime.now():
      c.active = 1

  if c.active:
    if c.currViolation:
      posNeg = len(violationType) > 0
      completed = c.currViolation.frameIncrement(posNeg, frame)
      if completed:
        c.currViolation = None
        c.active = 0
        c.noRepeatdt = datetime.datetime.now() + datetime.timedelta(minutes=2)
    else:
      c.addViolation(violationType)

def detectHelmet(modelHelmet, device, im, im0, camId ):

  im = processIm(im, device) 
#   im = F.interpolate(im, size=(640, 640))
  im0 = im0[0].copy()
  pred = modelHelmet(im)
  pred = non_max_suppression(pred)
  names = ['helmet', 'noHelmet']
  colors = [(39,91,18), (33,50,240)]
  minConf = 0.6
  zone = []
  
  det = pred[0]
  annotator = Annotator(im0, line_width=3, example=str(names))
  violation = ''

  if len(det):
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round() # Rescale to im0 size
    for *xyxy, conf, cls in reversed(det):
      if conf > minConf:
        annotator.box_label(xyxy, names[int(cls)], color=colors[int(cls)])
        if int(cls) == 1:
          im0 = displayAlert(im0, 1050, 50, "NO HELMET!")
          violation = 'No Helmet'

  if len(zone) > 0:
    cv2.rectangle(im0, (zone[0], zone[1]), (zone[2], zone[3]), (23,23,255), 5)
    cv2.putText(im0, 'Detection Zone', (zone[0]+15,zone[1]+30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 1 , color = (250,250,250), thickness = 1 )

  cv2.rectangle(im0, (900, 3), (1260, 50), (17,165,123), -1)
  cv2.putText(im0, 'Helmet Compliance', (920,35), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
              fontScale = 1, color = (255,255,255), thickness = 2 )

  im0 = addCameraNumber(im0, camId)
  frame = annotator.result()
  frame = cv2.resize(frame, (480*3,270*3), interpolation = cv2.INTER_AREA)

  smartNotify( camList[camId], violation, frame)

  return frame



def detect_helmet(cam):
  
  device = select_device()

  cam_name = f"yolo/videos/stream{cam}.mp4"

  modelHelmet = DetectMultiBackend('yolo/weights/helmet.pt' , device=device)
  modelHelmet.model.float()

  cap = cv2.VideoCapture(cam_name)

  while True:
    _, frame = cap.read()
    img0 = [frame.copy()]
    im = [letterbox(x, 640, stride=32, auto=True)[0] for x in img0]
    im = np.stack(im, 0)
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)
    im0 = detectHelmet(modelHelmet, device, im, img0, int(cam))
    results = cv2.imencode('.jpg', im0)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + results + b'\r\n')
    time.sleep(0.1)

def stream_helmet(request):
  response = StreamingHttpResponse(detect_helmet(1),content_type="multipart/x-mixed-replace;boundary=frame")
  return response

def stream_helmet_vid(request, vid):
  response = StreamingHttpResponse(detect_helmet(vid),content_type="multipart/x-mixed-replace;boundary=frame")
  return response

def stream_PPE(request, rtsp):
  response = StreamingHttpResponse(detectYolov5_PPE(rtsp),content_type="multipart/x-mixed-replace;boundary=frame")
  return response

def stream_PPE_webcam(request):
  response = StreamingHttpResponse(detectYolov5_PPE(0),content_type="multipart/x-mixed-replace;boundary=frame")
  return response
