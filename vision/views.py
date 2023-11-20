import cv2
from django.http import HttpResponse, StreamingHttpResponse

def index(request):
  return HttpResponse("Computer Vision App: Human Detector.")

def detect():

	# get the webcam
	cap = cv2.VideoCapture(0)
	ret,img = cap.read()

	# load person detector from cv2 
	model = cv2.HOGDescriptor()
	model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# loop the video frames
	while ret:
		# read the frames
		ret,img = cap.read()

		# resize the img 
		img = cv2.resize(img, (640, 480))

		# detect people in the img
		(rects, weights) = model.detectMultiScale(img, winStride=(4, 4), padding=(4, 4), scale=1.05)
		
		# draw the bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# return the results as a stream of http response
		results = cv2.imencode('.jpg', img)[1].tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + results + b'\r\n')

def stream(request):
	return StreamingHttpResponse(detect(), content_type="multipart/x-mixed-replace;boundary=frame")
