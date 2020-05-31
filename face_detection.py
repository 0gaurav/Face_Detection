import cv2 as cv
face_cascade=cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
img=cv.imread('faces3.jpg')
img=cv.resize(img,(700,500))
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.1,4)

for x,y,w,h in faces:
	cv.rectangle(img,(x,y),(x+w,y+h),(255),3)


cv.imshow('FACES',img)
cv.waitKey(0)
cv.destroyAllWindows()
