import cv2

#creates a cascade classifier object
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# loads the img
img=cv2.imread("picture.jpg")

#loads in greyscale to increase cv2 accuracy
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#searches the img for a face and scales down 5%
faces=face_cascade.detectMultiScale(gray_img,
scaleFactor=1.1,
minNeighbors=5)

#draws the rectangle on the img
for x, y, w, h in faces:
    img=cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)


print(type(faces))
print(faces)

resized=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

cv2.imshow("Gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
