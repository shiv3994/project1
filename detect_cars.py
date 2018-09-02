import cv2

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#load test image
test1 = cv2.imread('d:\\test_image.jpg')

#load cascade classifier training file for haarcascade 
car_cascade = cv2.CascadeClassifier('d:\\cars.xml')



#convert the test image to gray image as opencv detector expects gray images 
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)


#detect multiscale images 
cars = car_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=0);  
 
#print the number of cars found 
print('cars found: ', len(cars))


#draw rectangles around cars
for (x, y, w, h) in cars:
  cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 0)


#convert image to RGB and show image #
cv2.imshow('op',convertToRGB(test1))
cv2.waitKey(0) 
cv2.destroyAllWindows()
