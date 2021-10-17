import cv2
import numpy as np
import pandas as pd
from pandas.core import frame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context()

#Fetching the data from the provided files
X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels-C122.csv')['labels']
print(pd.Series(y).value_counts())

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)

#Training and testing the Logistic Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver= 'saga', multi_class= 'multinomial').fit(X_train_scaled,y_train)

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test,y_pred)

print(accuracy)

#Strating the cam and doing the functionality
cap  = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right = (int(width/2+56),int(height/2+56))

        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)

        #Creating the region of interest
        # roi = (r = Region, o = Of, i = Interest) = Region of Interest 
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        im_pil = Image.fromarray(roi)

        #Convert to grayscle image - 'L' format means each pixel is
        #represented by a single value from 0 to 255
        img_bw = im_pil.convert('L')
        img_bw_resized = img_bw.resize((28,28),Image.ANTIALIAS)
        #Inverting the image
        img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
        pixel_filter = 20
        #Converting to scalar quantity
        min_pixel = np.percentile(img_bw_resized_inverted,pixel_filter)
        #Using clip to limit the values between 0,255
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted - min_pixel,0,255)
        max_pixel = np.max(img_bw_resized_inverted)
        #Converting the image to an array
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel
        #Creating a test sample and making a prediction
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("Predicted Class is: ", test_pred)

        #Making the frame as gray
        cv2.imshow('Frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()