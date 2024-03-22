import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tensorflow as tf

import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image,ImageTk
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array


top = tk.Tk()
top.geometry('1920x800')
top.title('Drowsiness Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font = ('arial',15,'bold'))
sign_image = Label(top)

def drowsiness():    
    #Initializing the face and eye cascade classifiers from xml files

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    
    #Variable store execution state

    first_read = True
    
    #Starting the video capture

    cap = cv2.VideoCapture(0)

    ret,img = cap.read()
    

    while(ret):

        ret,img = cap.read()

        #Converting the recorded image to grayscale

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Applying filter to remove impurities

        gray = cv2.bilateralFilter(gray,5,1,1)
    

        #Detecting the face for region of image to be fed to eye classifier

        faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))
        
        if(len(faces)>0):

            for (x,y,w,h) in faces:

                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    

                #roi_face is face which is input to eye classifier

                roi_face = gray[y:y+h,x:x+w]

                roi_face_clr = img[y:y+h,x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
    

                #Examining the length of eyes object for eyes
                c=0
                if(len(eyes)>=2):

                    #Check if program is running for detection 


                    if(first_read):

                        cv2.putText(img, 

                        "Eyes open!", (70,70),

                        cv2.FONT_HERSHEY_PLAIN, 3,

                        (0,0,255),2)
                    else:
                        cv2.waitKey(3000)

                        first_read=True

                else:

                    if(first_read):

                        #To ensure if the eyes are present before starting

                        cv2.putText(img, 

                        "Blink detected!", (70,70),

                        cv2.FONT_HERSHEY_PLAIN, 3,

                        (0,0,255),2)
                    else:
                        cv2.waitKey(3000)

                        first_read=True

        else:

            cv2.putText(img,

            "Sleep detected!...",(100,100),

            cv2.FONT_HERSHEY_PLAIN, 3, 

            (0,255,0),2)
    

        #Controlling the algorithm with keys

        cv2.imshow('img',img)

        a = cv2.waitKey(1)

        if(a==ord('q')):

            break

        elif(a==ord('s') and first_read):

            #This will start the detection

            first_read = False
    
    cap.release()
    cv2.destroyAllWindows()


upload = Button(top, text = "Drowsiness Detection", command=drowsiness,padx=2,pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
upload.pack(side='bottom',pady=20)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading =Label(top, text='Drowsiness Detector',pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()
