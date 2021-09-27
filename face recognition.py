import cv2
import face_recognition
import numpy as np
import os
import datetime

imagepath='Images_attendance'
images=[]
classname=[]
mylist=os.listdir(imagepath)


for i in mylist:
    current_image=cv2.imread(f"{imagepath}\{i}")
    images.append(current_image)
    name=os.path.splitext(i)[0]
    classname.append(name)

def encoding(image):
    encoding_list=[]
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoding_list.append(encode)
    return encoding_list
encodeListKnown = encoding(images)
def markAttendance(name):
    with open('Attendances.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry)
        if name not in nameList:
            now = datetime.datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

cap=cv2.VideoCapture(0)
while(True):
    set,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_location=face_recognition.face_locations(frame)
    frame_encoding=face_recognition.face_encodings(frame,frame_location)
    for encodeFace,faceLoc in zip(frame_encoding,frame_location):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classname[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
        
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

    
    

