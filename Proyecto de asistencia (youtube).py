import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendace'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images, classNames):  # Añade 'classNames' como un segundo argumento aquí
    encodeList = []
    for img, name in zip(images, classNames):  # Utiliza zip para iterar sobre ambas listas al mismo tiempo
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:
            encode = face_encodings[0]
            encodeList.append(encode)
        else:
            print(f"No se detectaron rostros en la imagen {name}.")  # Imprime el nombre de la imagen sin rostros detectados
            # Aquí puedes decidir si quieres agregar un valor nulo, continuar con la siguiente imagen, o hacer algo más.
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r') as f:  # Abrir en modo lectura para verificar los nombres existentes
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

    if name not in nameList:
        with open('Attendance.csv', 'a') as f:  # Abrir en modo 'append' para agregar información
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')  # Asegúrate de que los nombres se escriban en nuevas líneas



encodeListKnown = findEncodings(images, classNames)  # Asegúrate de pasar 'classNames' como un argumento adicional
print(len(encodeListKnown))

cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Deberías convertir imgS, no img

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)  # Debería ser imgS aquí también

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            # Escala las coordenadas de la cara de nuevo a las de la imagen original
            y1, x2, y2, x1 = [coordinate * 4 for coordinate in faceLoc]  # Multiplica todas las coordenadas por 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

