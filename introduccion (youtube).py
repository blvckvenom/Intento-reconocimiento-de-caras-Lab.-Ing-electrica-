import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carga del modelo preentrenado
base_model = InceptionResNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

# Modificación para multi-clase
data_dir = 'C:\\Users\\benit\\OneDrive\\Desktop\\Reconocimiento de caras\\Personas'
num_personas = len(os.listdir(data_dir))
predictions = Dense(num_personas, activation='softmax')(x)  # softmax para multi-clase

model = Model(inputs=base_model.input, outputs=predictions)

# Solo entrenar las capas superiores personalizadas, el resto permanecerá como está
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # categorical_crossentropy para multi-clase

batch_size = 32
img_height = 200
img_width = 200

# Usando ImageDataGenerator para leer imágenes desde directorios
train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # Escalando y definiendo split de validación

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # categorical para multi-clase
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # categorical para multi-clase
    subset='validation')

# Entrenamiento del modelo
model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // batch_size),
    epochs=10)

# Guardar el modelo para uso futuro
model.save('my_face_recognition_model.h5')

# Cargar el modelo
reloaded_model = tf.keras.models.load_model('my_face_recognition_model.h5')

# Mapeo inverso para convertir las etiquetas en nombres
class_indices = train_generator.class_indices
indices_class = {v: k for k, v in class_indices.items()}


def real_time_prediction(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('C:/Users/benit/OneDrive/Desktop/Reconocimiento de caras/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_for_model = cv2.resize(face, (200, 200))
        face_for_model = face_for_model.reshape((1, 200, 200, 3))
        face_for_model = face_for_model / 255.0

        prediction = model.predict(face_for_model)
        predicted_class = np.argmax(prediction[0])
        person_name = indices_class[predicted_class]  # obtener el nombre de la persona

        # Configurar color y etiqueta basados en la confianza de la predicción
        confidence = np.max(prediction[0])
        if confidence > 0.3:
            label = person_name
            color = (0, 255, 0)
        else:
            label = "No reconocido"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Video', frame)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    real_time_prediction(frame, reloaded_model)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
