import cv2
import numpy as np
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from keras.models import model_from_json
import time
import os

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def preprcess(frame):
  frame=cv2.resize(frame,(64,64))
  frame=np.array(frame)
  frame=frame/225.0
  frame=np.expand_dims(frame,axis=0)
  return frame

def preprcess1(frame):
  frame=cv2.cvtColor(frame,cv2.COLOR_RGBA2GRAY)
  frame=cv2.resize(frame,(48,48))
  frame=frame/225.0
  frame=np.expand_dims(frame,axis=0)
  return frame

def body(frame):
  results=hol.process(frame)
  mpdraw.draw_landmarks(frame,results.face_landmarks,mphol.FACEMESH_TESSELATION)
  mpdraw.draw_landmarks(frame,results.pose_landmarks,mphol.POSE_CONNECTIONS)
  mpdraw.draw_landmarks(frame,results.left_hand_landmarks,mphol.HAND_CONNECTIONS)
  mpdraw.draw_landmarks(frame,results.right_hand_landmarks,mphol.HAND_CONNECTIONS)


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image


# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.3)
detector = vision.ObjectDetector.create_from_options(options)

cap=cv2.VideoCapture(0)
gender_dict={0:'female',1:'male'}
age_dict={0:'0-18',1:'18-30',2:'30-45',3:'50+'}
et_dict={0:'white',1:'black',2:'asian',3:'Indian',4:'others'}
model=tf.keras.saving.load_model('ageet1.h5')
gen=tf.keras.saving.load_model('gender1.h5')
mphol=mp.solutions.holistic
mpdraw=mp.solutions.drawing_utils
emotion_dict = {0: "Neutral", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}
#emojis unicode
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)
# load weights into new model
model1.load_weights("model.h5")
hol=mphol.Holistic()
old=time.time()
frame=0
objects=None
gender=None
et=None
em=None
age=None
while True:

# STEP 3: Load the input image.
 ret,frame = cap.read()
 new=time.time()
 diff=new-old
 image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
 cv2.imwrite('image.jpg',image)
 # STEP 4: Detect objects in the input image.
 image = mp.Image.create_from_file('image.jpg')
 detection_result = detector.detect(image)
 if diff>2:
      harrcas=cv2.CascadeClassifier('facedetect.xml')
      rect=harrcas.detectMultiScale(frame,scaleFactor=1.2,minNeighbors=2)
      for x,y,w,h in rect:
       img=frame[y:y+h,x:x+w]
       
       img1=preprcess1(img)
       img=preprcess(img)
       res=model.predict(img)
       gen_res=gen.predict(img)[0]
       gender=gender_dict[[1 if gen_res>0.5 else 0][0]]
       age=age_dict[np.argmax(res[0][0])]
       em=emotion_dict[np.argmax(model1.predict(img1)[0])]
       et=et_dict[np.argmax(res[1][0])]

 # STEP 5: Process the detection result. In this case, visualize it.
 image_copy = np.copy(image.numpy_view())
 annotated_image = visualize(image_copy, detection_result)
 body(annotated_image)
 os.remove('image.jpg')
 cv2.putText(annotated_image,f'gender:{gender}',(0,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,225,0),1,cv2.LINE_AA)
 cv2.putText(annotated_image,f'Age:{age}',(0,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,225,0),1,cv2.LINE_AA)
 cv2.putText(annotated_image,f'Ethincity:{et}',(0,70),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,225,0),1,cv2.LINE_AA)
 cv2.putText(annotated_image,f'emotion:{em}',(0,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,225,0),1,cv2.LINE_AA)

 cv2.imshow('frame',annotated_image)
 if cv2.waitKey(5) & 0xFF==ord('q'):
   break

