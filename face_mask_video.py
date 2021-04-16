import cv2
import tensorflow as tf
import numpy as np

def preprocessing(img):
    img = cv2.resize(img, (150,150))
    img = img/255.0
    img=np.reshape(img,(1,150,150,3))
    return img

model = tf.keras.models.load_model('C:\\Users\\ASUS\\PycharmProjects\\College\\face_mask_model.h5')
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()

    pred = model.predict(preprocessing(img))
    final_pred = np.round(pred)[0]

    if final_pred[0] == 1:
        cv2.putText(img, "No Mask", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Mask", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Mask Detection", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()