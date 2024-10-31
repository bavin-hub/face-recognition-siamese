import os
import numpy as np
import cv2
import argparse



cwd = os.getcwd()
anchor_path = os.path.join(cwd, "data\\anchor")
positive_path = os.path.join(cwd, "data\\positive")
negative_path = os.path.join(cwd, "data\\negative")
# C:\Users\Bavin\Desktop\my_git\face-recognition-siamese\data\negative

cap = cv2.VideoCapture(0)
anchor_ctr = 0
positive_ctr = 0

print("Started collecting images for anchor and positive dataset type")
print("""Press "p" key continuously to store positive images and "a" key to store anchor images......""")
print("""Press "q" key to quit the window""")
while True:
    _, frame = cap.read()

    frame = frame[0:250, 0:250, :]
    cv2.imshow('image', frame)

    if cv2.waitKey(1) & 0XFF == ord('a'):
        anchor_ctr += 1
        img_name = f'anchor_{anchor_ctr}.jpg'
        img_path = os.path.join(anchor_path, img_name)
        cv2.imwrite(img_path, frame)

    if cv2.waitKey(1) & 0XFF == ord('p'):
        positive_ctr += 1
        img_name = f'positive_{positive_ctr}.jpg'
        img_path = os.path.join(positive_path, img_name)
        cv2.imwrite(img_path, frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()