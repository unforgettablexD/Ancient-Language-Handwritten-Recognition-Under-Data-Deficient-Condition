import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model


def load_images(root_folder):
    images = []
    labels = []

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, filename))
                if img is not None:
                    images.append(img)
                    labels.append(folder_name)  

    return np.array(images), np.array(labels)



X_test,y_test= load_images("Tamil/test")
X_test=X_test[:,:,:,0]
X_test=np.expand_dims(X_test, axis=-1)
X_test = X_test / 255.0
y_test = y_test.astype(int)

print("Please wait.....")
# Evaluate the model on the test set
model = load_model('tamil_model.h5')
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc * 100} %")
