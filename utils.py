import numpy as np
import math
from scipy import io as spio
from keras.utils import to_categorical


def combine_images(generated_images, height=28, width=28):
    num_images = generated_images.shape[0]
    if width is None and height is None:
        width, height = int(math.sqrt(num_images)), int(
            math.ceil(float(num_images) / width)
        )
    elif width is not None and height is None:
        height = int(math.ceil(float(num_images) / width))
    elif height is not None and width is None:
        width = int(math.ceil(float(num_images) / height))

    image_height, image_width = generated_images.shape[1:3]
    combined_image = np.zeros(
        (height * image_height, width * image_width), dtype=generated_images.dtype
    )
    for index, img in enumerate(generated_images):
        row, col = divmod(index, width)
        combined_image[
            row * image_height : (row + 1) * image_height,
            col * image_width : (col + 1) * image_width,
        ] = img[:, :, 0]
    return combined_image


def load_emnist_balanced(count, classes=47):
    emnist = spio.loadmat("emnist-digits.mat")
    x_train, y_train, x_test, y_test = extract_data(emnist)

    x_train, y_train = create_balanced_dataset(x_train, y_train, count, classes)
    x_test, y_test = preprocess_data(x_test, y_test)

    return (x_train, y_train), (x_test, y_test)


def extract_data(emnist):
    x_train, y_train, x_test, y_test = (
        emnist["dataset"][0][0][0][0][0][0].astype(np.float32),
        emnist["dataset"][0][0][0][0][0][1],
        emnist["dataset"][0][0][1][0][0][0].astype(np.float32),
        emnist["dataset"][0][0][1][0][0][1],
    )
    return x_train, y_train, x_test, y_test


def create_balanced_dataset(x_data, y_data, count, classes):
    x_balanced, y_balanced = [], []
    count_per_class = [0] * classes
    for i in range(x_data.shape[0]):
        if sum(count_per_class) == classes * count:
            break
        label = int(y_data[i])
        if count_per_class[label] >= count:
            continue
        count_per_class[label] += 1
        x_balanced.append(x_data[i])
        y_balanced.append(label)

    x_balanced = np.array(x_balanced).reshape(-1, 28, 28, 1, order="A") / 255.0
    y_balanced = to_categorical(np.array(y_balanced).astype("float32"))
    return x_balanced, y_balanced


def preprocess_data(x_data, y_data):
    x_data = x_data.reshape(-1, 28, 28, 1, order="A").astype("float32") / 255.0
    y_data = to_categorical(y_data.astype("float32"))
    return x_data, y_data


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
