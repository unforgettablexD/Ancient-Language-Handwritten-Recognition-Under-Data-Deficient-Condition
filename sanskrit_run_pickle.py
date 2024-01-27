from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

#add to increase the number of test images.
test_input = ImageDataGenerator(rescale= 1/255.0, rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest")

test_image_path = "Sanskrit/sanskrit_test_data"
test_gen = test_input.flow_from_directory(
    directory=test_image_path,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=8,
    class_mode="categorical",
)

model_path = 'sanskrit_pickle.hdf5'

print("Please wait.....")
best_model = load_model(model_path)

best_model.load_weights(model_path)
print("Loaded model from disk")
score = best_model.evaluate(test_gen, verbose=0)
print('Test accuracy percentage: ' + str(score[1] * 100) + "%")