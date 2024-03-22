import tensorflow as tf

BASE_DIR = "/content/drive/MyDrive/WOT_HW/images"
MODEL_DIR="/content/model.h5"


def evaluate_model(test_dir, model_path):
    # Constants
    IMAGE_SIZE = 100  # Adjust the image size as needed
    BATCH_SIZE = 32

    # Load test data
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale')

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model on the test data
    evaluation = model.evaluate(test_generator)

    # Print evaluation metrics
    print("Test Loss:", evaluation[0])
    print("Test Accuracy:", evaluation[1])

    # Return evaluation metrics
    return evaluation

evaluate_model(BASE_DIR,MODEL_DIR)