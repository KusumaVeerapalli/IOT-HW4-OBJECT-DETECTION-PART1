import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def main():
    # Define constants
    BASE_DIR = "/content/drive/MyDrive/WOT_HW/images"
    IMAGE_SIZE = 100  # Adjust the image size as needed
    BATCH_SIZE = 32
    learning_rate = 0.001
    num_epochs = 10

    # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Print model summary
    model.summary()

    # Define data generators
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1. / 255)
    
    train_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        color_mode='grayscale')

    val_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        color_mode='grayscale')

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=num_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator))

    # Save the model
    model.save('model.h5')

if __name__ == "__main__":
    main()