import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# ==========================================
# SETTINGS
# ==========================================

DATASET_DIR = "dataset"

MODEL_PATH = "model/lost_item_model.h5"

IMG_SIZE = (224, 224)

BATCH_SIZE = 5


# ==========================================
# TRAINING DATA
# ==========================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)


# ==========================================
# VALIDATION DATA
# ==========================================

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)


validation_data = validation_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# ==========================================
# BASE MODEL
# ==========================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Freeze base model first

base_model.trainable = False


# ==========================================
# CUSTOM CLASSIFIER
# ==========================================

model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        train_data.num_classes,
        activation="softmax"
    )

])


# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# FIRST TRAINING
# ==========================================

print("\n================================")
print("STARTING AI TRAINING")
print("================================\n")


model.fit(

    train_data,

    validation_data=validation_data,

    epochs=15

)


# ==========================================
# FINE TUNING
# ==========================================

print("\n================================")
print("STARTING FINE TUNING")
print("================================\n")


base_model.trainable = True


# Freeze most layers

for layer in base_model.layers[:-30]:

    layer.trainable = False


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


model.fit(

    train_data,

    validation_data=validation_data,

    epochs=10

)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)


print("\nSaving AI Model...")


model.save(
    MODEL_PATH
)


# ==========================================
# SHOW RESULT
# ==========================================

print("\n================================")
print("CUSTOM AI MODEL SAVED!")
print("================================")


print(
    "Model:",
    MODEL_PATH
)


print("\nClass Names:")


for name, index in train_data.class_indices.items():

    print(
        index,
        "=",
        name
    )


print("\nTensorFlow Version:")

print(
    tf.__version__
)


print("\nTraining Completed Successfully!")