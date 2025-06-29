import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from utils.config import IMAGE_SIZE

def build_model(num_classes):
    input_shape = IMAGE_SIZE + (3,)  # e.g., (224, 224, 3)

    # Load EfficientNetB0 base model
    base_model = EfficientNetB0(include_top=False,
                                 weights='imagenet',
                                 input_shape=input_shape)

    base_model.trainable = False  # Freeze base model for transfer learning

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model
