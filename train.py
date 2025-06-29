from tensorflow.keras.optimizers import Adam
from models.efficientnetb0_model import build_model
from preprocessing.preprocess import preprocess_data

# Load data
train_gen, val_gen = preprocess_data()

# Get number of classes from generator
num_classes = train_gen.num_classes

# Build model
model = build_model(num_classes)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Show model summary
model.summary()
