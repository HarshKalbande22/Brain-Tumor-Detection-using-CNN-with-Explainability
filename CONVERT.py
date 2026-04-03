from tensorflow.keras.models import load_model

# Load old model
model = load_model("vgg19_brain_tumor_95acc.h5", compile=False)

# Save in new format
model.save("model.keras")

print("Model converted successfully!")