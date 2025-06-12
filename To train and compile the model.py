// To train and compile the model

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

model = build_steganography_model()

# Losses: stego should match cover, reconstructed secret should match original
mse = MeanSquaredError()
lambda_cover = 1.0
lambda_secret = 1.0

def custom_loss(y_true, y_pred):
    return mse(y_true, y_pred)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        "stego_output": lambda y_true, y_pred: lambda_cover * mse(y_true, y_pred),
        "secret_reconstructed": lambda y_true, y_pred: lambda_secret * mse(y_true, y_pred)
    }
)