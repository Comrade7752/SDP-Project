// Generatror

from tensorflow.keras.layers import Input, Conv2D, Concatenate, BatchNormalization
from tensorflow.keras.models import Model

def build_encoder(input_shape=(256, 256, 3)):
    cover_input = Input(shape=input_shape, name='cover_input')
    secret_input = Input(shape=input_shape, name='secret_input')

    x = Concatenate(axis=-1)([cover_input, secret_input])  # (256, 256, 6)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    stego_output = Conv2D(3, (1, 1), activation='sigmoid', padding='same', name='stego_output')(x)
    return Model(inputs=[cover_input, secret_input], outputs=stego_output, name='Encoder')