// Discriminator

def build_decoder(input_shape=(256, 256, 3)):
    stego_input = Input(shape=input_shape, name='stego_input')

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(stego_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    secret_reconstructed = Conv2D(3, (1, 1), activation='sigmoid', padding='same', name='secret_reconstructed')(x)
    return Model(inputs=stego_input, outputs=secret_reconstructed, name='Decoder')