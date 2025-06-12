// End to end workflow

from tensorflow.keras.models import Model

def build_steganography_model(input_shape=(256, 256, 3)):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(input_shape)

    cover_input = Input(shape=input_shape, name='cover_input')
    secret_input = Input(shape=input_shape, name='secret_input')

    stego_output = encoder([cover_input, secret_input])
    secret_reconstructed = decoder(stego_output)

    full_model = Model(inputs=[cover_input, secret_input], outputs=[stego_output, secret_reconstructed], name='SteganographyModel')
    return full_model