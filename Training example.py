//Training example

# Assume you have preprocessed batches:
# cover_images: (batch_size, 256, 256, 3)
# secret_images: (batch_size, 256, 256, 3)

model.fit(
    x={"cover_input": cover_images, "secret_input": secret_images},
    y={"stego_output": cover_images, "secret_reconstructed": secret_images},
    batch_size=8,
    epochs=50
)