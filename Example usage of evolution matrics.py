// Example usage of evolution matrics 

results = evaluate_steganography(
    cover_images=test_cover_images,
    stego_images=model.predict([test_cover_images, test_secret_images])[0],
    secret_images=test_secret_images,
    recovered_secrets=model.predict([test_cover_images, test_secret_images])[1]
)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")