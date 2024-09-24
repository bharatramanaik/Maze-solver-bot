import numpy as np
from PIL import Image

def generate_matrix(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    gray_image = image.convert("L")

    # Define the block size (20x20 pixels)
    block_size = 20
    num_blocks = gray_image.size[0] // block_size  # Number of blocks (21x21)

    # Initialize the matrix
    mapped_matrix = np.zeros((num_blocks, num_blocks), dtype=int)

    # Process each block
    for i in range(num_blocks):
        for j in range(num_blocks):
            # Get the coordinates of the block
            block = gray_image.crop((j * block_size, i * block_size, (j + 1) * block_size, (i + 1) * block_size))
            
            # Calculate the average color in the block
            avg_color = np.mean(block)
            
            # If the average color is closer to white (255), mark the block as 1, else 0
            mapped_matrix[i, j] = 1 if avg_color > 128 else 0

    return mapped_matrix

# Example usage
image_path = "E:/Major_project/Major Project/Major Project/Final_model/archive/3.png"  # Replace with your image path
matrix = generate_matrix(image_path)

# Print the matrix
for row in matrix:
    print(row)
