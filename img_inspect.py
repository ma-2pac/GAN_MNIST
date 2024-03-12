import numpy as np
import os

def read_idx3_file(filename):
    with open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        if magic_number != 2051:
            raise ValueError("Invalid magic number for IDX3 file")
        
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape((num_images, num_rows, num_cols))
        
        return images

def save_images(images, output_dir, num_images_to_save=5):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(num_images_to_save, len(images))):
        image = images[i]
        image_path = os.path.join(output_dir, f'image_{i}.png')
        plt.imsave(image_path, image, cmap='gray')
        print(f"Image {i+1} saved to: {image_path}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    input_file = 'data/train-images.idx3-ubyte'
    output_directory = 'imgs/output_images'
    num_images_to_save = 5
    
    images = read_idx3_file(input_file)
    save_images(images, output_directory, num_images_to_save)