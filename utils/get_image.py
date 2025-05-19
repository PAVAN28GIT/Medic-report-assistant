from datasets import load_dataset
import os

ds = load_dataset("itsanmolgupta/mimic-cxr-cleaned-old")



train_ds = ds['train']

example1 = train_ds[8000]

print("\n\nFinding :  ")
print(example1['findings'])

# Get the image object
img = example1['image']

# Define a filename
image_filename = "image8000.jpg"

# Save the image
img.save(image_filename)
print(f"\nImage from example saved to: {image_filename}")