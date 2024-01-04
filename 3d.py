import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have a 3D NumPy array
# Replace this with your actual data
data = np.load('image_9.npy')

# Choose the slice index along the third axis (Z-axis)
slice_index = 64

# Select the slice along the Z-axis
rgb_slice = data[:, :, slice_index, :]

# Display the RGB slice using imshow
plt.imshow(rgb_slice)
plt.title(f'3D Image Slice (RGB) - Slice {slice_index}')
plt.show()


