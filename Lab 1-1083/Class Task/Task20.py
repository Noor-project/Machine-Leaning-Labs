from sklearn.datasets import fetch_olivetti_faces

# fetch the faces data
faces = fetch_olivetti_faces()
faces.keys()
n_samples, n_features = faces.data.shape
print((n_samples, n_features))
print(faces.images.shape)
print(faces.data.shape)
print(faces.DESCR)
X, y = faces.data, faces.target
# set up the figure
import matplotlib.pyplot as plt

# Show one image (5th index)
plt.imshow(faces.images[5], cmap="gray")
plt.title(f"Face ID: {faces.target[5]}")
plt.axis("off")  # hide axis
plt.show()

# Show first 16 images
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(faces.images[i], cmap="gray")
    plt.axis("off")  # hide axes
    plt.title(faces.target[i])
plt.show()

