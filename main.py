import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn import neighbors
from scipy.stats import mode


K = 1


# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']
cosine_loss = tf.keras.losses.CosineSimilarity(axis=0)
neigh = neighbors.KNeighborsClassifier(n_neighbors=1)


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def get_vector(path):
    image = tf.io.read_file(path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

    # Run model inference.
    outputs = movenet(image)

    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']
    new_img = keypoints[:, :, :, :2].numpy().squeeze()
    y_values, x_values = tf.unstack(new_img, axis=1)
    return tf.reshape(tf.stack([x_values, y_values], axis=1), [-1, 1])
    # return new_img


# get vector for new image
new_path = 'test.jpg'
new_path = get_vector(new_path)

# compare to others
features = []
labels = []
for index, p in enumerate(['pose.jpg', 'pose2.jpg', 'pose3.jpg']):
    img = get_vector(p)
    features.append(img.numpy().flatten())
    labels.append(index)

features = np.array(features)
labels = np.array(labels)

knn = neigh.fit(features, labels)
print(knn.predict([new_path.numpy().flatten()]))


if __name__ == '__main__':
    print('PyCharm')

