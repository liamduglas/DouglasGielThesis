import time
import numpy as np
import sklearn.metrics
import tensorflow as tf
from keras import Model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import Data_Driver_Final as id
from PlotLearning import PlotLearning
import matplotlib

callback = [PlotLearning()]

x = id.images_x
y = id.images_y

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
num_classes = 1
def UNet(inputs):
    print("Shape inputs: ", str(inputs.shape))
    c1 = tf.keras.layers.Conv2D(16, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(16, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    print("Shape c1: " + str(c1.shape))
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    b1 = tf.keras.layers.BatchNormalization()(p1)

    print("Shape b1: " + str(b1.shape))
    c2 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(b1)
    c2 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    print("Shape c2: " + str(c2.shape))
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    b2 = tf.keras.layers.BatchNormalization()(p2)

    print("Shape b1: " + str(b2.shape))
    c3 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(b2)
    c3 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    print("Shape c3: " + str(c3.shape))
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    b3 = tf.keras.layers.BatchNormalization()(p3)

    print("Shape b3: " + str(b3.shape))
    c4 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(b3)
    c4 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    print("Shape c4: " + str(c4.shape))
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    b4 = tf.keras.layers.BatchNormalization()(p4)

    print("Shape b4: " + str(b4.shape))
    c5 = tf.keras.layers.Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(b4)
    c5 = tf.keras.layers.Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    print("Shape c5: " + str(c5.shape))

    u1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    cc1 = tf.keras.layers.concatenate([u1, c4], axis=3)
    b5 = tf.keras.layers.BatchNormalization()(cc1)
    print("Shape b5: " + str(b5.shape))
    c6 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(b5)
    c6 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    print("Shape c6: " + str(c6.shape))

    u2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    cc2 = tf.keras.layers.concatenate([u2, c3], axis=3)
    b6 = tf.keras.layers.BatchNormalization()(cc2)
    print("Shape b6: " + str(b6.shape))
    c7 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(b6)
    c7 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    print("Shape c7: " + str(c7.shape))

    u3 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    cc3 = tf.keras.layers.concatenate([u3, c2], axis=3)
    b7 = tf.keras.layers.BatchNormalization()(cc3)
    print("Shape b7: " + str(b7.shape))
    c8 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(b7)
    c8 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    print("Shape c8: " + str(c8.shape))

    u4 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    cc4 = tf.keras.layers.concatenate([u4, c1], axis=3)
    b8 = tf.keras.layers.BatchNormalization()(cc4)
    print("Shape b8: " + str(b8.shape))
    c9 = tf.keras.layers.Conv2D(16, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(b8)
    c9 = tf.keras.layers.Conv2D(16, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    print("Shape c9: " + str(c9.shape))

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    print("Shape out: " + str(outputs.shape))
    return outputs

inputs = tf.keras.layers.Input(shape=(128, 128, 3))
outputs = UNet(inputs)
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=["accuracy"])

x = x.squeeze()
y = y.squeeze()

x = x[0:400]
y = y[0:400]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=80)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/9, random_state=80 )
print(y_train.shape)
print(x_train.shape)

#model.fit(x_train, y_train, epochs=250, batch_size=32, validation_data=(x_val, y_val), use_multiprocessing=True, callbacks=[callback])
#model.save_weights('final_weights_1channel/weights')

model.load_weights('final_weights_1channel/weights')
y_pred = model.predict(x_test)

dims = y_pred.shape
rows = dims[1]
col = dims[2]
for k in range(0, dims[0]):
    sum = 0
    for i in range(0, rows):
        for j in range(0, col):
            y_pred[k, i, j] = np.round(y_pred[k, i, j])

show_images = True

if show_images:
    cmap = matplotlib.colors.ListedColormap(['green', 'white'])
    for i in range(0, int(20)):
        plt.imshow(x_test[i], interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        plt.show()
        plt.imshow(y_test[i][:, :], interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        plt.show()
        plt.imshow(y_pred[i][:, :], interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        plt.show()
        #time.sleep(5)

metric_y_test = y_test[:, :, :].flatten()
metric_y_pred = y_pred[:, :, :].flatten()
print(sklearn.metrics.f1_score(y_test[:, :, :].flatten(), y_pred[:, :, :].flatten()))
print(sklearn.metrics.confusion_matrix(y_test[:, :, :].flatten(), y_pred[:, :, :].flatten()))
print(sklearn.metrics.jaccard_score(metric_y_test, metric_y_pred))

tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
