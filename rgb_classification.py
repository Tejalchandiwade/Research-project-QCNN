
import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from pennylane.templates import RandomLayers

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
n_epochs = 30
n_layers = 1
n_train =50
n_test =30
n_qubits=12

PREPROCESS = True
np.random.seed(0)
tf.random.set_seed(0)

train_images = x_train[:n_train]
train_labels = y_train[:n_train]
test_images = x_test[:n_test]
test_labels = y_test[:n_test]

train_images = train_images / 255
test_images = test_images / 255

#train_images = np.array(train_images[..., tf.newaxis],requires_grad=False)
#test_images = np.array(test_images[..., tf.newaxis],requires_grad=False)

dev = qml.device("default.qubit", wires=12)
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))
params = 2 * np.pi * tf.random.uniform([3,12])
@qml.qnode(dev)
def circuit(data):

    for i in range(0, n_qubits):
        qml.RX(np.pi * data[i], wires=i)

    for i in range(0, n_qubits):
        qml.U1(params[0,i],wires=i)

    qml.CNOT(wires=[0, n_qubits - 1])
    for i in range(n_qubits - 1, 0, -1):
        qml.CNOT(wires=[i, i - 1])

    for i in range(0, n_qubits):
        qml.U3(params[1,i], params[0,i], params[2,i], wires=i)

    qml.CNOT(wires=[0, n_qubits - 1])
    for i in range(n_qubits - 1, 0, -1):
        qml.CNOT(wires=[i, i - 1])




    for i in range(0, n_qubits):
        qml.U1(params[1, i], wires=i)

    qml.CNOT(wires=[0, n_qubits - 1])
    for i in range(n_qubits - 1, 0, -1):
        qml.CNOT(wires=[i, i - 1])

    for i in range(0, n_qubits):
        qml.U3(params[0, i], params[1, i], params[2, i], wires=i)

    qml.CNOT(wires=[0, n_qubits - 1])
    for i in range(n_qubits - 1, 0, -1):
        qml.CNOT(wires=[i, i - 1])

    return [qml.expval(qml.PauliZ(w)) for w in range(12)]



SAVE_PATH=r"C:\Users\tejal\Documents\Quatum_op\oom"

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((16, 16, 12))
    #params = 2 * np.pi * tf.random.uniform([4])
    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 32, 2):
        for k in range(0, 32, 2):
            #for i in range(1):

            # Process a squared 2x2 region of the image with a quantum circuit
             q_results = circuit(
                [   image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0],
                    image[j, k, 1],
                    image[j, k + 1, 1],
                    image[j + 1, k, 1],
                    image[j + 1, k + 1, 1],
                    image[j, k, 2],
                    image[j, k + 1, 2],
                    image[j + 1, k, 2],
                    image[j + 1, k + 1, 2]
                ]

             )
             for i in range(12):
               out[j // 2, k // 2, i] = q_results[i]

            # Assign expectation values to different channels of the output pixel (j/2, k/2)


    return out

if PREPROCESS == True:
    q_train_images = []
    print("Quantum pre-processing of train images:")
    for idx, img in enumerate(train_images):
        print("{}/{}        ".format(idx + 1, n_train), end="\r")
        q_train_images.append(quanv(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print("\nQuantum pre-processing of test images:")
    for idx, img in enumerate(test_images):
        print("{}/{}        ".format(idx + 1, n_test), end="\r")
        q_test_images.append(quanv(img))
    q_test_images = np.asarray(q_test_images)

    # Save pre-processed images
    np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
    np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


# Load pre-processed images
q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
q_test_images = np.load(SAVE_PATH + "q_test_images.npy")




model = tf.keras.models.Sequential([#tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'),
        #tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.Conv2D(filters=20, kernel_size=(2, 2), activation='relu'),
        #tf.keras.layers.MaxPooling2D((2, 2)),


        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

opt=tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],)




from tensorflow import keras
import matplotlib.pyplot as plt
n_epochs=100


q_history = model.fit(q_train_images,train_labels,validation_data=(q_test_images, test_labels),batch_size=25,epochs=n_epochs,verbose=2)
score = model.evaluate(q_test_images, test_labels, verbose=0)


model1 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=4, kernel_size=(2,2),input_shape=(32,32,3),activation='relu'),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(2,2),activation='relu'),
        #tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.Conv2D(filters=20, kernel_size=(2, 2), activation='relu'),
        #tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(10, activation="softmax")
    ])
opt=tf.keras.optimizers.Adam(learning_rate=0.001)
model1.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],)
from tensorflow import keras
import matplotlib.pyplot as plt
n_epochs=100


c_history = model1.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    batch_size=25,
    epochs=n_epochs,
    verbose=2,
)
#score = model.evaluate(q_test_images, test_labels, verbose=0)
