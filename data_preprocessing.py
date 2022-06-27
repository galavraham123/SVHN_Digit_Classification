import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

TRAIN_PATH = 'data/train_32x32.mat'
TEST_PATH = 'data/test_32x32.mat'
def XY_from_mat(path):
    mat_dict = sio.loadmat(path)
    X = np.asarray(mat_dict['X'])
    X = np.asarray([X[:,:,:,i] for i in range(X.shape[3])])
    Y = mat_dict['y']
    for i in range(len(Y)):
        if Y[i] % 10 == 0:
            Y[i] = 0
    Y = to_categorical(Y, 10)

    X = X.astype('float32') # converting the arrays to Float type
    X = X / 255.0 # normalizing
    return (X,Y)

def load_train_data():
    return XY_from_mat(TRAIN_PATH)
def load_test_data():
    return XY_from_mat(TEST_PATH)

def create_data_gens(batch_size=256):
    (x_train, y_train) = load_train_data()
    (x_test, y_test) = load_test_data()

    data_gen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=[0.2,1.0]
    )

    data_gen.fit(x_train)
    train_gen = data_gen.flow(x_train, y_train, batch_size=batch_size)
    test_gen = data_gen.flow(x_test, y_test, batch_size=batch_size)
    return (train_gen, test_gen)

    
def main():
    (train, test) = create_data_gens()
    nparray, labels = next(train)
    for i in range(100):
        label = np.argmax(labels[i])
        image_arr = nparray[i]
        image_arr = image_arr.astype(int)
        print(label)
        plt.imshow(image_arr, interpolation='nearest')
        plt.show()
if __name__ == '__main__':
    main()