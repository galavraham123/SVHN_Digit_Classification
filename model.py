from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend
import math
from data_preprocessing import create_data_gens
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

class DigitModel:
    def __init__(self, model_save_location='model_saves/model1', initial_lr=1e-3):
        self.initial_lr = initial_lr
        self.model_location = model_save_location
        self.data = None
        self.training_history = None
        self.init_model()
    def init_model(self):
        self.model = Sequential()
        
        self.model.add(Conv2D(32, kernel_size=5, input_shape=(32, 32, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))
        
        self.model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        
        self.model.add(Conv2D(128, kernel_size=5, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(256, kernel_size=5, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(512, kernel_size=5, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())

        self.model.add(Dense(1000, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Dense(500, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=self.initial_lr), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def ready_data(self, batch_size=256):
        self.data = create_data_gens(batch_size)
    
    def train_model(self, epochs=20):
        if self.data is None:
            raise Exception('data doesnt exist')
        if self.model is None:
            raise Exception('model doesnt exist')

        def lr_scheduler(epoch):
            return self.initial_lr * math.pow(2,-epoch/10)
        train, vali = self.data
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

        callbacks = [early_stopping, LearningRateScheduler(lr_scheduler)]
        self.training_history = self.model.fit(
            train,
            epochs=epochs,
            validation_data=vali,
            callbacks=callbacks,
            shuffle=True
        )
        self.model.save(self.model_location)

    def test_model(self):
        if self.data is None:
            raise Exception('data doesnt exist')
        if self.model is None:
            raise Exception('model doesnt exist')
        _, vali = self.data
        test_loss, test_accuracy = self.model.evaluate(vali)
        print('test loss: ' + str(test_loss))
        print('test accuracy: ' + str(test_accuracy))

    def predict(self, image):
        if self.model is None:
            raise Exception('model doesnt exist')
        prediction = self.model.predict(image)
        return np.argmax(prediction)

    def show_graphs(self, plot_save_location=None):
        if self.training_history is None:
            raise Exception('training history doesnt exist')

        plot1 = plt.figure(1)
        plt.plot(self.training_history.history['accuracy'])
        plt.plot(self.training_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plot2 = plt.figure(2)
        plt.plot(self.training_history.history['loss'])
        plt.plot(self.training_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plt.show()
        
        if plot_save_location is not None:
            plt.savefig(plot_save_location)

    def switch_to_pretrained(self):
        backend.clear_session()
        self.model = load_model('model_saves/model')

def main():
    model = DigitModel()
    plot_model(model.model, to_file='model.png', show_shapes=True, show_layer_names=True)
    Image('model.png',width=400, height=200)
    # dig_model = DigitModel()
    # dig_model.ready_data()
    # dig_model.train_model(2)
    # dig_model.test_model()
    # dig_model.show_graphs()

if __name__ == '__main__':
    main()