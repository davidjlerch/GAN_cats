from keras.models import Sequential
from keras.layers import Dense, Conv2D, Add, UpSampling2D, Flatten, MaxPooling2D, Conv2DTranspose, concatenate
from keras.layers import Reshape, Input, Dropout
from keras import optimizers, Model, callbacks, regularizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
from keras.models import load_model
from PIL import Image
from keras.utils import plot_model


def gen_model():
    """
    If no generator exists, a new generator is built.
    :return: Model of the generator
    """
    try:
        gen = load_model('generator_.h5')
    except Exception as e:
        print(e)
        drop_gen = 0.0
        noise = Input(shape=(299, 299, 3))

        x = MaxPooling2D(13)(noise)
        x = Flatten()(x)
        x = Dense(1587, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
        img_norm = Reshape((23, 23, 3))(x)
        x11 = Conv2D(256, 1, name='conv11', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(img_norm)
        x12 = Conv2D(256, 3, name='conv12', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(img_norm)
        x13 = Conv2D(256, 5, name='conv13', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(img_norm)
        x1 = Add(name="add1")([x11, x12, x13])

        t11 = Conv2DTranspose(128, 1, name='convt11', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x1)
        t12 = Conv2DTranspose(128, 3, name='convt12', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x1)
        t13 = Conv2DTranspose(128, 5, name='convt13', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x1)
        t11 = Dropout(drop_gen)(t11)
        t12 = Dropout(drop_gen)(t12)
        t13 = Dropout(drop_gen)(t13)
        t1 = Add(name="add5")([t11, t12, t13])

        t21 = Conv2DTranspose(64, 1, name='convt21', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t1)
        t22 = Conv2DTranspose(64, 3, name='convt22', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t1)
        t23 = Conv2DTranspose(64, 5, name='convt23', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t1)
        t21 = Dropout(drop_gen)(t21)
        t22 = Dropout(drop_gen)(t22)
        t23 = Dropout(drop_gen)(t23)
        t2 = Add(name="add6")([t21, t22, t23])
        t2 = UpSampling2D(13)(t2)

        t31 = Conv2DTranspose(32, 1, name='convt31', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t2)
        t32 = Conv2DTranspose(32, 3, name='convt32', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t2)
        t33 = Conv2DTranspose(32, 5, name='convt33', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t2)
        t3 = Add(name="add7")([t31, t32, t33])

        t41 = Conv2DTranspose(16, 1, name='convt41', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t3)
        t42 = Conv2DTranspose(16, 3, name='convt42', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t3)
        t43 = Conv2DTranspose(16, 5, name='convt43', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t3)
        t4 = Add(name="add8")([t41, t42, t43])

        out = Conv2DTranspose(3, 1, name='convt51', padding='same', activation='tanh', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(t4)

        gen = Model(inputs=noise, outputs=out, name="generator")
        gen.summary()

    classifier = load_model('classifier_.h5')

    for layer in classifier.layers:
        layer.trainable = False

    classifier = Sequential()
    classifier.add(gen)
    classifier.add(classifier)
    plot_model(classifier, to_file='model.png')
    classifier = Model(inputs=classifier.input, outputs=classifier.layers[-1].get_output_at(1), name="Model")
    classifier.summary()
    return classifier, gen


def class_model(classifier_dropout1=0.0, classifier_dropout2=0.0):
    """
    This function creates the classification model.
    :return: Model of the classifier
    """

    img = Input(shape=(299, 299, 3))
    x11 = Conv2D(16, 1, name='conv11', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(img)

    x12 = Conv2D(16, 3, name='conv12', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(img)

    x13 = Conv2D(16, 5, name='conv13', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(img)

    x14 = MaxPooling2D(13, padding='same')(img)
    x14 = UpSampling2D(13)(x14)

    x11 = Dropout(classifier_dropout1)(x11)
    x12 = Dropout(classifier_dropout1)(x12)
    x13 = Dropout(classifier_dropout1)(x13)

    x1 = concatenate([x11, x12, x13, x14])
    x1 = MaxPooling2D(3)(x1)

    x21 = Conv2D(32, 1, name='conv21', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x1)

    x22 = Conv2D(32, 3, name='conv22', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x1)

    x23 = Conv2D(32, 5, name='conv23', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x1)

    x24 = MaxPooling2D(3, padding='same')(x1)
    x24 = UpSampling2D(3)(x24)

    x21 = Dropout(classifier_dropout1)(x21)
    x22 = Dropout(classifier_dropout1)(x22)
    x23 = Dropout(classifier_dropout1)(x23)

    x2 = concatenate([x21, x22, x23, x24])
    x2 = MaxPooling2D(3)(x2)

    x31 = Conv2D(64, 1, name='conv31', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x2)

    x32 = Conv2D(64, 3, name='conv32', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x2)

    x33 = Conv2D(64, 5, name='conv33', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x2)

    x34 = MaxPooling2D(3, padding='same')(x2)
    x34 = UpSampling2D(3)(x34)

    x31 = Dropout(classifier_dropout1)(x31)
    x32 = Dropout(classifier_dropout1)(x32)
    x33 = Dropout(classifier_dropout1)(x33)
    x34 = Dropout(classifier_dropout1)(x34)

    x3 = concatenate([x31, x32, x33, x34])
    x3 = MaxPooling2D(3)(x3)

    x41 = Conv2D(128, 1, name='conv41', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x3)

    x42 = Conv2D(128, 3, name='conv42', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x3)

    x43 = Conv2D(128, 5, name='conv43', padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x3)

    x44 = MaxPooling2D(11, padding='same')(x3)
    x44 = UpSampling2D(11)(x44)

    x41 = Dropout(classifier_dropout1)(x41)
    x42 = Dropout(classifier_dropout1)(x42)
    x43 = Dropout(classifier_dropout1)(x43)

    x4 = concatenate([x41, x42, x43, x44])
    x4 = MaxPooling2D(4)(x4)

    x = Flatten()(x4)
    x = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
    x = Dropout(classifier_dropout2)(x)
    x = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
    x = Dropout(classifier_dropout2)(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)

    classifier = Model(img, out, name="classifier")
    classifier.summary()

    return classifier


def noise_gen():
    noise = []
    dim = 299
    for r in range(dim):
        for c in range(dim):
            for f in range(3):
                noise.append(random.randint(0, 255))

    noise = np.reshape(np.asarray(noise), (1, dim, dim, 3))
    noise = noise/255.
    return np.asarray(noise)


def compiler_generator(model):
    adamax = optimizers.Adamax(learning_rate=0.1, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=adamax)
    
    
def compiler_classifier(model):
    adam = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=adam)


def training_gen(model, generator_steps, num_epochs=50):
    my_callback = callbacks.callbacks.EarlyStopping(monitor='loss', min_delta=0.0, patience=1, verbose=2)
    generator_history = model.fit_generator(generator_generator(), steps_per_epoch=generator_steps, epochs=num_epochs, shuffle=True, callbacks=[my_callback], verbose=2)
    generator_loss = generator_history.history
    return generator_loss


def training_classifier(model, classifier_steps, num_epochs=50):
    my_callback = callbacks.callbacks.EarlyStopping(monitor='loss', min_delta=0.0, patience=0, verbose=2)
    classifier_history = model.fit_generator(classifier_generator(), steps_per_epoch=classifier_steps, epochs=num_epochs, shuffle=True, callbacks=[my_callback], verbose=2)
    classifier_loss = classifier_history.history
    return classifier_loss


def generator_generator(batch_size=4):
    """
    This generator provides a random image for the generator. The label is always 1 which means it's ground truth is "real image".
    However for the classifier the label of generated images is 0 which means "generated image".
    :return: Batches of generated images
    """
    while True:
        image_array = []
        label_array = []
        for i in range(batch_size):
            image = np.asarray(noise_gen())
            image_array.append(np.reshape(image, (299, 299, 3)))
            label = 1
            label_array.append(label)
        yield [np.asarray(image_array), np.asarray(label_array)]


def classifier_generator(batch_size=10):
    """
    This generator provides data for the classifier. Generated images get label 0, cat images get label 1.
    :param batch_size:
    :return:
    """
    gen = ImageDataGenerator(rescale=1/255., vertical_flip=True)
    train_data = gen.flow_from_directory('Cats/', target_size=(299, 299), shuffle=True, class_mode='binary', batch_size=1)
    try:
        generator_model = load_model('generator_.h5')
    except Exception as e:
        print(e)
        generator_model = None
    while True:
        image_array = []
        label_array = []
        for i in range(batch_size):
            count = random.randint(1, 6)
            if count % 2 == 0:
                image = train_data.next()[0]
                image = np.reshape(image, (299, 299, 3))
                label = 1
            else:
                if generator_model is not None:
                    generated = generator_model.predict(noise_gen())
                    image = np.asarray(generated)
                else:
                    image = np.asarray(noise_gen())
                image = np.reshape(image, (299, 299, 3))
                label = 0
            image_array.append(image)
            label_array.append(label)
        yield [np.asarray(image_array), np.asarray(label_array)]


def train(generator_steps=100, classifier_steps=80, limit_training=-1, num_epochs_classifier=50, num_epochs_generator=50):
    """
    The training is executed in several steps. First a classifier is loaded or trained if it doesn't exist.
    Afterwards the generator is trained using the now existing classifier.
    Furthermore there are generated images saved every 10 epochs to see the current progress of the generator.
    :param generator_steps: The amount of steps per epoch in training.
    :param classifier_steps: The amount of steps per epoch in training.
    :param limit_training: Limit training if desired. Default is unlimited training time.
    :param num_epochs_classifier: The number of epochs performed for the classifier.
    :param num_epochs_generator: The number of epochs performed for the generator.
    :return:
    """
    decade = 0
    while True:
        for z in range(10):
            try:
                classifier = load_model('classifier_.h5')
            except Exception as e:
                print(e)
                classifier = class_model()
            compiler_classifier(classifier)
            training_classifier(classifier, classifier_steps, num_epochs=num_epochs_classifier)
            classifier.save("classifier_.h5")
            model, gen = gen_model()
            compiler_classifier(model)
            compiler_generator(gen)
            training_gen(model, generator_steps, num_epochs=num_epochs_generator)
            gen.save("generator_.h5")
            model.save("model_.h5")
        generator = load_model('generator_.h5')
        for i in range(10):
            img = generator.predict(noise_gen())*255
            arr = np.around(np.asarray(np.reshape(img, (299, 299, 3))))
            img = Image.fromarray(arr.astype('uint8'), 'RGB')
            img.save('gen/' + str(i) + 'img.png')
        if decade == limit_training:
            return
        decade += 1


def create(num_images):
    """
    Generates some images, created by the latest generator trained.
    :param num_images: amount of images that should be created
    :return:
    """
    generator_model = load_model('generator_.h5')
    for i in range(num_images):
        img = generator_model.predict(noise_gen())
        img = img/np.amax(img)*255
        arr = np.around(np.asarray(np.reshape(img, (299, 299, 3))))
        img = Image.fromarray(arr.astype('uint8'), 'RGB')
        img.save('gen/cat_' + str(i) + '.png')


if __name__ == '__main__':
    train()
    # create(20)
