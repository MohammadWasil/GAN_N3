import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose, Embedding, Concatenate
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization

# define the standalone discriminator model
def define_cond_discriminator(args):
    """
    define_cond_discriminator: This function defines the architecture of the discriminator model, where we 
                               transformed the labels of attributes into the same shape dimension as image 
                               using embedding layers. And the labels are concatenated with the image 
                               as a channel. The model has 3 layers of convolutional neural networks with 
                               same numbers of filters.

    Arguments:
        args:   Parser which contains all the variables and paths.

    Returns:
        model:  It returns the architecture of the model.
    """

    n_classes = args.NUMBER_OF_CLASSES
    in_shape = (args.IMAGE_SIZE, args.IMAGE_SIZE, 3)

    # label input
    in_label = Input(shape = (1,), name = "Input_Label")
    li = Embedding( n_classes , 50, name="Embedding_D")(in_label)

    n_nodes = in_shape[0] * in_shape[1] * 1 # 128*128*1
    li = Dense(n_nodes, name="Cond_D_Dense_1")(li)
    li = Reshape((in_shape[0], in_shape[1], 1), name="Cond_D_Reshape_1")(li)

    # image input
    in_image = Input(shape=in_shape, name="Cond_D_Input_Image")

    # concat label as a channel
    merge = Concatenate(name="Cond_D_Concatenate_1")([in_image, li])

    fe = Conv2D(64, (3,3), strides=(2,2), padding='same', name="Cond_D_Conv_1")(merge)
    fe = LeakyReLU(alpha=0.2, name="Cond_D_LeakyRelu_1")(fe)

    fe = Conv2D(128, (3,3), strides=(2,2), padding='same', name="Cond_D_conv_2")(fe)
    fe = LeakyReLU(alpha=0.2, name="Cond_D_LeakyRelu_2")(fe)

    fe = Conv2D(256, (3,3), strides=(2,2), padding='same', name="Cond_D_conv_3")(fe)
    fe = LeakyReLU(alpha=0.2, name="Cond_D_LeakyRelu_3")(fe)

    fe = Conv2D(512, (3,3), strides=(2,2), padding='same', name="Cond_D_conv_4")(fe)
    fe = LeakyReLU(alpha=0.2, name="Cond_D_LeakyRelu_4")(fe)
    
    fe = Flatten(name="Cond_D_Flatten_1")(fe)
    fe = Dropout(0.3, name="Cond_D_Dropout_1")(fe)

    out_layer = Dense(1, activation='sigmoid', name="Cond_D_Dense_2")(fe)

    model = Model([in_image, in_label], out_layer)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
def define_cond_generator(latent_dim, args):
    """
    define_cond_generator: This function defines the architecture of the generator model, where we 
                          transformed the labels of attributes into the same shape dimension as image 
                          using embedding layers. And the labels are concatenated with the image 
                          as a channel. The model has 5 layers of convolutional neural networks with 
                          decreasing numbers of filters. The generator function is not trained alone, 
                          but with discriminator function.

    Arguments:
        latent_dim:  Array of random input noise of length = 100.
        args:        Parser which contains all the variables and paths.

    Returns:
        model:       It returns the architecture of the model.

    """
    n_classes = args.NUMBER_OF_CLASSES

    # label input
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 8*8*1
    li = Dense(n_nodes)(li)
    li = Reshape((8, 8, 1), name="Cond_G_Reshape_2")(li)

    # image generator input
    in_lat = Input(shape=(latent_dim,))

    n_nodes = 3*8*8    # since 3 channels
    gen = Dense(n_nodes)(in_lat)
    gen = ReLU()(gen)
    gen = Reshape((8, 8, 3), name="Cond_G_Reshape_3")(gen)

    merge = Concatenate()([gen, li])
    
    # 16x16
    gen = Conv2DTranspose(1024, (4,4), strides=(2,2), padding='same')(merge)
    gen = ReLU()(gen)
    
    # 32x32
    gen = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same')(gen)
    gen = ReLU()(gen)
    
    # 64x64
    gen = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(gen)
    gen = ReLU()(gen)
    
    # 128x128
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = ReLU()(gen)

    # 1X1 conv, reduce channels to 3 - rgb
    out_layer = Conv2D(3, (7, 7), activation='tanh', padding='same')(gen)  # or 128, 128

    model = Model([in_lat, in_label], out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_cond_gan(g_model, d_model):
    """
    define_cond_generator: This function defines the architecture of the composite model, of both discriminator function
                          and generator function. But, only the generator function trained and discriminator functio 
                          is not.

    Arguments:
        g_model:  Instance of Generator model.
        d_model:  Instance of discriminator model.
    Returns:
        model:       It returns the architecture of the model.

    """
    d_model.trainable = False

    gen_noise, gen_label = g_model.input
    gen_output = g_model.output
    
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

