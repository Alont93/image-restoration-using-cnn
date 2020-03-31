import numpy as np
import scipy.misc as misc
import skimage.color as color
from scipy import ndimage

import sol5_utils
# from . import sol5_utils

from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

MAX_PIXEL_VALUE = 255

GREYSCALE_REPRESENTATION = 1
RGB_REPRESENTATION = 2

CROP_RATIO = 3

CONV_FILTER_SIZE = (3,3)

TRAIN_VALID_RATIO = 0.8

BATCH_SIZE = 100
STEPS_PER_EPOCH = 100
NUM_VALID_SAMPLES = 1000

QUICK_BATCH_SIZE = 10
QUICK_STEPS_PER_EPOCH = 3
QUICK_NUM_EPOCH = 2
QUICK_NUM_VALID_SAMPLES = 30


# reads an image file and converts it into a given representation (greyscale or RGB)
def read_image(filename, representation):
    image = misc.imread(filename).astype(np.float64)

    if representation == GREYSCALE_REPRESENTATION:
        image = color.rgb2gray(image)

    return normalize(image)

# change image to [0,1] values
def normalize(image):
    image /= MAX_PIXEL_VALUE
    return image


def get_random_file(filenames, files_cache):

    num_of_files = len(filenames)
    random_file_index = np.random.randint(num_of_files)

    if random_file_index in files_cache:
        return files_cache[random_file_index]

    relpath = sol5_utils.relpath(filenames[random_file_index])
    new_file = read_image(relpath, GREYSCALE_REPRESENTATION)
    files_cache[random_file_index] = new_file
    return new_file


def crop_random(image, crop_size):
    image_y, image_x = image.shape
    crop_y, crop_x = crop_size

    start_y = np.random.randint(image_y - crop_y)
    start_x = np.random.randint(image_x - crop_x)

    return image[start_y:start_y+crop_y, start_x:start_x+crop_x]


def get_third_center_of(image):
    center_y, center_x = tuple([int(c / CROP_RATIO) for c in image.shape])
    return image[center_y:center_y+center_y, center_x:center_x+center_x]


def adjust_to_nn(image):
    new_shape = (image.shape[0], image.shape[1], 1)
    return image.reshape(new_shape) - 0.5


def load_dataset(filenames, batch_size, corruption_func, crop_size):

    files_cache = {}

    batch_shape = (batch_size, crop_size[0], crop_size[1], 1)
    source_batch = np.zeros(batch_shape)
    target_batch = np.zeros(batch_shape)

    while True:
        for i in range(batch_size):
            image = get_random_file(filenames, files_cache)
            wide_crop = tuple([c * CROP_RATIO for c in crop_size])
            image_crop = crop_random(image, wide_crop)
            original_crop_center = get_third_center_of(image_crop)

            corrupted_crop = corruption_func(image_crop)
            corrupted_crop_center = get_third_center_of(corrupted_crop)

            target_batch[i] = adjust_to_nn(original_crop_center)
            source_batch[i] = adjust_to_nn(corrupted_crop_center)

        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    last = Conv2D(num_channels, CONV_FILTER_SIZE, padding='same')(input_tensor)
    last = Activation('relu')(last)
    last = Conv2D(num_channels, CONV_FILTER_SIZE, padding='same')(last)
    last = Add()([last, input_tensor])
    last = Activation('relu')(last)
    return last


def build_nn_model(height, width, num_channels, num_res_blocks):
    input  = Input((height, width, 1))
    last = Conv2D(num_channels, CONV_FILTER_SIZE, padding='same')(input)
    last = Activation('relu')(last)

    for i in range(num_res_blocks):
        last = resblock(last, num_channels)

    last = Conv2D(1, CONV_FILTER_SIZE, padding='same')(last)
    last = Add()([last, input])

    return Model(input, last)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):

    train_size = int(len(images) * TRAIN_VALID_RATIO)
    train_images =  images[:train_size]
    valid_images = images[train_size:]

    crop_size = model.input_shape[1:3]

    model.compile (loss ='mean_squared_error', optimizer = Adam(beta_2=0.9))
    train_gen = load_dataset(train_images, batch_size, corruption_func, crop_size)
    valid_gen = load_dataset(valid_images, batch_size, corruption_func, crop_size)

    model.fit_generator(train_gen, steps_per_epoch, num_epochs,
                        validation_data=valid_gen, validation_steps=num_valid_samples)

    return model


def restore_image(corrupted_image, base_model):
    extended_image_shape = (corrupted_image.shape[0], corrupted_image.shape[1], 1)
    extended_corrupted_image = corrupted_image.reshape(extended_image_shape) - 0.5

    a = Input(extended_image_shape)
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)

    output = new_model.predict(extended_corrupted_image[np.newaxis, ...])[0,:,:,0]

    output = np.array(output, dtype=np.float64) + 0.5
    # output = output.reshape(corrupted_image.shape)
    return np.clip(output, 0, 1)


def rand_to_fraction_and_clip(image):
    image = np.round(image * MAX_PIXEL_VALUE) / MAX_PIXEL_VALUE
    return np.clip(image, 0, 1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, image.shape)

    noisy_image = image + noise

    return rand_to_fraction_and_clip(noisy_image)


DENOISE_PATCH_HEIGHT = 24
DENOISE_PATCH_WIDTH = 24
DENOISE_NUM_CHANNELS = 32
DENOISE_NUM_EPOCH = 5


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    filenames = sol5_utils.images_for_denoising()
    model = build_nn_model(DENOISE_PATCH_HEIGHT, DENOISE_PATCH_WIDTH, DENOISE_NUM_CHANNELS, num_res_blocks)
    corr_func = lambda im: add_gaussian_noise(im, 0, 0.2)

    if quick_mode:
        train_model(model, filenames, corr_func, QUICK_BATCH_SIZE, QUICK_STEPS_PER_EPOCH, QUICK_NUM_EPOCH, QUICK_NUM_VALID_SAMPLES)
    else:
        train_model(model, filenames, corr_func, BATCH_SIZE, STEPS_PER_EPOCH, DENOISE_NUM_EPOCH, NUM_VALID_SAMPLES)

    return model


def add_motion_blur(image, kernel_size, angle):
    blur_kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return ndimage.filters.convolve(image, blur_kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    kernel_size = np.random.choice(list_of_kernel_sizes)
    angle = np.random.random() * np.pi
    corrupted_image = add_motion_blur(image ,kernel_size, angle)
    return rand_to_fraction_and_clip(corrupted_image)


BLUR_PATCH_HEIGHT = 16
BLUR_PATCH_WIDTH = 16
BLUR_NUM_CHANNELS = 32
BLUR_NUM_EPOCH = 10


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    filenames = sol5_utils.images_for_deblurring()
    model = build_nn_model(BLUR_PATCH_HEIGHT, BLUR_PATCH_WIDTH, BLUR_NUM_CHANNELS, num_res_blocks)
    corr_func = lambda im: random_motion_blur(im, [7])

    if quick_mode:
        train_model(model, filenames, corr_func, QUICK_BATCH_SIZE, QUICK_STEPS_PER_EPOCH, QUICK_NUM_EPOCH, QUICK_NUM_VALID_SAMPLES)
    else:
        train_model(model, filenames, corr_func, BATCH_SIZE, STEPS_PER_EPOCH, BLUR_NUM_EPOCH, NUM_VALID_SAMPLES)

    return model