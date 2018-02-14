from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
from scipy.misc import imsave
import os

x_path = 'img/texted_val'

version = 'v3.2'

with open('results/{0}/model_architecture_{0}.json'.format(version)) as f:
    json_model = f.read()
    model = model_from_json(json_model)
model.load_weights('results/{0}/text_remover_{0}_weights.h5'.format(version))

if not os.path.exists('results/{}_val'.format(version)):
    os.makedirs('results/{}_val'.format(version))
else:
    raise Exception('Version already validated')

augment_args = dict(rescale=1/255, data_format='channels_first')

seed = 2727
x_datagen = ImageDataGenerator(**augment_args)
x_generator = x_datagen.flow_from_directory(x_path, target_size=(360, 360), batch_size=10, class_mode=None, seed=seed)
x_img_batch = x_generator.next()
y_img_batch = model.predict_on_batch(x_img_batch)
y_img_batch = np.floor(y_img_batch*255)


def deprocess_image(x):
    # convert to RGB array
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


num = 0
for (x_img, y_img) in zip(x_img_batch, y_img_batch):
    imsave('results/{0}_val/out{1}b.png'.format(version, num), deprocess_image(y_img))
    x_img = x_img.transpose([1, 2, 0])
    imsave('results/{0}_val/out{1}a.png'.format(version, num), x_img)
    num += 1

