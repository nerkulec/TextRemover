from keras.callbacks import Callback
from keras import backend as K
from matplotlib import pyplot as plt
from scipy.misc import imsave
import numpy as np
import os
import json


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_epoch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))

    def __len__(self):
        return len(self.losses)


class LogTests(Callback):
    def __init__(self, saver=None, num_logs=5):
        super().__init__()
        self.num_logs = num_logs
        self.saver = saver

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % (self.saver.epochs//self.num_logs) == 0 or epoch+1 == self.saver.epochs:
            self.saver.save(epoch)


class CheckpointSaver:
    def __init__(self, version='unknown', model=None, x_generator=None, y_generator=None,
                 epochs=10, mask=False, checkpoint=0):
        self.version = version
        self.model = model
        self.x_generator = x_generator
        self.y_generator = y_generator
        self.epochs = epochs
        self.mask = mask
        self.checkpoint = checkpoint

    def save(self, epoch=None):
        self.checkpoint += 1
        x_batch = self.x_generator.next()
        self.y_generator.next()
        num = 0
        if epoch+1 == self.epochs:
            path = 'results/{0}/final'.format(self.version)
        else:
            path = 'results/{0}/checkpoint_{1}'.format(self.version, self.checkpoint)
        if not os.path.exists(path):
            os.makedirs(path)
        if not self.mask:
            for (x_img, y_img) in zip(x_batch, self.model.predict_on_batch(x_batch)):
                imsave('{0}/out{1}x.png'.format(path, num), deprocess_image(x_img))
                imsave('{0}/out{1}y.png'.format(path, num), deprocess_image(y_img))
                num += 1
        else:
            for (x_img, m_img) in zip(x_batch, self.model.predict_on_batch(x_batch)):
                imsave('{0}/out{1}x.png'.format(path, num), deprocess_image(x_img))
                imsave('{0}/out{1}m.png'.format(path, num), deprocess_image(m_img))
                imsave('{0}/out{1}y.png'.format(path, num), deprocess_image(x_img-m_img))
                num += 1
        self.model.save_weights('{0}/text_remover_{1}_weights.h5'.format(path, self.version))
        with open('results/{0}/info_{0}.json'.format(self.version), 'r') as f:
            info = json.load(f)
            info['last_checkpoint'] = self.checkpoint
        with open('results/{0}/info_{0}.json'.format(self.version), 'w') as f:
            json.dump(info, f, indent=4)


def deprocess_image(img_in):
    img_in = np.floor(img_in * 255)
    # convert to RGB array
    if K.image_data_format() == 'channels_first':
        img_in = img_in.transpose((1, 2, 0))
    img_in = np.clip(img_in, 0, 255).astype('uint8')
    return img_in


def get_checkpoint(version, model=None):
    if not os.path.exists('results/{}'.format(version)):
        os.makedirs('results/{}'.format(version))
        print('New model')
        return 0
    else:
        with open('results/{0}/info_{0}.json'.format(version), 'r') as f:
            info = json.load(f)
            checkpoint = info['last_checkpoint']
        if not os.path.exists('results/{0}/checkpoint_{1}'.format(version, checkpoint)):
            os.rename('results/{0}/final'.format(version), 'results/{0}/checkpoint_{1}'.format(version, checkpoint))
        model.load_weights('results/{0}/checkpoint_{1}/text_remover_{0}_weights.h5'.format(version, checkpoint))
        print('Starting from checkpoint {}'.format(checkpoint))
        return checkpoint


def save_loss_plot(loss_history, path):
    plt.plot(range(len(loss_history)), loss_history.losses)
    plt.title('loss history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig(path, bbox_inches='tight')
