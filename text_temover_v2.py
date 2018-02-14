from keras.models import Model
from keras.layers import Input, Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from TextRemover.util import get_checkpoint, save_loss_plot, LossHistory, LogTests, CheckpointSaver
import json

x_path = r'H:\text_remover_img\texted'
y_path = r'H:\text_remover_img\clear'

version = 'v4.1'
comment = 'next test Subtract()'

steps_per_epoch = 100
epochs = 60
num_logs = 5
lr = 0.0001


# augment_args = dict(width_shift_range=5, height_shift_range=3,
#                     zoom_range=0.1, rescale=1/255, data_format='channels_first')


seed = 2137
augment_args = dict(rescale=1/255, data_format='channels_first')

x_datagen = ImageDataGenerator(**augment_args)
y_datagen = ImageDataGenerator(**augment_args)

x_generator = x_datagen.flow_from_directory(x_path, target_size=(360, 360), batch_size=10, class_mode=None, seed=seed)
y_generator = y_datagen.flow_from_directory(y_path, target_size=(360, 360), batch_size=10, class_mode=None, seed=seed)


def mask_generator_function():
    for (x_b, y_b) in zip(x_generator, y_generator):
        m_b = x_b - y_b
        yield x_b, m_b


mask_generator = mask_generator_function()

img = Input(shape=(3, 360, 360), name='img')

x = Conv2D(32, (3, 3), padding='same', activation='relu')(img)
x = Conv2D(48, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(3, (3, 3), padding='same')(x)

model = Model(inputs=[img], outputs=[x])
model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy")

checkpoint = get_checkpoint(version, model=model)

if checkpoint == 0:
    info = {'version': version,
            'comment': comment,
            'steps_per_epoch': steps_per_epoch,
            'epochs': epochs,
            'optimizer': type(model.optimizer).__name__,
            'loss': str(model.loss),
            'augment': augment_args,
            'last_checkpoint': 0}
    with open('results/{0}/info_{0}.json'.format(version), 'w') as f:
        json.dump(info, f, indent=4)
    with open('results/{0}/model_architecture_{0}.json'.format(version), 'w') as f:
        f.write(model.to_json())

loss_history = LossHistory()
saver = CheckpointSaver(version=version, model=model, x_generator=x_generator,
                        y_generator=y_generator, epochs=epochs, mask=True)
log_tests = LogTests(saver=saver, num_logs=10)
model.fit_generator(mask_generator, callbacks=[loss_history, log_tests],
                    steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2)

save_loss_plot(loss_history, 'results/{0}/loss_history_{0}_cp{1}.png'.format(version, checkpoint))
