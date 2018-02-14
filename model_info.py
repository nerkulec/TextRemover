from keras.models import load_model

version = 'v1.13'

model = load_model('results/{0}/text_remover_{0}.h5'.format(version))

print(model.summary())
