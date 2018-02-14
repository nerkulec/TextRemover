from PIL import Image, ImageDraw, ImageFont
import os
import random
import re

path_flow = 'H:\imgs'
path_in = './img/source_val/'
path_fonts = 'C:/windows/fonts'
path_clear = 'H:/text_remover_img/clear/png'
path_texted = 'H:/text_remover_img/texted/png'


def font_gen_function():
    while True:
        for i in range(6):
            for file in os.listdir(path_fonts):
                path = os.path.join(path_fonts, file)
                try:
                    loaded_font = ImageFont.truetype(path, 32 + 8*i)
                    yield loaded_font
                except OSError as e:
                    print(path)
                    print(e)


font_gen = font_gen_function()

# load words
with open('./text/text.txt') as f:
    text = f.read()
    words = re.split('[ ,\n;"]', text)

seed = 2137
random.seed(seed)
word_index = 0
num = 0
for (dirpath, dirnames, filenames) in os.walk(path_flow):
    for filename in filenames:
        path_img = os.path.join(dirpath, filename)
        image = Image.open(path_img)
        resized = image.resize((360, 360))
        try:
            resized.save('{}/{}.png'.format(path_clear, num), 'PNG')
            draw = ImageDraw.Draw(resized)
            height = 0
            for j in range(6):
                n_words = random.randint(3, 6)
                font = next(font_gen)
                text = ' '.join(words[word_index:word_index+n_words])
                w, h = draw.textsize(text, font=font)
                height += random.gauss(50, 10)
                pos = (resized.width/2 - w/2, resized.height - h - height)
                word_index += n_words
                word_index %= len(words)
                fill = (random.randint(0, 256), random.randint(0, 256),
                        random.randint(0, 256))
                try:
                    draw.text(xy=pos, text=text, fill=fill, font=font)
                except TypeError as e:
                    print(e)
                    draw.text(xy=pos, text=text, font=font)
            resized.save('{}/{}.png'.format(path_texted, num), 'PNG')
            num += 1
        except KeyError as e:
            print(e)
        except OSError as e:
            print(e)

