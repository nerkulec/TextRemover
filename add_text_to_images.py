from PIL import Image, ImageDraw, ImageFont
import os
import random
import textwrap

path_flow = 'H:\imgs'
path_fonts = 'C:/windows/fonts'
path_clear = 'H:/text_remover_img/clear_white/png'
path_texted = 'H:/text_remover_img/texted_white/png'


def font_gen_function():
    while True:
        for i in range(10):
            for file in os.listdir(path_fonts):
                path = os.path.join(path_fonts, file)
                try:
                    loaded_font = ImageFont.truetype(path, 32 + 12*i)
                    yield loaded_font
                except OSError as e:
                    print(path)
                    print(e)


font_gen = font_gen_function()

# load words
with open('./text/text.txt') as f:
    text = f.read()

seed = 2137
random.seed(seed)
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
            start = random.randint(0, len(text))
            txt = text[start:start+400]
            lines = textwrap.wrap(txt, width=20)
            font = next(font_gen)
            y_text = 0
            w = 360
            try:
                for line in lines:
                    width, height = font.getsize(line)
                    draw.text(((w - width) / 2, y_text), line, font=font, fill=(255, 255, 255))
                    y_text += height*1.2
            except TypeError:
                for line in lines:
                    width, height = font.getsize(line)
                    draw.text(((w - width) / 2, y_text), line, font=font)
                    y_text += height*1.2
            resized.save('{}/{}.png'.format(path_texted, num), 'PNG')
            num += 1
        except KeyError as e:
            print(e)
        except OSError as e:
            print(e)
