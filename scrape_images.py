import urllib.request
import os


def download_images():
    path = 'img/buildings'
    link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021939'
    urls = urllib.request.urlopen(link).read().decode()

    if not os.path.exists(path):
        os.makedirs(path)

    index = 0
    for img_url in urls.split('\n'):
        try:
            print(img_url)
            urllib.request.urlretrieve(img_url, path + '/' + str(index) + ".jpg")

            index += 1

        except Exception as e:
            print(str(e))


download_images()
