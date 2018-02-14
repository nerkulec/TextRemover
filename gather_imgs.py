import os

for (dirpath, dirnames, filenames) in os.walk('H:\imgs'):
    for filename in filenames:
        path = os.path.join(dirpath, filename)
        if path[-6:] == '.gstmp':
            os.remove(path)
