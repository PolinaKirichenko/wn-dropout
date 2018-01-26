import os
import subprocess

def download_mnist():
    if not os.path.exists('./data'): os.mkdir('./data')
    if not os.path.exists('./data/mnist'): os.mkdir('./data/mnist')

    base, lecun = './data/mnist', 'http://yann.lecun.com/exdb/mnist'

    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']

    print('loading mnist')

    for file in files:
        cmd = 'curl -o {base}/{file}  {lecun}/{file}'.format(base=base, lecun=lecun, file=file)
        subprocess.call(cmd.split())

    print('loading finish')
