from skimage import io
from skimage.morphology import skeletonize, dilation, square
from glob import glob
import numpy as np
import matplotlib.pyplot as pyplot

if __name__ == '__main__':
    for im_name in glob('*.png'):
        if 'temp_6' not in im_name: continue
        im = io.imread(im_name)
        im = np.array(im[:,:, 0:3], dtype='uint8')
        data = np.absolute(im - [255, 255, 0])
        
        sums = np.sum(data, axis=2)
        
        morf = sums < 210
        morf = dilation(morf, selem=square(5))
        morf = skeletonize(morf)

        threshold = np.where(morf == 1)
        pixs = threshold[0].shape[0]
        sumx = np.sum(threshold[0])

        magic = (sumx / pixs / 128 - 0.5) 

        print(pixs, sumx, magic)



        pyplot.imshow(morf, cmap='gray')
        pyplot.show()

        break