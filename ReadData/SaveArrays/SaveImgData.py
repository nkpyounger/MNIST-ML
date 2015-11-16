from pylab import imshow, show, cm
from struct import unpack
import gzip
from numpy import zeros, uint8, save, reshape
import cPickle as pickle


def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()


def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=uint8)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)
    
if __name__ == '__main__':
    print("Get testset")
    testing = get_labeled_data('t10k-images-idx3-ubyte.gz',
                               't10k-labels-idx1-ubyte.gz')
#    print(testing)
#    print(len(testing[0]))
#    print("RAWR")
    print("Got %i testing datasets." % len(testing[0]))
    testing_len = len(testing[0])
    print("Get trainingset")
    training = get_labeled_data('train-images-idx3-ubyte.gz',
                                'train-labels-idx1-ubyte.gz')
    print("Got %i training datasets." % len(training[0]))
    training_len = len(training[0]) 
    
    print("labels")
    print(testing[1][0])
    print("picture")
    print(testing[0][0])
    
    pickle.dump(testing, open('testarr.p','wb'))
    pickle.dump(training, open('trainarr.p','wb'))
    
#    testarr = testing.reshape(10000*28*28)
#    print("ARRAY")
#    print(testarr[0])
#    
#    testarr = testarr.reshape(10000, 28, 28)   
#    print("labels")
#    print(testing[1][0])
#    print("picture")
#    print(testing[0][0])
    
    
    
    
    # ValueError: could not broadcast input array from shape (10000,28,28)
    # into shape (10000)

#    f1 = open('testing_array', 'wb')
#    save(f1, testing)
#    f1.close
#    
#    f2 = open('training_array', 'wb')
#    save(f2, training)
#    f2.close
    
#    for i in range(testing_len):  
#        view_image(testing[0][i], testing[1][i])
    
    #for item in 
    