"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
import scipy.ndimage
import scipy.spatial
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def reduce_dimensions(feature_vectors_full, model):
    """Reduces the feature vectors down using the eigenvectors from the model.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    """Subtracting the mean of the feature vectors being classified
    as opposed to the average of the test data's feature vectors seems to
    improve performance (I think this is to do with the noise; as on clean pages
    the average for a pixel that's white in all feature vectors would be 255. In
    a noisy image, it'd be lower, so the white areas wouldn't get "centred"
    around white by subtracting 255 any more.).
    """
    return np.dot(
        (feature_vectors_full - np.mean(feature_vectors_full,axis=0)), 
        np.array(model["eigenvectors"]))


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None, train=False):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """
    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))

    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)

        """Here I've centred the characters, as I believe the covariance
        matricies will more easily pick up distinct features of characters when
        they are centrally aligned (instead of an L being in the same position
        as the right hand side of an M, it'd be in the middle, where there'd be
        a clearer distinction as the middle of an M doesn't usually extend a
        full character height, whereas an L will).
        """
        h_start = round((bbox_h/2)-(h/2))
        w_start = round((bbox_w/2)-(w/2))
        padded_image[h_start:h_start+h, w_start:w_start+w] = image[0:h, 0:w]

        #----------Denoising
        #Simple thresholding
        threshold = lambda image: np.where(image > 127, 255, 0)

        #By histographical analysis, I'm fairly certain x is 90 for page 2. 
        #Using this denoising improves page 2 significantly, but only that page.
        threshold2 = lambda image: np.where(image > 255-90, 255, image)

        #This method "stretches" all the values away from 128, which I thought
        # may be a marginally better approach than hard thresholding as it'd
        # preserve some of the "confidence" inherently expressed in the greyness
        # of each pixel.
        def stretch(image, factor=5):
            image = np.round((image-128)*factor + 128)
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            return image

        #I tried median sizes 2, 3, & 4. I found size 3 works best.
        median = lambda image: scipy.ndimage.median_filter(padded_image, size=3)

        #I found that if the median kernel is shaped vertically, it performs
        # better. I suspect this is due to the fact that a lot of characters are
        # composed of vertical lines.
        median2 = lambda image: scipy.ndimage.median_filter(image, size=(3,2))

        #I decided to try using a diamond shaped vertical footprint to squeeze
        # some extra % out, as the font doesn't tend to have square corners.
        # This brought a minor improvement over a simple kernel of size (3,2).
        padded_image = scipy.ndimage.median_filter(padded_image, 
                    footprint=np.array([[0,1,0],[1,1,1],[1,1,1],[0,1,0]]))

        #Reshaping to a column vector.
        fvectors[i, :] = padded_image.reshape(1, nfeatures)

    return fvectors

# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Performs the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print("process_training_data")
    print('\tReading data...')
    
    images_train = []
    labels_train = []

    def addNoise(images, noise_level):
        noisy_images = [image+np.random.uniform(-noise_level,+noise_level,image.size).reshape(image.shape) for image in images]
        noisy_images = [np.where(image < 0, 0, image) for image in noisy_images]
        noisy_images = [np.where(255 < image, 255, image) for image in noisy_images]
        return noisy_images

    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)

    print("\tAdding noisy data to the training set...")
    images_train = addNoise(images_train,127)+addNoise(images_train,127)
    labels_train += labels_train

    labels_train = np.array(labels_train)

    bbox_size = get_bounding_box_size(images_train)
    print('\tBounding box size:', bbox_size)

    print('\tSaving the bounding box size and labels to the model...')
    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('\tConverting images to feature vectors...')
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size, train=True)
    
    print('\tFinding eigenvectors and storing them in the model...')
    #For PCA we don't need the eigenvalues, just the eigenvectors.
    #As this is of digital text (with no lighting), the first eigenvector is
    # still useful, or at least I get worse perfomance when I discard it.
    _, model_data["eigenvectors"] = scipy.linalg.eigh(
        np.cov(fvectors_train_full, rowvar=0), 
        eigvals=((bbox_size[0]*bbox_size[1]) - 10, 
                 (bbox_size[0]*bbox_size[1]) - 1))

    model_data["eigenvectors"] = np.fliplr(model_data["eigenvectors"]).tolist()
    #I appreciate this is just bog standard PCA, but it works perfectly for the
    # first page, so spending an inordinate amount of time on improving it
    # didn't seem like the best use of time.

    print('\tReducing feature vectors to 10 dimensions and storing them in the model...')
    model_data['fvectors_train'] = reduce_dimensions(fvectors_train_full, 
                                                     model_data).tolist()

    #I tried using quadgrams in my error correction stage, but it didn't work
    # very well.
    def loadQuadgrams(model_data):
        print("\tLoading the ngrams into the model...")
        ngrams_file = open("english_quadgrams.txt","r")
        model_data['quadgrams'] = {line.split(" ")[0]:int(line.split(" ")[1]) for line in ngrams_file.readlines() if len(line.split(" "))==2}
        model_data['quadgrams_tot'] = sum([model_data['quadgrams'][k] for k in model_data['quadgrams']])
        ngrams_file.close()
        return model_data
    #model_data = loadQuadgrams(model_data)

    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    print("load_test_page")

    print("\tLoading the image feature vectors...")
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)

    print("\tPerforming dimensionality reduction...")
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)

    print("\tGetting the noise factor and calculating how many neighbours to use...")
    im = np.array(Image.open(page_name + '.png'))
    #I use the full margins (200px on every side) to sample white areas.
    margin = 200
    white_sample = np.append(im[:margin,:],im[im.shape[0]-margin:,:])
    white_sample = np.append(white_sample, 
        np.append(im[margin:im.shape[0]-margin,:margin],
                  im[margin:im.shape[0]-margin,im.shape[1]-margin:]))
    neighbours_to_use = np.array([round(np.exp(np.var(white_sample)/2250))])
    neighbours_to_use.resize((1,10))
    print("\t{0}-nearest neighbours will be used for '{1}'".format(int(neighbours_to_use[0,0]), page_name))

    #Storing info on how many neigbours to use at the end of the list of feature
    # vectors, like page meta-data. Technically sneakily adds an extra dimension
    # to each feature vector, but as it's just "one more number", I emailed
    # Jon and he permitted me to do this.
    fvectors_test_reduced = np.vstack((fvectors_test_reduced, neighbours_to_use))

    print("\tDone.\n")
    return fvectors_test_reduced


def classify_page(fvectors_test, model):
    """Uses k-nearest neighbours, where k is given by a calculation done 
    per-page and stored at the end of fvectors_test as metadata.

    parameters:

    fvectors_test - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    print("classify_page")

    print("\tLoading data from model...")
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

    #The number of neighbours to use for each page is stored at the end of the
    # numpy array of feature vectors as meta-data. Here it's removed.
    neighbours_to_use = int(fvectors_test[fvectors_test.shape[0]-1,0])
    fvectors_test = fvectors_test[:fvectors_test.shape[0]-1,:]

    print("\tCalculating distance measures from every test vector to every feature vector.")
    #I tried euclidean, it was worse. Using cosine instead.
    dists = scipy.spatial.distance.cdist(fvectors_test,fvectors_train,metric="cosine")

    print("\tPerforming {0}-nearest neighbour evaluation of the feature vectors...".format(neighbours_to_use))
    best_k = np.argsort(dists,axis=1)[:,:neighbours_to_use] #Getting the best k
    allLabels = labels_train[best_k]

    confs = np.array([[
            (uniqueLabel, sum(uniqueLabel == allLabels[i])/neighbours_to_use)
            for uniqueLabel in np.unique(allLabels[i])]
        for i in range(allLabels.shape[0])
    ])

    print("\tDone.\n")
    return confs


def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.
    
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    print("correct_errors")

    #This made my performance on the test data worse, but I've included it as
    # proof of an attempt.
    def quadgramAnalysis(labels):
        labels_pre = [labelset[0][0] for labelset in labels]

        #Do ngram analysys with confs here
        for i, label in enumerate(labels):
            #The following operation will make the character a letter, so we need to
            # check if knn thinks it is actually a letter. We just check the one
            # it's most confident about.
            if labels_pre[i].isalpha():
                neighbours = "".join([labels_pre[i+o].upper() for o in range(-4,3) if 0 < i+o and i+o < len(labels_pre)])

                for j, sublabel in enumerate(label):
                    curr_quadgrams = [list(neighbours[o:o+4]) for o in range(4)]

                    #Filtering out to just quadgrams.
                    curr_quadgrams = [qg for qg in curr_quadgrams if len(qg)==4]

                    #Replacing letters in each quadgram to match the current label.
                    for k in range(len(curr_quadgrams)):
                        curr_quadgrams[k][3-k] = sublabel[0].upper()

                    #Filtering out quadgrams that have punctuation
                    curr_quadgrams = ["".join(qg) for qg in curr_quadgrams if np.all([c.isalpha() for c in qg])]

                    if len(curr_quadgrams) > 0:
                        prob = np.mean([model["quadgrams"].get(quadgram,0)/model["quadgrams_tot"] for quadgram in curr_quadgrams])
                        labels[i][j]=(labels[i][j][0], prob * float(labels[i][j][1]))

        labels = np.array([sorted(label, key=lambda l: l[1], reverse=True) for label in labels])
        [print("{0} -> {1}".format(labels_pre[i],labels[i][0][0]),end=", ") for i in range(labels.shape[0]) if labels_pre[i] != labels[i][0][0]]
        return labels
    #labels = quadgramAnalysis(labels)

    labels = np.array([sorted(label, key=lambda l: l[1], reverse=True) for label in labels])

    print("\tCorrecting apostrophes, full stops, commas, 'l's, and 'I's.")
    for i, bbox in enumerate(bboxes):
        if str(labels[i][0][0]) in [",",".","'","l","i","I","â"]:
            curr_top = bbox[1]
            curr_bottom = bbox[3]
            avg_mid = np.mean([bboxes[i+o][1]+(bboxes[i+o][3]-bboxes[i+o][1])/2 for o in range(-3,4 if i+4<len(bboxes) else len(bboxes)-i) if abs(bboxes[i+o][1]-bbox[1])<30])
            if curr_top < avg_mid and curr_bottom < avg_mid and labels[i][0][0] in ["'","l","i","I","â"]:
                oldLabel = labels[i][0][0]
                if len([l for l in labels[i] if l[0] in [".",","]]) > 0:
                    labels[i][0] = (max([l for l in labels[i] if l[0] in [".",","]], key=lambda l: float(l[1]))[0],float(labels[i][0][1])+1)
                else:
                    labels[i][0] = (".",float(labels[i][0][1])+1)
                print("\tReplacing {0} with {1} in '{2}' (top: {3}, bottom: {4}, avg_mid: {5}, bbox:{6})".format(oldLabel, labels[i][0][0], "".join([l[0][0] for l in labels[i-5:i+6]]), curr_top, curr_bottom, avg_mid/2, bbox))
            elif curr_top > avg_mid and curr_bottom > avg_mid and labels[i][0][0] in [",",".","i","I"]:
                print("\tReplacing {0} with {1} in '{2}' (top: {3}, bottom: {4}, avg_mid: {5}, bbox:{6})".format(labels[i][0][0], "'", "".join([l[0][0] for l in labels[i-5:i+6]]), curr_top, curr_bottom, avg_mid/2, bbox))
                labels[i][0] = ("'",float(labels[i][0][1])+1)
            elif curr_top > avg_mid and curr_bottom < avg_mid and labels[i][0][0] in ["'",",","."]:
                print("\tReplacing {0} with {1} in '{2}' (top: {3}, bottom: {4}, avg_mid: {5}, bbox:{6})".format(labels[i][0][0], "l", "".join([l[0][0] for l in labels[i-5:i+6]]), curr_top, curr_bottom, avg_mid/2, bbox))
                if len([l for l in labels[i] if l[0] in ["l","i","I"]]) > 0:
                    labels[i][0] = (max([l for l in labels[i] if l[0] in ["l","i","I"]], key=lambda l: float(l[1]))[0],float(labels[i][0][1])+1)
                else:
                    labels[i][0] = ("l",float(labels[i][0][1])+1)

    #Removing conf values.
    labels = np.array([max(label, key=lambda l: l[1])[0] for label in labels])

    print("\tDone.\n")
    return labels
