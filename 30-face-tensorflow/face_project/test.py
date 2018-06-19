import FaceToolKit as ftk
from scipy import misc
import numpy as np

verification_threshhold = 1.188
image_tensor_size = 160


# Class instantiations
v = ftk.Verification()

# Pre-load model for Verification
v.load_model("./models/20180204-160909/")
v.initial_input_output_tensors()

#read images
img1 = misc.imread("./images/1.jpg")
img2 = misc.imread("./images/2.jpg")
img3 = misc.imread("./images/3.jpg")

#generate embeddings
emb1 = v.img_to_encoding(img1, image_tensor_size)
emb2 = v.img_to_encoding(img2, image_tensor_size)
emb3 = v.img_to_encoding(img3, image_tensor_size)

#distance
diff = np.subtract(emb1, emb2)
dist = np.sum(np.square(diff))
is_same = dist < verification_threshhold
print ("distance img1 and img2 =", dist, " is+same =", is_same)

diff = np.subtract(emb1, emb3)
dist = np.sum(np.square(diff))
is_same = dist < verification_threshhold
print ("distance img1 and img3 =", dist, " issame =", is_same)