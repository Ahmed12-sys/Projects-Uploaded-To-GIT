import cv2 # computer vision library
import helpers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
help(helpers)
# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_training_night = "day_night_images/training/night/"
image_dir_test = "day_night_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(image_dir_training)
IMAGE_LIST_NIGHT=helpers.load_dataset(image_dir_training_night)
# Select an image and its label by list index
image_index = 0
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]
## TODO: Print out 1. The shape of the image and 2. The image's label `selected_label`
# =============================================================================
# plt.imshow(selected_image)
# print("Image Dimension: ",selected_image.shape)
# print("Day[1] or night[0]: ",str(selected_label))
# print("Matrix Data: ",str(IMAGE_LIST[image_index][0]))
# =============================================================================

## TODO: Display a night image
# =============================================================================
# selected_image = IMAGE_LIST[120][0]
# plt.imshow(selected_image)
# print("Image Dimension: ",selected_image.shape)
# print("Day[1] or night[0]: ",str(IMAGE_LIST[120][1]))
# print("Matrix Data: ",str(IMAGE_LIST[120][0]))
# =============================================================================
"""
for image_index in range(len(IMAGE_LIST)):
    print("Image Number",image_index,"Image Data",selected_image.shape)
    selected_image = IMAGE_LIST[image_index][0]
    selected_label = IMAGE_LIST[image_index][0]
    plt.imshow(selected_image)
    print("Day[1] or night[0]: ",str(IMAGE_LIST[image_index][1]))
    print("Matrix Data: ",str(IMAGE_LIST[image_index][0]))
"""
"""
ERROR
## TODO: Display a night image
night_image =IMAGE_LIST_NIGHT[image_index][0]
night_label=IMAGE_LIST_NIGHT[image_index][1]
plt.imshow(night_image)
print("Image Dimension: ",night_image.shape)
print("Day[1] or night[0]: ",str(night_label))
# Note the differences between the day and night images
# Any measurable differences can be used to classify these images
"""
def standerized_input(image):
    standarized_image= cv2.resize(image,(600,1100))
    return standarized_image

#Day=1 , Night =0
def integer_encoding(label):
    if label=="day":
        return 1
    else:
        return 0

def standardize(image_list):
    standard_list=[]

#loop through every item in the image_list    
    for item in image_list:
        image = item[0]
        label = item[1]

        std_image=standerized_input(image)
        binary_label=integer_encoding(label)
        standard_list.append((std_image,binary_label))
    return standard_list
"""
multiple temporary variable
for image,label in image_list:
    image=item
    label = item[1]
"""
STANDARIZED_LIST= standardize(IMAGE_LIST)
image_number=100
std_image=STANDARIZED_LIST[image_number][0]
std_label=STANDARIZED_LIST[image_number][1]
# =============================================================================
# plt.imshow(std_image)
# print("Image Dimension: ",std_image.shape)
# print("Day[1] or night[0]: ",str(std_label))
# =============================================================================

###Average Brightness Feature###

def average_brightness(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    brightness=np.sum(hsv[:,:,2])
    return brightness/(1100*600)
#print("Average brightness is: ",average_brightness(std_image))
sum_night=0
sum_day=0
for i in range(119):
    sum_day+=average_brightness(STANDARIZED_LIST[i][0])
sum_day/=120    
print("the average brightness of all Day images is: ",sum_day)
    
for i in range(120,239,1):
    sum_night+=average_brightness(STANDARIZED_LIST[i][0])
sum_night/=120    
print("the average brightness of all Night images is: ",sum_night)

sum_avg=(sum_night+sum_day)/(2)
print(sum_avg)
###the sum_avg is 102

def estimate_label(rgb_image):
    threshold =102
    if(average_brightness(rgb_image))>threshold:
        predicted_label=1
    else:
        predicted_label=0

    return predicted_label


print(estimate_label(STANDARIZED_LIST[2][0]))
###printing the whole test
# =============================================================================
# for image_index in range(len(STANDARIZED_LIST)):
#     print("Image Number",image_index)
#     selected_image = IMAGE_LIST[image_index][0]
#     selected_label = IMAGE_LIST[image_index][0]
#     print("Day[1] or night[0]: ",str(IMAGE_LIST[image_index][1]))
#     print("estimated_label: ",estimate_label(selected_image))
# =============================================================================
# =============================================================================
# def get_misclassified_images(std_list):
#     #image,predicted, true
#     misclassified_images_labels=[]
#     
#     for i in range(len(std_list)):
#         actual_label =std_list[i][1]
#         predicted_label = estimate_label(std_list[i][1])
#         
#         if actual_label != predicted_label:
#             misclassified_images_labels.append(std_list[i][0],actual_label,predicted_label)
#             
#     return misclassified_images_labels
#     
# =============================================================================
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels



misclassified_images=get_misclassified_images(STANDARIZED_LIST)

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARIZED_LIST)

# Accuracy calculations
total = len(STANDARIZED_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total
print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))
fig=plt.figure(figsize=(8, 8))

#for i in range(len(MISCLASSIFIED)):
#    plt.subplot(1,len(MISCLASSIFIED),i+1)#because it starts from 1
#    plt.imshow(MISCLASSIFIED[i][0])
#    plt.title(MISCLASSIFIED[i][1])
#    
#    plt.show()
w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(MISCLASSIFIED[i][0])
plt.show()





















