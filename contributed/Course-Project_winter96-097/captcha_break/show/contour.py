from imutils import paths
import numpy as np
import imutils
import cv2
from random import randint


new_image = cv2.imread('./dataset/%d.png' %randint(0, 1000))

# loop over the image paths
for i in range(1):
    cv2.imwrite("org.jpg", new_image)
    # Load the image and convert it to grayscale
    image = new_image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT)

    # threshold the image (convert it to pure black and white)
    ret,thresh = cv2.threshold(image, 115, 255, cv2.THRESH_BINARY)
    
    
    dilation = cv2.erode(thresh, (17,17), iterations = 1)
    dilation = cv2.bilateralFilter(dilation, 3, 75, 75)
    dilation = cv2.fastNlMeansDenoising(dilation, None, 15, 15, 7)	
    dilation = cv2.dilate(dilation, (31,31), iterations = 1)
    ret,dilation = cv2.threshold(dilation, 10, 255, cv2.THRESH_BINARY)
    

	# Show the annotated image
    cv2.imshow("Output", dilation)
    cv2.imwrite("preprocess.jpg", dilation)
    cv2.waitKey()
	
    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]
    
    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk

        if w > 3 and h > 3 and w < 30:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 5:
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        #letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        #prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        #letter = lb.inverse_transform(prediction)[0]
        #predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        #cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    # Show the annotated image
    cv2.imshow("Output", output)
    cv2.imwrite("contour.jpg", output)
    cv2.waitKey()