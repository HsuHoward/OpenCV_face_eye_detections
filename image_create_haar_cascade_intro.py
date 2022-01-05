"""
make my own Haar Cascade for image and video object classification
steps:
1. collect "negative" or "background" images.
- any image will do, just make sure your object is no present in them! Get thousands.
- generally a bg.txt file that contains the path to each image, by line
    e.g. line, neg/1.jpg
2. collect or create "positive" image. (image-net.org)
- Thousands of images of your object (50x50 pixels). Can make these based on one image, or manually create them.
- sometimes called "info," pos.txt, or something of this sort.
  Contains path to each image, by line, along with how many objects, and where they are located.
    e.g. line, pos/1.jpg 1 0 0 50 50 (image, num objects, start points, rectangle coordinates)
3. create a positive vector file by stitching together all positives
- This is done with an OpenCV command.
4. train cascade.
- Done with OpenCV command.

Notes
1. you want negative images larger than positive images generally if you are going to "create samples" rather than
   collect and label positives.
2. try to use small images. 100x100 for negatives, 50x50 for positives
3. will get even smaller when it comes to training!
4. have~double positive images compared to negative for training
"""

