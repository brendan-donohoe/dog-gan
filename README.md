# dog-gan
A simple, somewhat silly little generative adversarial network to generate images of dogs, consisting of a single kernel script written in Tensorflow for the Kaggle project at https://www.kaggle.com/c/generative-dog-images/overview.  The project uses the Stanford dogs dataset - a 744 MB collection of dog images with annotation files for each image, containing bounding box information for each dog in each image.

If you wish to run this script yourself, set EPOCHS at line 25 to the number of epochs you wish to run the script, IMAGE_PATH at line 26 to the location of the dog image input directory, and ANNOTATION_PATH at line 27 to the location of the annotation CSV file directory.  Wildcards may be used as appropriate.

Here are a few sample images after 269 epochs (roughly 9 hours of wall clock runtime on Kaggle servers):

![Dog Images](https://www.kaggleusercontent.com/kf/18566877/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..WXyXxVsY0E472x8tql0AsA.B3euZcSYsF8MseOyTqb8hJTQWTJNjh1sU65vPZNwzDf6Hhd4os-LuG2toW-vmOmi4raGPml2CFszFrqf-9Uy0PuxyrToj9x9_-xwPsFLgjR92fewUddiohFuRYRvkBohkuVIMNWOSeKjecBgqiLBDg.En2mz-zxeqNIu4jAAHrmWg/IMGS_EPOCH_269.png)
