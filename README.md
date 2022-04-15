# GAN_N3

The research project was associated with "[INF-DSAM9] Computational Foundations of Data Science: Deep Learning", Summer Semester 2020, for my Masters of Science: Data Science, University of Potsdam, Germany. 

You can find the Technical Report on [ResearchGate](https://www.researchgate.net/publication/355917170_Conditional_Generative_Adversarial_Network_generate_new_face_images_based_on_attributes).

## Task
Train a conditional generative adversial network on the CelebA dataset. The result should be a model that generates instances of faces with the desired attribute.

## Executing the code in this repository

The notebook is compatible with Google Colab. 

How to run the notebook:

1) Download the notebook "N3_GAN_CelebA_final.ipynb" along with all other "*.py" files into your Colab working directory (indicated by the folder icon on the left when you have a colab notebook open).
2) Change the path directories in "default" argument of parser.add_argument() function according to your need.
3) The first part of the notebook is for data generation. Restart the runtime after executing this part.
4) Load the training in the second part. Let it train.

There is also a version that can be used locally (N3_GAN_local.ipynb) that was used in development and might be easier to understand for some.

See the Colab Demo for a small demonstration of results.
