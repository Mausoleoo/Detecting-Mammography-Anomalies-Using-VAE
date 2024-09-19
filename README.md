# Description
A Variational Autoencoder was used to detect anomalies in breast mammographies. Trained on healthy breast images, with ELBO as the loss function and a latent space of 1024, the model identifies anomalies when the reconstruction error exceeds the average. It also generates realistic images of healthy breasts.

---------------------------------------------------------------------------------------------------------------------------------------------
# Libraries

Torch 2.4.1

Pandas 2.1.3

Numpy 1.26.2

Matplotlib 3.8.3

---------------------------------------------------------------------------------------------------------------------------------------------
# Dataset

The VinDr-Mammo dataset was used, containing images from 5,000 patients, with 4 images per patient, totaling 20,000 images. Each patient has 2 views for both breasts. The dataset's accompanying CSV file, Metadata.csv, was slightly modified to suit the specific needs of the project.

The dataset was previously preprocessed by SSMan and downloaded from

https://www.kaggle.com/datasets/ssmann/vindr-mammo-dataset

---------------------------------------------------------------------------------------------------------------------------------------------
# Training

Due to computational limitations, a sample of 1,000 images was used for training, consisting exclusively of images with no findings (healthy). The model was trained for 450 epochs, with all images resized to 128x128 to ensure consistency and optimize processing efficiency.
