# Practice 3 - Generative Adversarial Networks (GANs)

## Objective

To investigate and compare Generative Adversarial Networks (GANs) with Variational Autoencoders (VAEs) using the CelebA dataset.

## Dataset

The CelebA dataset, which contains more than 200,000 face images, will be used. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

## Tasks Performed

1. **Preliminary Research:**
   - Reviewing fundamental concepts of VAEs and GANs.
   - Studying specific architectures such as WGAN-GP and DCGANs.

2. **Data Preprocessing:**
   - Downloading and preprocessing the CelebA dataset.

3. **Training and Evaluation:**
   - Training VAE and WGAN-GP models.
   - Experimenting with hyperparameters and architectural modifications.
   - Evaluating using Frechet Inception Distance (FID).

4. **Latent Space Exploration:**
   - Evaluating generated images.
   - Interpolating in the latent space and generating random images.

5. **Comparative Analysis and Reporting:**
   - Comparing the performance of GANs and VAEs.
   - Documenting detailed findings and observations.

## Execution Instructions

Ensure that the necessary libraries, mainly TensorFlow and Keras, are installed and run the notebooks in a Jupyter environment.


### Important Files

1. `/02_wgan/Assignment_wgan_gp.ipynb` - This notebook contains the implementation and training process of various WGAN-GP models, with exploration of the latent space, some explanations of the development process and some conclusions on the observed results.
2. `/03_VAE/01_VAE_faces.ipynb` - 
3. `Results_Comparison.ipynb` - This notebook includes the implementation and training process of the VAE model, also with an exploration of the latent space. It also includes some explanations of the development process.
4. `fid.py` - This file contains utility functions to compute de FID during the training of the models.
