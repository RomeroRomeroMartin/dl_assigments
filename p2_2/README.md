# Practice 2_2 - Recurrent Neural Networks (RNNs)

## Objective

The objective of this practice is to develop various deep recurrent neural network models to solve a text classification problem using the Amazon Reviews for Sentiment Analysis dataset.

## Dataset

The "Amazon Reviews for Sentiment Analysis" dataset from Kaggle is used. This dataset consists of Amazon customer reviews (input text) and star ratings (output labels).

- The classes are `__label__1` (1- and 2-star reviews) and `__label__2` (4- and 5-star reviews). Neutral 3-star reviews are excluded.
- The dataset includes reviews primarily in English, with some in other languages such as Spanish.
- A reduced version of the dataset is used, with 25,000 examples for training and 25,000 for testing.

## Tasks Performed

1. **Data Loading and Transformation:**
   - Implemented the `generateAmazonDataset.ipnb` file to load and transform the data.
   - Used `transformData` to convert text input to integer input based on a vocabulary using the Keras function `TextVectorization`. This includes setting the vocabulary size (`maxFeatures`) and the maximum length of the text (`seqLength`).

2. **Model Development:**
   - Created several RNN models using different Keras layers such as Dense, SimpleRNN, and LSTM.
   - Ensured that at least one layer in each model is an RNN.

3. **Hyperparameter Tuning:**
   - Experimented with various hyperparameters, including the number of layers, number of units, activation functions, regularization, batch size, vocabulary size (`maxFeatures`), maximum text length (`seqLength`), and embedding dimensions.

4. **Training and Evaluation:**
   - Trained the models using the training set.
   - Evaluated the models using the test set, with classification accuracy as the evaluation metric.

5. **Results and Discussion:**
   - Included a detailed explanation of the process, problems encountered, and justifications for decisions made.
   - Discussed the results and interpretations in the notebook.


## Conclusion

This practice explores the use of deep recurrent neural networks for text classification. Through various experiments and model evaluations, insights were gained into the performance and behavior of RNNs in sentiment analysis tasks.

