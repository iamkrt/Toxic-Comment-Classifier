# 1.Introduction

Conversational toxicity is an issue that can lead people both to stop genuinely expressing themselves and to stop seeking others’ opinion out of fear of abuse/harassment. The goal of this project will be to use deep learning to identify toxicity in tests, which could be used to help deter users from posting potentially hurtful messages, craft more civil arguments when engaging in disclosure with others, and to gauge the toxicity of other users’ comments. Online forums and social media platforms have provided individuals with the means to put forward their thoughts and freely express their opinion on various issues and incidents. In some cases, these online comments contain explicit language which may hurt the readers. Comments containing explicit
language can be classified into myriad categories such as Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate. The threat of abuse and harassment means that many people stop expressing themselves and give up on seeking different opinions.
To protect users from being exposed to offensive language on online forums or social media sites, companies have started flagging comments and blocking users who are found guilty of using unpleasant language. Several Machine Learning models have been developed and deployed to filter out the unruly language and protect internet users from becoming victims of online harassment and cyberbullying.





1.1 Problem statement

To build a multi-headed model using Deep Learning algorithm, that is capable of detecting and classifying different types of toxicity based on the comments; and to Implement the final neural network model with an interactive user interface using Gradio.




1.2 Data


Dataset: Toxic Comment Classification Challenge
Source: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-
challenge/data
Dataset is taken from Kaggle’s classification challenge to identify and classify toxic online comments. It has eight columns in which toxic comments are categorised into six types.
<img width="912" alt="Screen Shot 2022-06-17 at 12 21 13 PM" src="https://user-images.githubusercontent.com/42109704/174337888-c3a382e2-f448-4022-a518-64f253e01605.png">







# 2. Methodology



**4.1 Data Exploration, Data Pre-processing
**
<img width="1276" alt="Screen Shot 2022-06-17 at 12 21 37 PM" src="https://user-images.githubusercontent.com/42109704/174337941-c813cd8b-bb4b-4bdb-8896-98ee74ccd413.png">


Step 1: Import and Explore the data



● Import requires libraries and reads the csv dataset file as pandas dataframe.
● Load the dataset and explore the data.
● There are eight columns in the dataset. In which the first column is the unique id, for the
corresponding comment text in the second column. The remaining columns are labels for classifying toxicity. These columns have binary values to show if the comment can be classified into that particular toxicity.
● Check for null values in the dataset
● Check the category of toxic comments and plot a bar graph visualising the number of
occurrences against each category. This is to identify balance in the dataset.

<img width="802" alt="Screen Shot 2022-06-17 at 12 15 48 PM" src="https://user-images.githubusercontent.com/42109704/174337022-65b4596d-d065-4d71-88ff-3b43ed20ab7e.png">



## Step 2: Stopwords Removal.

Stopwords Removal, as we all know, is one of the most critical steps in text pre-processing for use-cases that involve text classification. Removing stopwords ensures that more focus is on those words that define the meaning of the text.


● To remove stopwords from data, “nltk” library was imported. nltk has a list of common stopwords, “STOP_WORDS” that can be used to remove stopwords from any textual data.
● Once the above task is completed, search for words in the dataset that could be possible stopwords. Remove stopwords from training data and test data. Once this step is completed, a clean data set that is free from all inconsistencies is obtained.
● The difference in the data before and after stopwords removal can be checked.



# Step 3: Text Normalization.

The text normalization steps performed are listed below:

● Removing Characters in between Text.
● Removing Repeated Characters.
● Converting data to lower-case.
● Removing Punctuation.
● Removing unnecessary white spaces in between words.
● Removing “\n”.
● Removing Non-English characters.


## 2.2 Split Training Data into Train-Set and Validation-Set.
<img width="1297" alt="Screen Shot 2022-06-17 at 12 22 32 PM" src="https://user-images.githubusercontent.com/42109704/174338090-1a16e102-798b-4f4d-b6c4-02553ed28033.png">


After completing the data-preprocessing of the project, Since we have completed the data pre- processing and feature engineering part of our project, we move on to the model creation and model assessment part of the project. Before trying to fit a deep learning model on the training data, randomly split the data into train-set and test-set. The validation set accounts for 20% of the training data.
  
  
  
  # Model  - Bidirectional LSTM
  <img width="1091" alt="Screen Shot 2022-06-17 at 12 23 18 PM" src="https://user-images.githubusercontent.com/42109704/174338188-719af114-2e3d-4ea6-8aed-53bbf5809e5a.png">

  
The principle of Bidirectional LSTM is to split the neurons of a regular LSTM into two
directions, one for positive time direction forward states, and another for negative time direction
backward states, the outputs are not connected to inputs of the opposite direction states. By using
two time directions, input information from the past and future of the current time frame can be
used unlike standard LSTM which requires the delays for including future information. Our
model here consisted of the same LSTM structure only difference was every LSTM cell was
made bidirectional, so that the propagation of the signal might be in both the forward and the
backward ways. 


<img width="832" alt="Screen Shot 2022-06-17 at 12 19 14 PM" src="https://user-images.githubusercontent.com/42109704/174337556-94287447-2b05-4834-b328-b1ad00d14f07.png">

**
The train and test accuracy of the model are 95.59% and 95.54% respectively, whereas the train and test loss are 5.57% and 5.97% respectively. Bidirectional LSTM outperforms the other models when we want our model to learn from long term dependencies. Its ability to forget, remember and update the information pushes it one step ahead of RNNs.**


# User Interface with GRADIO for Sentiment Analysis.


It is important to demonstrate our project to the audience. We used the python library Gradio to create a GUI, so that
the dynamics of the project can be exhibited clearly. The user has to type
the comment in the UI and click the submit button to see the prediction results. The machine
learning model classifies the comment and will display classes with the probability. Following is
the Sentiment analysis user interface.


<img width="917" alt="Screen Shot 2022-06-17 at 12 20 01 PM" src="https://user-images.githubusercontent.com/42109704/174337688-5ac38310-2763-4553-bb7d-29ec96af4329.png">

