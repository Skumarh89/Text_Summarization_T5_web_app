# Text_Summarization_T5_Streamlit
  This repository contains code for training a T5-based text summarization model and deploying it as a Streamlit web app. The T5 model is fine-tuned on a summarization task using the Hugging Face Transformers library, and the trained model is then loaded and used in a Streamlit web app for generating text summaries.
  #### Table of Contents

   * Prerequisites
   * Training the T5 Summarizer
   * Running the Streamlit Web App
  
  #### Prerequisites

  Before getting started, make sure you have the following:

   * Python (>=3.6)
   * transformers library by Hugging Face (pip install transformers)
   * streamlit library (pip install streamlit)
  ### Training the T5 Summarizer

  1. Clone this repository:
        * git clone https://github.com/Skumarh89/t5-summarization-and-streamlit.git
        * cd t5-summarization-and-streamlit
  2. Prepare your training data and update the train.py script with the dataset and training parameters.
  3. Run the training script to fine-tune the T5 model:
         * python train_T5_sum.py
  4. Once training is complete, the trained model will be saved in the 'saved_model' directory.

  ### Running the Streamlit Web App
  1. Make sure you have the trained T5 model saved in the saved_model directory.
  2. Update the app.py script with the path to the saved model.
  3. Run the Streamlit web app:
       *  streamlit run summary_app.py
  4. Open your web browser and navigate to the provided URL (usually http://localhost:8501) to access the app.
  5. Enter your text in the input area and click the "Generate Summary" button to see the generated text summary.

