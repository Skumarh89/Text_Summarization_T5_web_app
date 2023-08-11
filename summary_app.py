import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the saved model and tokenizer
loaded_model = T5ForConditionalGeneration.from_pretrained("saved_model")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loaded_model = loaded_model.to(device)

# Function to generate summary
def generate_summary(text):
    input_encoding = tokenizer.batch_encode_plus([text], max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = input_encoding['input_ids'].to(device, dtype=torch.long)
    input_mask = input_encoding['attention_mask'].to(device, dtype=torch.long)

    generated_ids = loaded_model.generate(
        input_ids,
        attention_mask=input_mask,
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    summarized_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summarized_text

# Streamlit web app
def main():
    st.title("Text Summarization App")
    st.write("Enter your text and get a summarized version!")

    user_input = st.text_area("Input Text", "", height=200)
    if st.button("Generate Summary"):
        if user_input:
            summary = generate_summary(user_input)
            st.subheader("Generated Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to generate a summary.")

if __name__ == '__main__':
    main()
