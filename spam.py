import tkinter as tk
from tkinter import messagebox
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download stopwords if you haven't already
nltk.download('stopwords')

# Preprocessing text (similar to the model's preprocessing)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Load the trained model and vectorizer
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Function to predict if the email is spam or not
def classify_email():
    email_text = email_entry.get("1.0", tk.END)
    if not email_text.strip():
        messagebox.showerror("Input Error", "Please enter some email text!")
        return
    # Preprocess and predict
    processed_text = clean_text(email_text)
    email_vector = tfidf.transform([processed_text]).toarray()
    prediction = model.predict(email_vector)
    
    result = "SPAM" if prediction[0] == 1 else "NOT SPAM"
    messagebox.showinfo("Classification Result", f"The email is classified as: {result}")

# GUI Design
root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("400x300")

# Title Label
title_label = tk.Label(root, text="Spam Email Classifier", font=("Helvetica", 16))
title_label.pack(pady=10)

# Email Input Label
input_label = tk.Label(root, text="Enter the email text:", font=("Helvetica", 12))
input_label.pack()

# Email Text Box
email_entry = tk.Text(root, height=10, width=40)
email_entry.pack(pady=10)

# Classify Button
classify_button = tk.Button(root, text="Classify", command=classify_email, font=("Helvetica", 12))
classify_button.pack(pady=10)

# Run the GUI event loop
root.mainloop()
