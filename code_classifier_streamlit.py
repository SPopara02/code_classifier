import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
from transformers import pipeline
@st.cache_resource
def fetch_model():
    return pipeline(model= 'facebook/bart-large-mnli')
bert_zeroShot = fetch_model()
candidate_labels = ['Java', 'Javascript', 'C', 'C++', 'C#', 'Python']
df = pd.read_csv('code_snippets.csv')
X, Y = df['code'], df['language']
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle= True, random_state= 42, test_size=0.2)
pipelineSVC = joblib.load('SVC.pkl')
pipelineMNB = joblib.load('MultinomialNB.pkl')
pipelineRF = joblib.load('RandomForest.pkl')
model = object()




st.title("AI klasifikator programskih jezika 🤖")
st.write("Unesite kod, odaberite željeni model iz liste, a potom kliknite na dugme 'Klasifikuj' kako biste dobili klasifikaciju.")
user_input = st.text_area("Unesite kod")
dropdown = st.selectbox(options= ['SVC + bag-of-words', 'MultinomialNB + TF-IDF', 'RandomForest'], label='Odaberite model', index=0)

if st.button("Klasifikuj"):
    if user_input.strip == "":
        st.warning('Morate uneti kod!')
    else:
        match dropdown:
            case 'SVC + bag-of-words':
                model = pipelineSVC
            case 'MultinomialNB + TF-IDF':
                model = pipelineMNB
            case 'RandomForest':
                model = pipelineRF
        prediction = model.predict([user_input])[0]
        bert_prediction = bert_zeroShot(user_input, candidate_labels, multi_label= False)
        st.subheader(f"Rezultat klasifikacije {dropdown} modelom:")
        st.write(f"Ovaj kod je pisan u **{prediction}** programskom jeziku")
        st.subheader("Rezultat klasifikacije BERT LLM modelom:")
        st.code(f"{bert_prediction['labels']}\n{bert_prediction['scores']}")
        st.subheader(f"Metrike performansi {dropdown} modela:")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, model.predict(x_test)),cmap='rocket_r',annot=True,xticklabels=model.classes_,yticklabels=model.classes_)
        plt.title(f"Matrica konfuzije {dropdown} modela")
        plt.xlabel("Predikcije")
        plt.ylabel("Stvarne vrednosti")
        st.pyplot(fig)
        st.write(f"Accuracy Score: {accuracy_score(y_test, model.predict(x_test)):.2f}")
        st.write(f"Classification report:")
        st.code(classification_report(y_test, model.predict(x_test)), language= 'text')