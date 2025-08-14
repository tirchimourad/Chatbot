import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import streamlit as st
import os

# Initialisation lemmatizer et stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fonction de prétraitement
def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [w.lower() for w in words if w.lower() not in stop_words and w not in string.punctuation]
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

# Fonction pour trouver la phrase la plus pertinente
def get_most_relevant_sentence(query, corpus):
    query_processed = preprocess(query)
    max_similarity = 0
    most_relevant_sentence = "Désolé, je n'ai pas trouvé de réponse pertinente."
    
    for i, sentence_words in enumerate(corpus):
        union = set(query_processed).union(sentence_words)
        if len(union) == 0:
            continue
        similarity = len(set(query_processed).intersection(sentence_words)) / float(len(union))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = st.session_state.sentences[i]
    return most_relevant_sentence

# Fonction chatbot
def chatbot(question, corpus):
    return get_most_relevant_sentence(question, corpus)

# Streamlit
def main():
    st.title("💬 Chatbot basé sur Les Misérables")
    st.write("Posez-moi n'importe quelle question sur le texte !")

    # Charger et prétraiter le corpus UNE SEULE FOIS
    if "processed_corpus" not in st.session_state:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, '135-0.txt')

        with open(path, 'r', encoding='latin-1') as f:
            data = f.read()

        sentences = sent_tokenize(data, language='english')
        st.session_state.sentences = sentences
        st.session_state.processed_corpus = [preprocess(s) for s in sentences]

    # Initialiser l'historique
    if "history" not in st.session_state:
        st.session_state.history = []

    # Champ de saisie
    question = st.text_input("Vous :", value="", placeholder="Tapez votre question ici")

    if st.button("Envoyer"):
        if question.strip() != "":
            response = chatbot(question, st.session_state.processed_corpus)
            st.session_state.history.append({"user": question, "bot": response})

    # Affichage de l'historique
    if st.session_state.history:
        st.subheader("📜 Historique de la conversation")
        for chat in st.session_state.history:
            st.markdown(f"**🧑 Vous :** {chat['user']}")
            st.markdown(f"**🤖 Chatbot :** {chat['bot']}")

if __name__ == "__main__":
    main()

