import streamlit as st
from rake_nltk import Rake
import spacy
from textblob import TextBlob
from wordcloud import WordCloud

st.title("Entity Extraction App")

text = st.text_area("Введите текст")

if st.button("Показать граф связей слов") and text.strip():
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]

    G = nx.Graph()
    G.add_nodes_from(keywords)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold',
            node_size=500, node_color='skyblue', font_size=8)
    st.pyplot(plt)

if st.button("Извлечь ключевые фразы (RAKE)") and text.strip():
    r = Rake(language='russian')
    r.extract_keywords_from_text(text)
    phrases_with_scores = r.get_ranked_phrases_with_scores()[:5]
    keywords = [phrase for _, phrase in phrases_with_scores]
    st.write("Ключевые фразы (RAKE):")
    st.write(keywords)

if st.button("Извлечь ключевые слова (spaCy)") and text.strip():
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    st.write("Ключевые слова (spaCy):")
    st.write(keywords)

if st.button("Извлечь сущности (spaCy)") and text.strip():
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    st.write("Сущности (spaCy):")
    st.write(entities)

if st.button("Анализ тональности") and text.strip():
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    polarity_percentage = int(polarity * 100)
    polarity_str = 'Нейтральная'
    if polarity > 0:
        polarity_str = f'Положительная ({polarity_percentage}%)'
    elif polarity < 0:
        polarity_str = f'Отрицательная ({polarity_percentage}%)'
    st.write(f'Тональность текста: {polarity_str}')
    st.write(f'Полярность: {polarity:.2f}, Субъективность: {subjectivity:.2f}')

if st.button("Облако слов") and text.strip():
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
