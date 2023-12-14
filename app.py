
from flask import Flask, request, render_template
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder
from transformers import BertTokenizer, TFBertModel
import xgboost as xgb
import numpy as np



# Declare a Flask app
app = Flask(__name__)

# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    predicted_labels=[]
    # If a form is submitted
    if request.method == "POST":

        # Get values through input bars
        diplome = request.form.get("diplome")
        annee = request.form.get("annee_experiences")
        nom = request.form.get("nom_projet")
        service = request.form.get("service_recherche")
        competence = request.form.get("competence")
        n = int(request.form.get("nombre_expert_predire"))

        nom=[str(nom)]
        service=[str(service)]
        competence=[str(competence)]

        # the tokenizer and model will be loaded specifically for multilingual text, including French.
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        # Tokenize and encode the input texts for each column
        input_ids_list = []
        attention_masks_list = []
        for texts in [nom, service, competence]:
            input_ids = []
            attention_masks = []
            for text in texts:
                encoded = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=20,  # Adjust as needed
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='tf'
                )
                input_ids.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])
            input_ids_list.append(tf.concat(input_ids, axis=0))
            attention_masks_list.append(tf.concat(attention_masks, axis=0))
        # Generate word embeddings using BERT for each text_feature
        embeddings_list = []
        for input_ids, attention_masks in zip(input_ids_list, attention_masks_list):
            embeddings = bert_model(input_ids, attention_mask=attention_masks)[0]
            embeddings_list.append(embeddings.numpy())
        # Concatenate the embeddings from 3 text features
        concatenated_embeddings = np.concatenate(embeddings_list, axis=1)
        numeric_feature = np.reshape(annee, (1, 1))
        combined_data = np.zeros((1, 62, 768))
        combined_data[:, :60, :] = concatenated_embeddings
        combined_data[:, 60, :] = numeric_feature
        # Encode the labels using OrdinalEncoder
        ordinal_encoder2 = OrdinalEncoder()
        feature_txt_encoded = ordinal_encoder2.fit_transform(np.array(diplome).reshape(-1, 1))
        combined_data[:, 61, :] = feature_txt_encoded
        # concatenated_embeddings=combined_data
        combined_data_reshaped = combined_data.reshape(1, -1)
        # Unpickle classifier
        model = pickle.load(open("model.pkl", "rb"))
        #Making prediction for the input
        #pred = model.predict(combined_data_reshaped)
        pred_probabilities = model.predict_proba(combined_data_reshaped)  # Get predicted class probabilities
        # Find the indices of the top classes with highest probabilities
        top_classes_indices = np.argsort(pred_probabilities, axis=1)[:, ::-1][:, :n]
        predicted = top_classes_indices.reshape(-1, 1)
        ordinal_encoder = pickle.load(open("C://Users//MSI//Desktop//pfe_app//ordinal_encoder.pkl", "rb"))
        # Perform inverse transformation on the predicted encoded values
        predicted_labels = ordinal_encoder.inverse_transform(predicted)
    else:
        prediction = ""

    return render_template("index.html", predictions=predicted_labels)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)