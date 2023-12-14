import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder
from transformers import BertTokenizer, TFBertModel
import xgboost as xgb
from sklearn.metrics import accuracy_score

df=pd.read_csv("C://Users//MSI//Desktop//PFE//TEST.csv")
df=df.dropna()
lst1=df["preprocessed_nom_projet"].tolist()
lst2=df["preprocessed_Précision_service_recherché"].tolist()

def One_String(lst):
    data_list = eval(lst)
    mystring=''
    for x in data_list:
        mystring+=' '+x
    return mystring
df['Preprocessed_skills']=df['skills'].apply(One_String)  #create new column for lowercase text
df[['Preprocessed_skills','skills']].head()
lst3=df["Preprocessed_skills"].tolist()

#features
column1_texts =lst1
column2_texts =lst2
column3_texts =lst3
#target
y = df['nom']
labels=y
#bert for embedding
# Load the pre-trained BERT model
# the tokenizer and model will be loaded specifically for multilingual text, including French.
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
# Tokenize and encode the input texts for each column
input_ids_list = []
attention_masks_list = []
for texts in [column1_texts, column2_texts, column3_texts]:
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=20,  # Adjusted
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    input_ids_list.append(tf.concat(input_ids, axis=0))
    attention_masks_list.append(tf.concat(attention_masks, axis=0))
# Generate word embeddings using BERT for each column
embeddings_list = []
for input_ids, attention_masks in zip(input_ids_list, attention_masks_list):
    embeddings = bert_model(input_ids, attention_mask=attention_masks)[0]
    embeddings_list.append(embeddings.numpy())

# Concatenate the embeddings from different columns
concatenated_embeddings = np.concatenate(embeddings_list, axis=1)
#Adding annee_experience
numeric_feature=df["annee_experience"].values
numeric_feature = np.reshape(numeric_feature, (1072, 1))
combined_data = np.zeros((1072, 62, 768))
combined_data[:, :60, :] = concatenated_embeddings
combined_data[:, 60, :] = numeric_feature
concatenated_embeddings=combined_data
#Adding diplome
feature_txt=df["preprocessed_diplome"]
# Encode the labels in y_train using OrdinalEncoder
ordinal_encoder2 = OrdinalEncoder()
feature_txt_encoded = ordinal_encoder2.fit_transform(np.array(feature_txt).reshape(-1, 1))
combined_data[:, 61, :] = feature_txt_encoded
concatenated_embeddings=combined_data
# Split data: train and test
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(concatenated_embeddings, labels, test_size=0.2, random_state=42)
# Reshape the input data
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
# Ordinal encoder for the target
# Encode the labels in y_train using OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
y_train_encoded = ordinal_encoder.fit_transform(np.array(y_train).reshape(-1, 1))
# Encode the labels in y_test using OrdinalEncoder with handling unknown values
ordinal_encoder_with_unknown = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder_with_unknown.fit(np.array(y_train).reshape(-1, 1))
y_test_encoded = ordinal_encoder_with_unknown.transform(np.array(y_test).reshape(-1, 1))
#Applying XGBoost
# Train the XGBoost classifier
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train_encoded)
# Make predictions on the test set
y_pred = xgb_model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Accuracy:", accuracy)
import pickle
# Assuming your trained model is stored in a variable called `xgb_model`
pickle.dump(xgb_model, open("C://Users//MSI//Desktop//pfe_app//model.pkl","wb"))