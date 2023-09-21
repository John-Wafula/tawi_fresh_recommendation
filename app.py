import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from sklearn.decomposition import TruncatedSVD


# load test data for seeing current image
test_data = pickle.load(open('test_data.pkl','rb'))
test_data_ = pd.DataFrame(test_data)

# load train data for seeing recommendations
train_data = pickle.load(open('img_data.pkl','rb'))
train_data_ = pd.DataFrame(train_data)

# load model;
knn = pickle.load(open('model_recommend.pkl','rb'))




# tfidf for text:
X_test = pickle.load(open('test_array.pkl','rb'))

st.title("Fashion Recommendation system (Product Based Recommender System) -  Tawi Fresh Demo")

st.header('About Recommendation model:')

st.markdown("The model used here is 'Nearest Neighbours'. For a given data point it gives "
            "us similar points within the neighbourhood. Here, for a given women wear we get 10 more recommendations. "
            "Also this model depends heavily on the 'title' of the product but also takes into "
            "consideration the color and brand of the same.")

title_current = st.selectbox('Search for the product you want here:',
                    list(test_data_['title']))
product = test_data_[(test_data_['title'] == title_current)]
s1 = product.index[0]
captions = [test_data_['brand'].values[s1],test_data_['formatted_price'].values[s1]]
c1,c2,c3 = st.columns(3)
with c1:
    st.image(test_data_['medium_image_url'].values[s1])
with c2:
    st.text('Brand--->')
    st.text('Color--->')
    st.text('Price in $--->')
with c3:
    st.text(test_data_['brand'].values[s1])
    st.text(test_data_['color'].values[s1])
    st.text(test_data_['formatted_price'].values[s1])
    



distances, indices = knn.kneighbors([X_test.toarray()[s1]])
result1 = list(indices.flatten())[:5]
result2 = list(indices.flatten())[5:]


if st.button('get more products'):
    st.success('Hope you like the below recommendations :)')
    col1,col2,col3,col4,col5 = st.columns(5)
    lst1 = [col1,col2,col3,col4,col5]
    for i,j in zip(lst1,result1):
        with i:
            st.text(train_data_['brand'].values[j])
            st.text(train_data_['color'].values[j])
            st.image(train_data_['medium_image_url'].values[j])

    col6, col7, col8, col9, col10 = st.columns(5)
    lst2 = [col6, col7, col8, col9, col10]
    for k,l in zip(lst2,result2):
        with k:
            st.text(train_data_['brand'].values[l])
            st.text(train_data_['color'].values[l])
            st.image(train_data_['medium_image_url'].values[l])

    st.success('Thank You for Shopping, Karibu Tena  !!')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer: {
	content:Made by Bairagi Saurabh :);
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Beauty products Recommendation system (Collaborative filtering) ")

st.header('About Recommendation model:')

st.markdown("1. Recommend items to users based on purchase history: This means that the recommendation system takes into account the history of what a user has previously bought or interacted with on the platform. It uses this historical data as one of the factors in making recommendations.")

st.markdown("2. Similarity of ratings provided by other users who bought items: This involves analyzing the ratings and feedback given by other users who have purchased or interacted with similar items. The system identifies users who have similar tastes or preferences to the target user and looks at the ratings they've given to various items.")

st.markdown("3. Similarity of ratings to that of a particular customer: The system then compares the ratings and preferences of a particular customer (the target user) to those of other users with similar tastes. It identifies items that these similar users have liked and recommends those items to the particular customer.")

import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv("ratings_Beauty.csv")
data = data.head(1000)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
product_id = st.selectbox('Pick a product to list similar products ',
                    list(train_data['ProductId'].unique()))
ratings_utility_matrix = train_data.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
X = ratings_utility_matrix.T

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

G = nx.Graph()

product_names = list(X.index)
G.add_nodes_from(product_names)

i = product_id
product_ID = product_names.index(i)

correlation_product_ID = decomposed_matrix[product_ID]

threshold = 0.90  

for j, corr_value in enumerate(correlation_product_ID):
    if i != j and corr_value > threshold:
        G.add_edge(i, product_names[j], weight=corr_value)

personalized_pagerank = nx.pagerank(G, alpha=0.85)
sorted_recommendations = sorted(personalized_pagerank.items(), key=lambda x: x[1], reverse=True)

recommendations = [product for product, score in sorted_recommendations if product != i]

mae_values = []
rmse_values = []

for user_id in test_data['UserId'].unique():
    user_test_data = test_data[test_data['UserId'] == user_id]
    
    for _, row in user_test_data.iterrows():
        product_id = row['ProductId']
        actual_rating = row['Rating']
        
        
        if recommendations:
            predicted_rating = recommendations[0][1]
            mae_values.append(abs(actual_rating - float(predicted_rating)))
            rmse_values.append((actual_rating - float(predicted_rating)) ** 2)

# Calculate MAE and RMSE
mae = np.mean(mae_values)
rmse = np.sqrt(mae)

# Print evaluation metrics
# st.write("Mean Absolute Error (MAE):", mae)
st.write("Root Mean Squared Error (RMSE):", rmse)

st.write(" Total similar products based on user interaction for "+ " "+ product_id + " " + "are:" )
st.write(recommendations)

st.text("Top 10 recommended products for "+ " "+ product_id + " " + "are:" )
st.write(recommendations[0:10])

st.title("Beauty products Recommendation system (Popularity Based Recommender System) ")

st.header('About Recommendation model:')

st.markdown("The model targets new customers who do not have any product history by recommending the most highly rated products")
data = pd.read_csv("ratings_Beauty.csv")
popular = (data.groupby('ProductId')[['Rating']].count().sort_values('Rating', ascending=False).reset_index()).head(20)
st.markdown("The Most popular products are:")

st.text(popular['ProductId'])


df = data[data['ProductId'].isin(popular['ProductId'])]

product_reviews = df.groupby('ProductId').agg({
    'UserId': 'count',
    'Rating': lambda x: (x == 4.0).sum() + (x == 5.0).sum()
})

product_reviews = product_reviews.rename(columns={'UserId': 'Total_Reviews', 'Rating': '4_and_5_Star_Ratings'})

product_reviews.reset_index(inplace=True)

product_reviews = product_reviews.sort_values(by='Total_Reviews', ascending=False)



st.table(product_reviews.head(20))
