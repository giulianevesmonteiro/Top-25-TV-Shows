# Top-25-TV-Shows
As an unstructured data analytics project, I scraped a page of the Rotten Tomatoes website to complete an analysis of the current top 25 TV shows.  I did an analysis comparing the critics' and audiences' ratings scores. Additionally, I obtained the comments and respective ratings for all TV shows and did sentiment analysis and topic modeling.

## Introduction
The beginning of 2025 was very promising for us TV show enthusiasts. Many new seasons for some of our favorite shows were finally coming back. Some even after a few years of waiting, such as Severance season 2. There was also the addition of a few new and very unique TV shows like Paradise. As someone who loves her TV shows and has a great ability to bing watching new seasons as soon as they come out, I decided to take a look at the top 25 TV shows of 2025 according to Rotten Tomatoes. 

This would be useful for a couple of reasons. First, it would help me decide which TV shows to watch next. There are just so many new ones coming out at the same time, I didn't evne know where to start with. Second, it would be ineterseting to see if the shows I'm watching and enjoying are also popular among other people.

So for this unstructured data project, we will be scrapping the top 25 TV shows website from Rotten Tomatoes. We will be looking at the TV show's title, rating, critics consensus, and popularity scores. We will then dive into the comments section for each of these shows and complete some topic models and sentiment analysis.

## Data Collection and Preping
```{python}
# Importing the relevant packages
from bs4 import BeautifulSoup
import pandas as pd
import requests

link = 'https://editorial.rottentomatoes.com/guide/popular-tv-shows/'
# Get the information from the website
page = requests.get(link) 

# Passing the html into a BeautifulSoup object
soup = BeautifulSoup(page.content, 'html.parser')

# Prepare a list to hold rows of data
data_list = []

# Selecting the items
shows = soup.select('.countdown-item')  # Adjusted to a parent class for better scoping

# Iterate through each show and scrape required data
for show in shows:
    try:
        # Scrape information from the selected show
        Title = show.select_one('.article_movie_title').text.strip()  
        TomatoRating = show.select('.tMeterScore')[0].text.strip()
        PopcornRating = show.select('.tMeterScore')[1].text.strip() 
        CriticsConsensus = show.select_one('.info.critics-consensus').text.strip()  
        PopularityScore = show.select_one('.countdown-index').text.strip()  

        # Store the scraped data in a dictionary and changing some column names
        data_list.append({
            'TVShow': Title,
            'TomatoRating': TomatoRating,
            'PopcornRating': PopcornRating,
            'CriticsConsensus': CriticsConsensus,
            'Top25Rating': PopularityScore,
        })
    except AttributeError:  # In case there are some missing elements
        continue

# Changing data_list dictionary into a data frame
data_frame = pd.DataFrame(data_list)
# Let's confirm if they are all there:
print(data_frame)
```

## Data Cleaning
We need to adjust the columns to make them clean with only the information we actually need.
```{python}
# Cleaning the TVShow column by only extracting the words before the ':' to only have the TV show name
data_frame['TVShow'] = data_frame['TVShow'].str.extract(r'([^:]+)')[0].str.strip() 
    # using str.strip to remove any whitespace from the resulting string, if any
# Print the updated data_frame to check the changes
#print(data_frame)

# Adjusting Top25Rating so that it removed the '#' and only leaves the number itself
data_frame['Top25Rating'] = data_frame['Top25Rating'].str.replace(r'#', '').str.strip()
# Print the updated data_frame to check the changes
#print(data_frame)

# Adjusting TomatoRating so that it removed the '%' and only leaves the number itself
data_frame['TomatoRating'] = data_frame['TomatoRating'].str.replace(r'%', '').str.strip()
# Print the updated data_frame to check the changes
#print(data_frame)

# Adjusting PopcornRating so that it removed the '%' and only leaves the number itself
data_frame['PopcornRating'] = data_frame['PopcornRating'].str.replace(r'%', '').str.strip()
# Print the updated data_frame to check the changes
#print(data_frame)

# Clean the CriticsConsensus column by removing 'Critics Consensus:' and just leaving the comment itself
data_frame['CriticsConsensus'] = data_frame['CriticsConsensus'].str.replace('Critics Consensus:', '')
# Print the updated data_frame to check the changes
#print(data_frame)

# Adjusting for the 1 missing value in the data frame. Having it be the same as the TomatoRating, for simplicity of this project
data_frame['PopcornRating'] = data_frame['PopcornRating'].str.replace('- -', '100')
# Print the updated data_frame to check the changes
print(data_frame)
```

We still need some more data cleaning, such as changing the data types of the columns.
```{python}
# Lets check the data types to see which we have to fix
data_frame.dtypes

# Changing the data type of the numeric columns
data_frame['TomatoRating'] = pd.to_numeric(data_frame['TomatoRating'])
data_frame['PopcornRating'] = pd.to_numeric(data_frame['PopcornRating'])
data_frame['Top25Rating'] = pd.to_numeric(data_frame['Top25Rating'])

# Checking if the changes were done correctly
print(data_frame)
data_frame.dtypes

round(data_frame['PopcornRating'])
```

## Correlation Analysis
Let's get the correlation between the Tomato and Popcorn ratings. The Tomato rating is from critics and the Popcorn ratings are from the audience. It would be interesting to see if there is a correlation between the two or if there is a discrepancy in opinions. 

```{python}
rating_correlation = data_frame['TomatoRating'].corr(data_frame['PopcornRating']).round(2)
print(f"The correlation between the Tomato ratings (done by critics) and the Popcorn ratings (the audience) is {rating_correlation}")
```

A correlation coefficient of 0.55 indicates a moderate to strong positive correlation between the Tomato and Popcorn ratings. This means that as one variable increases, the other variable also increases

## Data Visualization
Lets visualize the Tomato ratings and the Top 25 ratings in a scatterplot to see if there is any pattern between the two.
```{python}
import plotly.express as px

fig = px.scatter(data_frame, x = 'Top25Rating', y = 'TomatoRating')
fig.update_layout(showlegend=False) #removing the legend
fig.update_layout(title = 'Top TV Shows vs Tomato Rating') #adding a title
fig.show()
```

One would expect to see top rated TV shows to have higher Tomato ratings than the rest of the TV shows, but this scatterplot shows us that there is no pattern at all.  A TV show could be rated as number 30 in the list and still have a Tomato rating above a 90. Or the opposite scenario. 

Let's visualize the Tomato rating and Top 25 with a bar chart instead.
```{python}
fig2 = px.bar(data_frame, x = 'Top25Rating', y = 'TomatoRating') 
fig2.update_layout(title = 'Top TV Shows vs Tomato Rating')
fig2.show()
```
We can see that a Tomato rating of 100 is not just attributed to the top x% of TV shows, it can range between many. It also doesn't necessarily mean that the Top 1 TV show has the best Tomato Rating, as this isn't the case here.

Now let's take a look at the Popcorn ratings and the Top 25 ratings in a bar graph.
```{python}
fig4 = px.bar(data_frame, x = 'Top25Rating', y = 'PopcornRating') 
fig4.update_layout(title = 'Top TV Shows vs Popcorn Rating')
fig4.show()
```
They also don't display a pattern where top rated TV shows have the highest ratings. It's very spread out. 


## Sentiment Analysis
Let's dive into the comments section of the TV shows and see what people are saying about them. 

### Gathering the Comments
```{python}
#| eval: false
# Not running this code only showing it, since sync_playwright doesn't work with the code chunks.
# This ran in a separate python script and the data frame will be imported in the next chunk.
from playwright.sync_api import sync_playwright, Playwright

pw = sync_playwright().start()

chrome = pw.chromium.launch(headless=False)

page = chrome.new_page()  # will actually open the browser

page.goto('https://editorial.rottentomatoes.com/guide/popular-tv-shows/')

tv_show_count = page.locator('.article_movie_title a').count()
print(f"Total TV shows: {tv_show_count}")  # Confirm it shows 30

all_reviews = []  # List to store reviews from all TV shows

# Loop through the TV shows
for index in range(tv_show_count):
    # Get the TV show name
    tv_show_name = page.locator('.article_movie_title a').nth(index).inner_text().strip()

    # Clicking on the TV show
    page.locator('.article_movie_title a').nth(index).click()

    # Wait for the "View All" link for audience reviews and click it
    page.wait_for_selector('.audience-reviews :has-text("View All")')
    page.locator('.audience-reviews :has-text("View All")').nth(1).click()

    # Wait for audience reviews to load
    page.wait_for_timeout(2000)

    # Get the count of reviews
    review_count = page.locator('[data-qa="review-text"]').count()
    print(f"Found {review_count} reviews for TV show {tv_show_name}.")

    # Collect the text from each review along with the ratings
    for i in range(review_count):
        review_text = page.locator('[data-qa="review-text"]').nth(i).inner_text().strip()
        
        # Extract the rating for each review
        rating_element = page.locator('rating-stars-group').nth(i)
        if rating_element.count() > 0:
            rating = rating_element.get_attribute('score')
        else:
            rating = None  # Default to None if no rating is found
        
        all_reviews.append({"TV Show": tv_show_name, "Review": review_text, "Rating": rating})

    # Go back to the main TV shows page to select the next one
    page.goto('https://editorial.rottentomatoes.com/guide/popular-tv-shows/')
    page.wait_for_selector('.article_movie_title a')

# Store all reviews in a DataFrame
df = pd.DataFrame(all_reviews)

# Print the DataFrame
print(df.head(30))

# Close the browser
chrome.close()
```

### Data Cleaning
Time to clean the data and remove any comments that are not in English since I saw some in Spanish and Portuguese.

```{python}
import pandas as pd
# Importing the csv file that contains the data frame with all the comments and reviews.
df = pd.read_csv('C:\\Users\\Graduate\\Documents\\ND\\Unstructured_Data_Analytics\\shows.csv')

# Cleaning the TVShow column by only extracting the words before the ':' to only have the TV show name
df['TV Show'] = df['TV Show'].str.extract(r'([^:]+)')[0].str.strip() 
# Chek if it worked
df.head(30)


# Removing the comments not in English
from langdetect import detect, DetectorFactory #package that detects languages

# Function to filter out non-English reviews
def is_english(comment):
    try:
        return detect(comment) == 'en'
    except:
        return False


# Apply the function to the 'Review' column. Will remove all rows that have a comment not in English
df = df[df['Review'].apply(is_english)]
df.head(30)
df.tail(30)

# For the sake of cleaning, lets remove the first column using indexing
df = df.iloc[:, 1:]
# Check it worked:
print(df.head(30))
```

Let's remove any special characters and emojis from the comments. This will help us with the sentiment analysis later on.
```{python}
import re

# Function to remove emojis
def remove_emojis(text):
    # The regex pattern used here matches various ranges of Unicode characters for emojis
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F"  # emoticons
                                 "\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 "\U0001F680-\U0001F6FF"  # transport & map symbols
                                 "\U0001F700-\U0001F77F"  # alchemical symbols
                                 "\U0001F780-\U0001F7FF"  # geometric shapes extended
                                 "\U0001F800-\U0001F8FF"  # supplemental arrows-C
                                 "\U0001F900-\U0001F9FF"  # supplemental emojis
                                 "\U0001FA00-\U0001FA6F"  # chess symbols
                                 "\U00002702-\U000027B0"  # various emojis
                                "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Remove emojis from the 'Text' column
df['Review'] = df['Review'].apply(remove_emojis)

# Display the DataFrame after removing emojis to check if it worked:
print(df.head(30))
```


## Sentiment Analysis
Lets create the sentiment analysis using TextBlob to get the sentiment scores for each comment. We will then get the correlation between the sentiment scores and the ratings from the comments:
```{python}
from textblob import TextBlob

scores = [] #creating an empty list to store the sentiment scores

for review in df["Review"]:
  blob = TextBlob(review)
  score = blob.polarity
  scores.append(score)

#Confirming if the length of the scores list is the same as the length of the ratings list
len(scores), len(df["Rating"]) #rows line up

# Adding the sentiment score to the pandas dataframe
df["blob_score"] = scores 
df.head(30)


# Getting the correlation between the ratings and the simple score
blob_score_correlation = df["blob_score"].corr(df["Rating"])
blob_score_correlation = blob_score_correlation.round(2)

print(f"The correlation between the sentiment score using Textblob and the ratings from the comments is {blob_score_correlation}")
```
This moderate positive correlations indicated there's a noticeable relationship between the sentiment scores and the ratings, but the relationship isn't significantly strong. It falls in the range where you can observe a clear trend with significant variation among data points. 

Let's visualize the ratings and the sentiment scores in a scatterplot to see if there are any pattern between the two.
```{python}
# But first, sort the DataFrame by rating to ensure order
blob_df = df # making a copy of our data frame
blob_df = df.sort_values(by='Rating')

# Now plot:
fig = px.scatter(blob_df, x = 'Rating', y = 'blob_score')
fig.update_layout(showlegend=False) #removing the legend
fig.update_layout(title = 'Ratings vs Sentiment') #adding a title
fig.show()

# Another way for us to visualize it
fig2 = px.box(blob_df, x = 'Rating', y = 'blob_score') #making a boxplot
fig2.update_layout(title = 'Ratings vs Sentiment')
fig2.show()
```

We will now filter the data frame for a specific TV show and then create a box plot to see the ratings and sentiment scores for that show. I picked Paradise as an example, as I'm currently loving this show and want to see if people share the same opinion.
```{python}
# Specify the TV show you want to filter by
specific_show = 'PARADISE'

# Filter the DataFrame for the specific TV show
filtered_df = blob_df[blob_df['TV Show'] == specific_show]

# Create the box plot using the filtered DataFrame
fig2 = px.box(filtered_df, x='Rating', y='blob_score', title=f'Ratings vs Sentiment for {specific_show}')
fig2.show()
```
As we can see, the higher blob scores, which indicatemore positive sentiment, are associated with higher ratings. This is a good sign that people are enjoying the show. And for lower ratings like a 2, the blob scores are also lower, some even below zero indicating neutral and negative sentiment.

I was curious so decided to do another one for Severance, which is a show I'm also watching and enjoying.
```{python}
# Specify the TV show you want to filter by
specific_show = 'SEVERANCE'

# Filter the DataFrame for the specific TV show
filtered_df = blob_df[blob_df['TV Show'] == specific_show]

# Create the box plot using the filtered DataFrame
fig2 = px.box(filtered_df, x='Rating', y='blob_score', title=f'Ratings vs Sentiment for {specific_show}')
fig2.show()
```
At lower ratings (e.g., 1 and 2), the box plots are wider and extend into negative blob scores. This suggests a more significant presence of negative sentiments associated with these ratings.
Funny enough, since I'm enjoying the show and would have expected to see more positive sentiment score ranges. 


All this made me curious to see the average rating and sentiment score for each TV show. Let's do that next.
```{python}
df_grouped = df.groupby('TV Show').agg({'Rating': 'mean', 'blob_score': 'mean'}).reset_index()
df_grouped = df_grouped.sort_values(by='Rating', ascending=False)
df_grouped.round(2)
```


## Topic Modeling
Time to shift gears to topic modeling analysis. We will be using the Latent Dirichlet Allocation (LDA) algorithm to identify the topics in the comments and see if there are any patterns.

First, load in all libraries
```{python}
# bertopic is for transformers
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
# joblib is for saving and loading objects
from joblib import load, dump

# gensim is for LDA
import gensim
from gensim.models.coherencemodel import CoherenceModel
# nltk is for cleaning/prep
import nltk
nltk.download('stopwords')
import pprint as pprint
# spacy is for cleaning/prep
import spacy
```

Now we define some functons to remove stopwords to clean the data and we will form bigrams and trigrams to group words that are often together to hopefully obtain a better topic model.
```{python}
stop_words = nltk.corpus.stopwords.words('english')

def sent_to_words(sentences): #takes in a list of sentences
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc = True))  # deacc=True removes punctuations
        # yield is like return, but will return sequences

data_words = list(sent_to_words(df['Review'])) 

#removing stop words:
def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts] 

# Form Bigrams:
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# Form Trigrams:
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


bigram = gensim.models.Phrases(
  # higher threshold fewer phrases.
  data_words, min_count=5, threshold=100) 
  #bigram = occurance of 2 words right by each other 'the dog' 'dog ate' 'ate food'

trigram = gensim.models.Phrases(
  bigram[data_words], threshold=100
)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

nlp = nlp = spacy.load('en_core_web_lg')
```

```{python}
def lemmatization(
  texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    # Loop through each sentence in the input texts
    for sent in texts:
        doc = nlp(" ".join(sent))  # Join the tokens in the sentence into a single string and process it with NLP
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags]) # Append the lemmatized tokens to the output list, filtering by allowed part-of-speech tags
    return texts_out

# Remove stopwords from the data_words list
data_words_nostops = remove_stopwords(data_words)

# Form bigrams from the words that have had stopwords removed
data_words_bigrams = make_bigrams(data_words_nostops)
data_words_trigrams = make_trigrams(data_words_nostops)
# Create a Dictionary mapping each word to a unique ID, for the corpus
id2word = gensim.corpora.Dictionary(data_words_trigrams) 
#corpus = a body/collection of text 

# Set the texts variable to the list of bigrams generated
texts = data_words_trigrams
```

Now we are going to create a bag of words! This is a common way to represent text data. We will use the `doc2bow` method to create it for each document in our corpus. This will give us a list of tuples where the first element is the word id and the second element is the frequency of that word in the document.
```{python}
corpus = [id2word.doc2bow(text) for text in texts]

#Now we can fit the model. We will use the `LdaModel` class from gensim to fit the model. We will use the `corpus` and `id2word` objects that we created earlier. We will also set the number of topics to 5 and the random state to 100. We will also set the `per_word_topics` parameter to `True` so that we can see the topic probabilities for each word in the document.
lda_model = gensim.models.ldamodel.LdaModel(
  corpus=corpus,
  id2word=id2word,
  num_topics=5, # number of topics
  random_state=100, # the seed
  update_every=1, # how often to update estimates
  chunksize=100, # how many docs in each training chunk
  passes=10, # how many rounds
  alpha='auto',
  per_word_topics=True #probability per word of being in a topic
)

dump(
  lda_model, 
  'C:\\Users\\Graduate\\Documents\\ND\\Unstructured_Data_Analytics\\unstructured_data_notes\\lda_model.joblib'
)


# the model will give words and numbers (probabilities of this word occuring) for each topic
pprint.pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```
The words don't seem very related to each other.

Lets visualize the topics using pyLDAvis. This allows us to hover over the circles to see the words associated with each topic and their relative weights. The size of the circles denotes the importance of the topics in the corpus. The closer the circles are, the more similar the topics.
```{python}
import pyLDAvis.gensim_models
import pyLDAvis
import matplotlib.pyplot as plt

# Create the pyLDAvis visualizations
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

# Save the visualization as an HTML file
pyLDAvis.save_html(vis, 'lda_visualization.html')

# To view in Jupyter Notebook:
pyLDAvis.display(vis)
```


Let's try doing this again but with more topics to see if we can get better results. I first changed it to 10, but with no significant improvements I changed it again to 15.
```{python}
lda_model2 = gensim.models.ldamodel.LdaModel(
  corpus=corpus,
  id2word=id2word,
  num_topics=15, # number of topics
  random_state=100, # the seed
  update_every=1, # how often to update estimates
  chunksize=100, # how many docs in each training chunk
  passes=10, # how many rounds
  alpha='auto',
  per_word_topics=True #probability per word of being in a topic
)

dump(
  lda_model2, 
  'C:\\Users\\Graduate\\Documents\\ND\\Unstructured_Data_Analytics\\unstructured_data_notes\\lda_model.joblib'
)


# the model will give words and numbers (probabilities of this word occuring) for each topic
pprint.pprint(lda_model2.print_topics())
doc_lda = lda_model2[corpus]
```

Visualize this one as well:
```{python}
# Create the pyLDAvis visualizations
vis2 = pyLDAvis.gensim_models.prepare(lda_model2, corpus, id2word)

# Save the visualization as an HTML file
pyLDAvis.save_html(vis2, 'lda_visualization.html')

# To view in our notebook:
pyLDAvis.display(vis2)

# If you want to show it inline in notebooks, use this command
#plt.figure(figsize=(12, 8))
#plt.imshow(vis2)
#plt.axis('off')
#plt.show()
```
We can definetly see that how adding more topics seem to make the topics a bit more clear and distinct. 

For instance, topic 1 seems to be about the plot, actors, and the TV show itself, as it has words like: series, story, season, show, plot, characters, episode, acting, and character. 

Topic 2 seems to be about positive reviews, as it has words like: loved, good, like, and best.

Topic 2 seems to be about the production and the cinematography, as it has words like: production, cinematography, visuals, direction, and style.

Topic 6 seems to have many negative comments about the show as it has words like: worst, waste, cold, and downhill.

Topic 8 seems to have even stronger positive wrods likeL fantastic, original, intriguing, and fan.

Topic 15 has words related to violence, like the word violence itself and shots.

## Conclusion
This project was very interesting and fun to do. I was able to gather data from Rotten Tomatoes, clean it, and analyze it. I was able to see the correlation between the ratings and the sentiment scores, and how they are related. I was also able to see the topics in the comments and see what people are talking about. Being able to also dive into specific TV shows that interest me and compare how other people rate it, their sentiment scores. 

This project was also a great way to practice scraping another website. Needing to go step by step, trying to make it work, overcoming the obstacles, until we got to the final result. It is then also very nice to see the results change as the days go by. When I started the project, Paradise was rated as the top 1 show and has gone down to 4. if this were a continuous project we could compare how the sentiment scores and topic models change as new episodes are added to each show.


```{python}
import webbrowser

linkfuck = 'https://foass.1001010.com///london/Seth/Giulia'

#Send a request to the website
response = requests.get(linkfuck)

#Check if the request worked:
if response.status_code == 200:
    #Parse the HTML content with beautifulsoup
    fucksoup = BeautifulSoup(response.text, 'html.parser')

    paragraph = fucksoup.find_all('h1') #'p')[0].text

    for i in paragraph:
        print(i.get_text())
else:
    print(f'Error: Could not retrieve the page. Status code: {response.status_code}')

webbrowser.open(linkfuck)
```
