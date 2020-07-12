# Get data online
if [ -d "data" ]
then
    echo "Data directory already exists. Do nothing!"
else
    mkdir "data"
    cd data
    wget http://xxxxxxxxxxxx/beer_review.zip
    unzip beer_review.zip
    cd ..
fi

# Get the embedding
if [ -d "embeddings" ]
then
    echo "Embeddings directory already exists. Do nothing!"
else
    mkdir "embeddings"
    # get embeddings
    cd embeddings
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
fi
