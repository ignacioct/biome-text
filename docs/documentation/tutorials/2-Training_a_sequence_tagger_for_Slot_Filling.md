# Training a sequence tagger for Slot Filling

<a target="_blank" href="https://www.recogn.ai/biome-text/documentation/tutorials/2-Training_a_sequence_tagger_for_Slot_Filling.html"><img class="icon" src="https://www.recogn.ai/biome-text/assets/img/biome-isotype.svg" width=24 /></a>
[View on recogn.ai](https://www.recogn.ai/biome-text/documentation/tutorials/2-Training_a_sequence_tagger_for_Slot_Filling.html)
    
<a target="_blank" href="https://colab.research.google.com/github/recognai/biome-text/blob/master/docs/docs/documentation/tutorials/2-Training_a_sequence_tagger_for_Slot_Filling.ipynb"><img class="icon" src="https://www.tensorflow.org/images/colab_logo_32px.png" width=24 /></a>
[Run in Google Colab](https://colab.research.google.com/github/recognai/biome-text/blob/master/docs/docs/documentation/tutorials/2-Training_a_sequence_tagger_for_Slot_Filling.ipynb)
        
<a target="_blank" href="https://github.com/recognai/biome-text/blob/master/docs/docs/documentation/tutorials/2-Training_a_sequence_tagger_for_Slot_Filling.ipynb"><img class="icon" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width=24 /></a>
[View source on GitHub](https://github.com/recognai/biome-text/blob/master/docs/docs/documentation/tutorials/2-Training_a_sequence_tagger_for_Slot_Filling.ipynb)

In this tutorial we will train a sequence tagger for filling slots in spoken requests.
The goal is to look for specific pieces of information in the request and tag the corresponding tokens accordingly. 
The requests will include several intents, from getting weather information to adding a song to a playlist, each requiring its own set of slots.
Therefore, slot filling often goes hand in hand with intent classification.
In this tutorial, however, we will only focus on the slot filling task.

Slot filling is closely related to [Named-entity recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) and the model of this tutorial can also be used to train a NER system.

In this tutorial we will use the [SNIPS data set](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines) adapted by [Su Zhu](https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/tree/master/data/snips) and our simple [data preparation notebook](https://biome-tutorials-data.s3-eu-west-1.amazonaws.com/token_classifier/data_prep.ipynb).

When running this tutorial in Google Colab, make sure to install *biome.text* first:


```python
!pip install -U pip
!pip install -U git+https://github.com/recognai/biome-text.git
```

Ignore warnings and don't forget to restart your runtime afterwards (*Runtime -> Restart runtime*).

*If* you want to log your runs with [WandB](https://wandb.ai/home), don't forget to install its client and log in.


```python
!pip install wandb
!wandb login
```

## Imports

Let us first import all the stuff we need for this tutorial.


```python
from biome.text import Pipeline, Dataset, PipelineConfiguration, VocabularyConfiguration
from biome.text.configuration import FeaturesConfiguration, WordFeatures, CharFeatures
from biome.text.modules.configuration import Seq2SeqEncoderConfiguration
from biome.text.modules.heads import TokenClassificationConfiguration
```

## Explore the data

Let's take a look at the data before starting with the configuration of our pipeline.
For this we create a `Dataset` instance providing a path to our downloaded data.


```python
!curl -O https://biome-tutorials-data.s3-eu-west-1.amazonaws.com/token_classifier/train.json
!curl -O https://biome-tutorials-data.s3-eu-west-1.amazonaws.com/token_classifier/valid.json
!curl -O https://biome-tutorials-data.s3-eu-west-1.amazonaws.com/token_classifier/test.json
```


```python
train_ds = Dataset.from_json("train.json")
valid_ds = Dataset.from_json("valid.json")
test_ds = Dataset.from_json("test.json")
```

The [Dataset](https://www.recogn.ai/biome-text/api/biome/text/dataset.html#dataset) class is a very thin wrapper around HuggingFace's awesome [datasets.Dataset](https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset).
Most of HuggingFace's `Dataset` API is exposed and you can checkout their nice [documentation](https://huggingface.co/docs/datasets/master/processing.html) on how to work with data in a `Dataset`. For example, let's quickly check the size of our training data and print the first 10 examples as a pandas dataframe:


```python
print("Training data size:", len(train_ds))

with train_ds.formatted_as("pandas"):
    display(train_ds[:10])
```

    Training data size: 13084



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>labels</th>
      <th>intent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Find, the, schedule, for, Across, the, Line, ...</td>
      <td>[O, O, B-object_type, O, B-movie_name, I-movie...</td>
      <td>SearchScreeningEvent</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[play, Party, Ben, on, Slacker]</td>
      <td>[O, B-artist, I-artist, O, B-service]</td>
      <td>PlayMusic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[play, a, 1988, soundtrack]</td>
      <td>[O, O, B-year, B-music_item]</td>
      <td>PlayMusic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Can, you, play, The, Change, Is, Made, on, Ne...</td>
      <td>[O, O, O, B-track, I-track, I-track, I-track, ...</td>
      <td>PlayMusic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[what, is, the, forecast, for, colder, in, Ans...</td>
      <td>[O, O, O, O, O, B-condition_temperature, O, B-...</td>
      <td>GetWeather</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[What's, the, weather, in, Totowa, WY, one, mi...</td>
      <td>[O, O, O, O, B-city, B-state, B-timeRange, I-t...</td>
      <td>GetWeather</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[Play, a, tune, from, Space, Mandino, .]</td>
      <td>[O, O, B-music_item, O, B-artist, I-artist, O]</td>
      <td>PlayMusic</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[give, five, out, of, 6, stars, to, current, e...</td>
      <td>[O, B-rating_value, O, O, B-best_rating, B-rat...</td>
      <td>RateBook</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[Play, some, chanson, style, music.]</td>
      <td>[O, O, B-genre, O, O]</td>
      <td>PlayMusic</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[I, would, give, French, Poets, and, Novelists...</td>
      <td>[O, O, O, B-object_name, I-object_name, I-obje...</td>
      <td>RateBook</td>
    </tr>
  </tbody>
</table>
</div>


As we can see we have two relevant columns for our task: *text* and *labels*. 
The *intent* column will be ignored in this tutorial. 

The text input already comes pre-tokenized as a list of strings and each token in the *text* column has a label/tag in the *labels* column, this means that both list always have the same length.
The labels are given in the [BIO tagging scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)), which is widely used in Slot Filling/NER systems.

We can quickly check how many different labels there are in our dataset:


```python
labels = {tag[2:] for tags in train_ds["labels"] for tag in tags if tag != "O"}
print("number of lables:", len(labels))
labels
```

    number of lables: 39





    {'album',
     'artist',
     'best_rating',
     'city',
     'condition_description',
     'condition_temperature',
     'country',
     'cuisine',
     'current_location',
     'entity_name',
     'facility',
     'genre',
     'geographic_poi',
     'location_name',
     'movie_name',
     'movie_type',
     'music_item',
     'object_location_type',
     'object_name',
     'object_part_of_series_type',
     'object_select',
     'object_type',
     'party_size_description',
     'party_size_number',
     'playlist',
     'playlist_owner',
     'poi',
     'rating_unit',
     'rating_value',
     'restaurant_name',
     'restaurant_type',
     'served_dish',
     'service',
     'sort',
     'spatial_relation',
     'state',
     'timeRange',
     'track',
     'year'}



Since the the [TaskHead](https://www.recogn.ai/biome-text/api/biome/text/modules/heads/task_head.html#taskhead) of our model (the [TokenClassification](https://www.recogn.ai/biome-text/api/biome/text/modules/heads/token_classification.html#tokenclassification) head) expects a *text* and a *tags* column to be present in the Dataset, we need to rename the *labels* column:


```python
for ds in [train_ds, valid_ds, test_ds]:
     ds.rename_column_("labels", "tags")
```

::: tip Tip

The [TokenClassification](https://www.recogn.ai/biome-text/api/biome/text/modules/heads/token_classification.html#tokenclassification) head also supports a *entities* column instead of a *tags* column, in which case the entities have to be a list of python dictionaries with a `start`, `end` and `label` key that correspond to the char indexes and the label of the entity, respectively. 

:::

## Configure your *biome.text* Pipeline

A typical [Pipeline](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#pipeline) consists of tokenizing the input, extracting features, applying a language encoding (optionally) and executing a task-specific head in the end.
After training a pipeline, you can use it to make predictions or explore the underlying model via the explore UI.

A *biome.text* pipeline has the following main components:

```yaml
name: # a descriptive name of your pipeline

tokenizer: # how to tokenize the input

features: # input features of the model

encoder: # the language encoder

head: # your task configuration

```

See the [Configuration section](https://www.recogn.ai/biome-text/documentation/user-guides/2-configuration.html) for a detailed description of how these main components can be configured.

In this tutorial we will create a [PipelineConfiguration](https://www.recogn.ai/biome-text/api/biome/text/configuration.html#pipelineconfiguration) programmatically, and use it to initialize the [Pipeline](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#pipeline).
You can also create your pipelines by providing a [python dictionary](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#from-config) (see the text classification [tutorial](https://www.recogn.ai/biome-text/documentation/tutorials/1-Training_a_text_classifier.html)), a YAML [configuration file](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#from-yaml) or a [pretrained model](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#from-pretrained).

A pipeline configuration is composed of several other [configuration classes](https://www.recogn.ai/biome-text/api/biome/text/configuration.html#biome-text-configuration), each one corresponding to one of the main components.

### Features

Let us first configure the features of our pipeline.
For our `word` feature we will use pretrained embeddings from [fasttext](https://fasttext.cc/docs/en/english-vectors.html), and our `char` feature will use the last hidden state of a [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) encoder to represent a word based on its characters.
Keep in mind that the `embedding_dim` parameter for the `word` feature must be equal to the dimensions of the pretrained embeddings!

::: tip Tip

If you do not provide any feature configurations, we will choose a very basic `word` feature by default.

:::


```python
word_feature = WordFeatures(
    embedding_dim=300,
    weights_file="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
)

char_feature = CharFeatures(
    embedding_dim=32,
    encoder={
        "type": "gru",
        "bidirectional": True,
        "num_layers": 1,
        "hidden_size": 32,
    },
    dropout=0.1
)

features_config = FeaturesConfiguration(
    word=word_feature, 
    char=char_feature
)
```

### Encoder

Next we will configure our encoder that takes as input a sequence of embedded word vectors and returns a sequence of encoded word vectors.
For this encoding we will use another larger GRU:


```python
encoder_config = Seq2SeqEncoderConfiguration(
    type="gru",
    bidirectional=True,
    num_layers=1,
    hidden_size=128,
)
```

### Head

The final configuration belongs to our [TaskHead](https://www.recogn.ai/biome-text/api/biome/text/modules/heads/task_head.html#taskhead).
It reflects the task our problem belongs to and can be easily exchanged with other types of heads keeping the same features and encoder.

::: tip Tip

Exchanging the heads you can easily pretrain a model on a certain task, such as [language modelling](https://www.recogn.ai/biome-text/api/biome/text/modules/heads/language_modelling.html#languagemodelling), and use its pretrained features and encoder for training the model on another task.

:::

For our task we will use a [TokenClassification](https://www.recogn.ai/biome-text/api/biome/text/modules/heads/token_classification.html#tokenclassification) head that allows us to tag each token individually:


```python
head_config = TokenClassificationConfiguration(
    labels=list(labels),
    label_encoding="BIO",
    top_k=1,
    feedforward={
        "num_layers": 1,
        "hidden_dims": [128],
        "activations": ["relu"],
        "dropout": [0.1],
    },
)
```

### Pipeline

Now we can create a [PipelineConfiguration](https://www.recogn.ai/biome-text/api/biome/text/configuration.html#pipelineconfiguration) and finally initialize our [Pipeline](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#pipeline).


```python
pipeline_config = PipelineConfiguration(
    name="slot_filling_tutorial",
    features=features_config,
    encoder=encoder_config,
    head=head_config,
)
```


```python
pl = Pipeline.from_config(pipeline_config)
```

## Create a vocabulary

Before we can start the training we need to create the vocabulary for our model.
Since we use pretrained word embeddings we will not only consider the training data, but also the validation data when creating the vocabulary. 
In addition, we get rid of the rarest words by adding the `min_count` argument and set it to 2 for the word feature vocabulary.
For a complete list of available arguments see the [VocabularyConfiguration API](https://www.recogn.ai/biome-text/api/biome/text/configuration.html#vocabularyconfiguration).


```python
vocab_config = VocabularyConfiguration(
    sources=[train_ds, valid_ds],
    min_count={"word": 2},
)
```

We then pass this configuration to our `Pipeline` to create the vocabulary.
Apart from the loading bar of building the vocabulary, there will be two more loading bars corresponding to the `weights_file` provided in the word feature:
 - the progress of downloading the file (this file will be cached)
 - the progress loading the weights from the file 


```python
pl.create_vocabulary(vocab_config)
```

After creating the vovocab_configbulary we can check the size of our entire model in terms of trainable parameters:


```python
pl.num_trainable_parameters
```




    1991086



## Train your model

Now we have everything ready to start the training of our model:
- training data set
- vocabulary

As `trainer` we will use the default configuration that has sensible values and works alright for our experiment.
[This tutorial](https://www.recogn.ai/biome-text/documentation/tutorials/1-Training_a_text_classifier.html) shows you an example of how to configure a trainer. 

::: tip Tip

By default we will automatically use a CUDA device if available. If you want to tune the trainer or specifically not use a CUDA device, you can pass a `trainer = TrainerConfiguration(cuda_device=-1, ...)` to the `Pipeline.train()` method. 
See the [TrainerConfiguration API](https://www.recogn.ai/biome-text/api/biome/text/configuration.html#trainerconfiguration) for a complete list of available configurations.

:::

The training output will be saved in a folder specified by the `output` argument of the `train` method. 
It will contain the trained model weights and the metrics, as well as the vocabulary and a *log* folder for visualizing the training process with [tensorboard](https://www.tensorflow.org/tensorboard/).

Apart from the validation data source to estimate the generalization error, we will also pass in a test data set in case we want to do some Hyperparameter optimization and compare different encoder architectures in the end. 

When the training has finished it will automatically make a pass over the test data with the best weights to gather the test metrics.


```python
pl.train(
    output="output",
    training=train_ds,
    validation=valid_ds,
    test=test_ds,
)
```

The model above achieves an overall F1 score of around **0.95**, which is not bad when compared to [published results](https://nlpprogress.com/english/intent_detection_slot_filling.html) of the same data set.
You could continue the experiment changing the encoder to an LSTM network, try out a transformer architecture or fine tune the trainer.
But for now we will go on and make our first predictions with this trained model.

## Make your first predictions

Now that we trained our model we can go on to make our first predictions.
First we must load our trained model into a new `Pipeline`:


```python
pl_trained = Pipeline.from_pretrained("output/model.tar.gz")
```

We then provide the input expected by our `TaskHead` to the `Pipeline.predict()` method.
In our case it is a `TokenClassification` head that classifies a `text` input. 

You can either provide pretokenized tokens (list of strings) **or** a raw string to the `predict` method. In the first case, you should make sure that those tokens were tokenized the same way the training data was tokenized, in the latter case you should make sure that the pipeline uses the same tokenizer as was used for the training data.

The prediction of the `TokenClassification` head will always consist of a `tags` and `entities` key. Both keys will include the `top_k` most likely tag/entity sequences for the input, where `top_k` is a parameter specified in the `TokenClassificationConfiguration` before the training.

### pretokenized input 

For pretokenized input, the `entities` key of the output holds dictionaries with the `start_token` id, `end_token` id and the label of the entity: 


```python
text = "Can you play biome text by backstreet recognais on Spotify ?".split()
prediction = pl_trained.predict(text=text)

print("Predicted tags:\n", list(zip(text, prediction["tags"][0])))
print("Predicted entities:\n", prediction["entities"][0])
```

    Predicted tags:
     [('Can', 'O'), ('you', 'O'), ('play', 'O'), ('biome', 'B-album'), ('text', 'I-album'), ('by', 'O'), ('backstreet', 'B-artist'), ('recognais', 'I-artist'), ('on', 'O'), ('Spotify', 'B-service'), ('?', 'O')]
    Predicted entities:
     [{'start_token': 3, 'end_token': 5, 'label': 'album'}, {'start_token': 6, 'end_token': 8, 'label': 'artist'}, {'start_token': 9, 'end_token': 10, 'label': 'service'}]


### string input

For a raw string input, the `entities` key of the output holds dictionaries with the `start_token` id, `end_token` id, `start` char id, `end` char id and the label of the entity:


```python
text = "Can you play biome text by backstreet recognais on Spotify ?"
prediction = pl_trained.predict(text=text)

print("Predicted tags:\n", list(zip(text.split(), prediction["tags"][0])))
print("Predicted entities:\n", prediction["entities"][0])
```

    Predicted tags:
     [('Can', 'O'), ('you', 'O'), ('play', 'O'), ('biome', 'B-album'), ('text', 'I-album'), ('by', 'O'), ('backstreet', 'B-artist'), ('recognais', 'I-artist'), ('on', 'O'), ('Spotify', 'B-service'), ('?', 'O')]
    Predicted entities:
     [{'start_token': 3, 'end_token': 5, 'label': 'album', 'start': 13, 'end': 23}, {'start_token': 6, 'end_token': 8, 'label': 'artist', 'start': 27, 'end': 47}, {'start_token': 9, 'end_token': 10, 'label': 'service', 'start': 51, 'end': 58}]

