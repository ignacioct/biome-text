# Training a short text classifier of German business names

<a target="_blank" href="https://www.recogn.ai/biome-text/documentation/tutorials/1-Training_a_text_classifier.html"><img class="icon" src="https://www.recogn.ai/biome-text/assets/img/biome-isotype.svg" width=24 /></a>
[View on recogn.ai](https://www.recogn.ai/biome-text/documentation/tutorials/1-Training_a_text_classifier.html)

<a target="_blank" href="https://colab.research.google.com/github/recognai/biome-text/blob/master/docs/docs/documentation/tutorials/1-Training_a_text_classifier.ipynb"><img class="icon" src="https://www.tensorflow.org/images/colab_logo_32px.png" width=24 /></a>
[Run in Google Colab](https://colab.research.google.com/github/recognai/biome-text/blob/master/docs/docs/documentation/tutorials/1-Training_a_text_classifier.ipynb)

<a target="_blank" href="https://github.com/recognai/biome-text/blob/master/docs/docs/documentation/tutorials/1-Training_a_text_classifier.ipynb"><img class="icon" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width=24 /></a>
[View source on GitHub](https://github.com/recognai/biome-text/blob/master/docs/docs/documentation/tutorials/1-Training_a_text_classifier.ipynb)

Test: In this tutorial we will train a basic short-text classifier for predicting the sector of a business based only on its business name. 
For this we will use a training dataset with business names and business categories in German.

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

## Explore the training data

Let's take a look at the data we will use for training. For this we will use the [`Dataset`](https://www.recogn.ai/biome-text/api/biome/text/dataset.html#dataset) class that is a very thin wrapper around HuggingFace's awesome [datasets.Dataset](https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset). We will download the data first and create a `Dataset` instance with it.


```python
from biome.text import Dataset
```


```python
# Downloading the dataset first
!curl -O https://biome-tutorials-data.s3-eu-west-1.amazonaws.com/text_classifier/business.cat.train.csv

# Loading from local
train_ds = Dataset.from_csv("business.cat.train.csv")
```

Most of HuggingFace's `Dataset` API is exposed and you can checkout their nice [documentation](https://huggingface.co/docs/datasets/master/processing.html) on how to work with data in a `Dataset`. For example, let's quickly check the size of our training data and print the first 10 examples as a pandas dataframe:


```python
print(len(train_ds))
```

    8000



```python
with train_ds.formatted_as("pandas"):
    display(train_ds[:10])
```


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
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Edv</td>
      <td>Cse Gmbh Computer Edv-service Bürobedarf</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Maler</td>
      <td>Malerfachbetrieb U. Nee</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gebrauchtwagen</td>
      <td>Sippl Automobilverkäufer Hausmann</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Handelsvermittler Und -vertreter</td>
      <td>Strenge Handelsagentur Werth</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gebrauchtwagen</td>
      <td>Dzengel Autohaus Gordemitz Rusch</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Apotheken</td>
      <td>Schinkel-apotheke Bitzer</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Tiefbau</td>
      <td>Franz Möbius Mehrings-bau-hude Und Stigge</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Handelsvermittler Und -vertreter</td>
      <td>Kontze Hdl.vertr. Lau</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Autowerkstätten</td>
      <td>Keßler Kfz-handel</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Gebrauchtwagen</td>
      <td>Diko Lack Und Schrift Betriebsteil Der Autocen...</td>
    </tr>
  </tbody>
</table>
</div>


As we can see we have two relevant columns *label* and *text*. Our classifier will be trained to predict the *label* given the *text*.

::: tip Tip

The [TaskHead](https://www.recogn.ai/biome-text/api/biome/text/modules/heads/task_head.html#taskhead) of our model below will expect a *text* and a *label* column to be present in the `Dataset`. In our dataset this is already the case, otherwise we would need to change or map the corresponding column names via `Dataset.rename_column_()` or `Dataset.map()`.

:::

We can also quickly check the distibution of our labels:


```python
with train_ds.formatted_as("pandas"):
    display(train_ds["label"].value_counts())
```


    Unternehmensberatungen              632
    Friseure                            564
    Tiefbau                             508
    Dienstleistungen                    503
    Gebrauchtwagen                      449
    Elektriker                          430
    Restaurants                         422
    Architekturbüros                    417
    Vereine                             384
    Versicherungsvermittler             358
    Maler                               330
    Sanitärinstallationen               323
    Edv                                 318
    Werbeagenturen                      294
    Apotheken                           289
    Physiotherapie                      286
    Vermittlungen                       277
    Hotels                              274
    Autowerkstätten                     263
    Elektrotechnik                      261
    Allgemeinärzte                      216
    Handelsvermittler Und -vertreter    202
    Name: label, dtype: int64


The `Dataset` class also provides access to Hugging Face's extensive NLP datasets collection via the `Dataset.load_dataset()` method. Have a look at their [quicktour](https://huggingface.co/docs/datasets/master/quicktour.html) for more details about their awesome library.

## Configure your *biome.text* Pipeline

A typical [Pipeline](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#pipeline) consists of tokenizing the input, extracting features, applying a language encoding (optionally) and executing a task-specific head in the end.

After training a pipeline, you can use it to make predictions or explore the underlying model via the explore UI.

As a first step we must define a configuration for our pipeline. 
In this tutorial we will create a configuration dictionary and use the `Pipeline.from_config()` method to create our pipeline, but there are [other ways](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#pipeline).

A *biome.text* pipeline has the following main components:

```yaml
name: # a descriptive name of your pipeline

tokenizer: # how to tokenize the input

features: # input features of the model

encoder: # the language encoder

head: # your task configuration

```

See the [Configuration section](https://www.recogn.ai/biome-text/documentation/user-guides/2-configuration.html) for a detailed description of how these main components can be configured.

Our complete configuration for this tutorial will be following:


```python
pipeline_dict = {
    "name": "german_business_names",
    "tokenizer": {
        "text_cleaning": {
            "rules": ["strip_spaces"]
        }
    },
    "features": {
        "word": {
            "embedding_dim": 64,
            "lowercase_tokens": True,
        },
        "char": {
            "embedding_dim": 32,
            "lowercase_characters": True,
            "encoder": {
                "type": "gru",
                "num_layers": 1,
                "hidden_size": 32,
                "bidirectional": True,
            },
            "dropout": 0.1,
        },
    },
    "head": {
        "type": "TextClassification",
        "labels": train_ds.unique("label"),
        "pooler": {
            "type": "gru",
            "num_layers": 1,
            "hidden_size": 32,
            "bidirectional": True,
        },
        "feedforward": {
            "num_layers": 1,
            "hidden_dims": [32],
            "activations": ["relu"],
            "dropout": [0.0],
        },
    },       
}
```

With this dictionary we can now create a `Pipeline`:


```python
from biome.text import Pipeline
```


```python
pl = Pipeline.from_config(pipeline_dict)
```

## Create a vocabulary

Before we can start the training we need to create the vocabulary for our model.
For this we define a `VocabularyConfiguration`.

In our business name classifier we only want to include words with a general meaning to our word feature vocabulary (like "Computer" or "Autohaus", for example), and want to exclude specific names that will not help to generally classify the kind of business.
This can be achieved by including only the most frequent words in our training set via the `min_count` argument. For a complete list of available arguments see the [VocabularyConfiguration API](https://www.recogn.ai/biome-text/api/biome/text/configuration.html#vocabularyconfiguration).


```python
from biome.text.configuration import VocabularyConfiguration, WordFeatures
```


```python
vocab_config = VocabularyConfiguration(sources=[train_ds], min_count={WordFeatures.namespace: 20})
```

We then pass this configuration to our `Pipeline` to create the vocabulary:


```python
pl.create_vocabulary(vocab_config)
```

After creating the vocabulary we can check the size of our entire model in terms of trainable parameters:


```python
pl.num_trainable_parameters
```




    60566



## Configure the trainer

As a next step we have to configure the *trainer*.

The default trainer has sensible defaults and should work alright for most of your cases.
In this tutorial, however, we want to tune a bit the learning rate and limit the training time to three epochs only.
For a complete list of available arguments see the [TrainerConfiguration API](https://www.recogn.ai/biome-text/api/biome/text/configuration.html#trainerconfiguration).

::: tip Tip

By default we will use a CUDA device if one is available. If you have several devices or prefer not to use it, you can specify it here via the `cuda_device` argument.

:::


```python
from biome.text.configuration import TrainerConfiguration
```


```python
trainer_config = TrainerConfiguration(
    optimizer={
        "type": "adam",
        "lr": 0.01,
    },
    num_epochs=3,
)
```

## Train your model

Now we have everything ready to start the training of our model:
- training data set
- vocabulary
- trainer

Optionally we can provide a validation data set to estimate the generalization error.
For this we will create another `Dataset` pointing to our validation data.


```python
!curl -O https://biome-tutorials-data.s3-eu-west-1.amazonaws.com/text_classifier/business.cat.valid.csv

valid_ds = Dataset.from_csv("business.cat.valid.csv")
```

The training output will be saved in a folder specified by the `output` argument. It contains the trained model weights and the metrics, as well as the vocabulary and a *log* folder for visualizing the training process with [tensorboard](https://www.tensorflow.org/tensorboard/).


```python
pl.train(
    output="output",
    training=train_ds,
    validation=valid_ds,
    trainer=trainer_config,
)
```

After 3 epochs we achieve a validation accuracy of about 0.91.
The validation loss seems to be decreasing further, though, so we could probably train the model for a few more epochs without the risk of overfitting.

::: tip Tip

If for some reason the training gets interrupted, you can continue where you left off by setting the `restore` argument in the `Pipeline.train()` method to `True`. 
If you want to train your model for a few more epochs, you can also use the `restore` argument, but you have to modify the `epochs` argument in your `TrainerConfiguration` to reflect the total amount of epochs you aim for.

:::

## Make your first predictions

Now that we trained our model we can go on to make our first predictions.
First we must load our trained model into a new `Pipeline`:


```python
pl_trained = Pipeline.from_pretrained("output/model.tar.gz")
```

We then provide the input expected by our `TaskHead` of the model to the `Pipeline.predict()` method.
In our case it is a `TextClassification` head that classifies a `text` input:


```python
pl_trained.predict(text="Autohaus biome.text")
```

The returned dictionary contains the logits and probabilities of all labels (classes).
The label with the highest probability is stored under the `label` key, together with its probability under the `prob` key.

::: tip Tip

When configuring the pipeline in the first place, we recommend to check that it is correctly setup by using the `predict` method.
Since the pipeline is still not trained at that moment, the predictions will be arbitrary.

:::

## Explore the model's predictions

To check and understand the predictions of the model, we can use the **biome.text explore UI**.
Just calling the [Pipeline.predict](https://www.recogn.ai/biome-text/api/biome/text/pipeline.html#predict) method will open the UI in the output of our cell.
We will set the `explain` argument to true, which automatically visualizes the attribution of each token by means of [integrated gradients](https://arxiv.org/abs/1703.01365).


::: warning Warning

For the UI to work you need a running [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) instance.
We recommend installing [Elasticsearch with docker](https://www.elastic.co/guide/en/elasticsearch/reference/7.7/docker.html#docker-cli-run-dev-mode).

:::


```python
from biome.text import explore

explore.create(pl_trained, valid_ds, explain=True)
```

![Screenshot of the biome.text explore UI](./img/text_classifier_explore_screenshot.png)
*Screenshot of the biome.text explore UI*

Exploring our model we could take advantage of the F1 scores of each label to figure out which labels to prioritize when gathering new training data.
For example, although "Allgemeinärzte" is the second rarest label in our training data, it still seems relatively easy to classify for our model due to the distinctive words "Dr." and "Allgemeinmedizin".
