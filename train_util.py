from typing import List
import os

from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from flair.visual.training_curves import Plotter
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
from flair.models import SequenceTagger
from flair.optim import AdamW
from flair.visual.training_curves import Plotter
# initialize sequence tagger

def train_wrapper_util_single_dataset(data_folder, result_folder, hidden_size, learning_rate = 0.1, max_epochs = 200):
   
    # initialize embeddings. This takes time to load the the first time.
    embedding_types: List[TokenEmbeddings] = [
        # GloVe embeddings for arabic
        WordEmbeddings('ar'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
    # set up the courpus
    
    # 1. get the corpus
    columns = {0: 'text', 1: 'ner'}
    corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns, 
      train_file='train.txt', 
      dev_file='dev.txt')

    # 2. what tag do we want to predict?
    tag_type = 'ner'
    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.item2idx)
    
    # 4.set up the Sequence Tagger
    # have a relatively small hidden_size
    tagger: SequenceTagger = SequenceTagger(hidden_size= hidden_size,
                                            dropout = 0.2,
                                            rnn_layers = 2,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)# initialize trainer
    # 5. set up the trainner.
    trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=AdamW)

    # 6. start training.
    trainer.train(result_folder,
                  EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=learning_rate,
                  mini_batch_size=32,
                  weight_decay = 0.1,
                  max_epochs=max_epochs,
                  checkpoint=False)
    
#     # 7. plot the loss function
#     plotter = Plotter()
#     plotter.plot_training_curves(os.path.join(result_folder, 'loss.tsv'))
    

def train_wrapper_util_cross_val_dataset(data_root_folder, result_root_folder, hidden_size = 64, learning_rate = 0.1, max_epochs = 200, n_set = 5):
    for ii in range(n_set):
    
        data_folder = os.path.join(data_root_folder, 'data_{}'.format(ii))
        result_folder = os.path.join(result_root_folder, 'data_{}'.format(ii))

        train_wrapper_util_single_dataset(data_folder, result_folder, hidden_size, learning_rate = learning_rate, max_epochs = max_epochs)
        