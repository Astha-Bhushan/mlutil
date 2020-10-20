# Collection of Useful Machine Learning workflow methods, and viz tools.
# such as grid search, full viz,specific plotting mechanisms etc.
#

# For Grid Search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# For plotting
import matplotlib.pyplot as plt


def hyp_pipeline(self, X_train, X_test, y_train, y_test,
                 model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                 do_probabilities=False, search_mode='GridSearchCV', n_iterations=0, n_jobs=-1):
   '''Hyper Parameter searching with GridSearch and RandomSearch'''

   fitted_model = None

   if (search_mode == 'GridSearchCV'):
      gs = GridSearchCV(
         estimator=model,
         param_grid=param_grid,
         cv=cv,
         n_jobs=n_jobs,
         scoring=scoring_fit,
         verbose=1)
      fitted_model = gs.fit(X_train, y_train)
   elif (search_mode == 'RandomizedSearchCV'):
      rs = RandomizedSearchCV(
         estimator=model,
         param_distributions=param_grid,
         cv=cv,
         n_iter=n_iterations,
         n_jobs=n_jobs,
         scoring=scoring_fit,
         verbose=1)
      fitted_model = rs.fit(X_train, y_train)

   if (fitted_model != None):
      if do_probabilities:
         pred = fitted_model.predict_proba(X_test)
      else:
         pred = fitted_model.predict(X_test)

   return fitted_model, pred


def plot_history(self, history):
   ''' Plot accuracy and loss of keras Models '''
   loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
   val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
   acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
   val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

   if len(loss_list) == 0:
      print('Loss is missing in history')
      return

      # As loss always exists
   epochs = range(1, len(history.history[loss_list[0]]) + 1)

   # Loss
   plt.figure(1)
   for l in loss_list:
      plt.plot(epochs, history.history[l], 'b',
               label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
   for l in val_loss_list:
      plt.plot(epochs, history.history[l], 'g',
               label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

   plt.title('Loss')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()

   # Accuracy
   plt.figure(2)
   for l in acc_list:
      plt.plot(epochs, history.history[l], 'b',
               label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
   for l in val_acc_list:
      plt.plot(epochs, history.history[l], 'g',
               label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

   plt.title('Accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.show()


# Utility function to train the Pytorch model


# model = torch.nn.Linear(3, 2)
# loss_fn = torch.nn.functional.mse_loss
# opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# from torch.utils.data import DataLoader,TensorDataSet
# train_dl= DataLoader(TensorDataset(inputs, targets),batch_size,shuffle=True)

def pytorch_fit(num_epochs, model, loss_fn, opt, train_dl):
   # Repeat for given number of epochs
   for epoch in range(num_epochs):

      # Train with batches of data
      for xb, yb in train_dl:
         # 1. Generate predictions
         pred = model(xb)

         # 2. Calculate loss
         loss = loss_fn(pred, yb)

         # 3. Compute gradients
         loss.backward()

         # 4. Update parameters using gradients
         opt.step()

         # 5. Reset the gradients to zero
         opt.zero_grad()

      # Print the progress
      if (epoch + 1) % 10 == 0:
         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# For NLP Preprocessing
import string
string.punctuation
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

#Stop words are words like “and”, “the”, “him”, which are presumed to be uninformative & non-predicative , so avoided
stopwords.words("english")[100:110]                   # to see some of the stopwords


def remove_punctuation_and_stopwords(sms):
   sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
   sms_no_punctuation = "".join(sms_no_punctuation).split()

   sms_no_punctuation_no_stopwords = \
      [word.lower() for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]

   return sms_no_punctuation_no_stopwords

data['text'].apply(remove_punctuation_and_stopwords).head()

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer = remove_punctuation_and_stopwords).fit(data['text'])
print(len(bow_transformer.vocabulary_))
# It will print total no of words we have in our vocalbury
# Applying vocalbury to all words , here vocalbury = bow-transformer
bow_data = bow_transformer.transform(data['text'])
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(bow_data)

# To check the importanct of words from a ham label sms .
sample_ham = data['text'][4]
bow_sample_ham = bow_transformer.transform([sample_ham])
tfidf_sample_ham = tfidf_transformer.transform(bow_sample_ham)
print(tfidf_sample_ham)
