#import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')
print("Imported the modules.")

#load the datatest
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

#scale the labels
scale_factor = 1000.0

#scale the training set's label
train_df["median_house_value"] /= scale_factor

#scale the test set's label
test_df["median_house_value"] /= scale_factor

#shuffle the examples
train_df = train_df.reindex(np.random.permutation(train_df.index))

#create an empty list of features
feature_columns = []

#for latitude column
latitude = tf.feature_column.numeric_column("latitude")
feature_columns.append(latitude)

#for longitude column
longitude = tf.feature_column.numeric_column("longitude")
feature_columns.append(longitude)

#convert features column into layers
fp_feature_layer = layers.DenseFeatures(feature_columns)

#define functions to create and train models
def create_model(my_learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, dataset, epochs, batch_size, label_name):
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    epochs = history.epoch
    hist = pd. DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs,rmse


def plot_the_loss_curve(epochs, rmse):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="loss")
    plt.legend()
    plt.ylim([rmse.min()*0.94, rmse.max()*1.05])
    plt.show()
print("Defined the create_model, traing_model, and plot the loss curve functions")

#hyperparameters
learning_rate = 0.05
epochs = 30
batch_size = 100
label_name = 'median_house_value'

my_model = create_model(learning_rate, fp_feature_layer)

#traing the model in training set
epochs,rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)
print("\n: Evaluate the new model against the test set")
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)



resolution_in_degree = 1.0
feature_columns = []

#create a bucket feature column for latitude
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arrange(int (min(train_df["latitude"])),
                                      int (max(train_df["latitude"])),
                                           resolution_in_degree))

latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                               latitude_boundaries)
feature_columns.append(latitude)

#create a bucket feature column for longitude
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arrange(int (min(train_df["longitude"])),
                                      int (max(train_df["longitude"])),
                                           resolution_in_degree))

longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                               longitude_boundaries)
feature_columns.append(longitude)

#convert columns into layers
bucket_feature_layer = layers.DenseFeatures(feature_columns)

#new hyperparmeters
learning_rate = 0.04
epochs =35

#build models
my_model = create_model(learning_rate, bbucket_feature_layer)

epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set")
my_model.evaluate(x=test_features, y=test_label, batch_size= batch_size)






















