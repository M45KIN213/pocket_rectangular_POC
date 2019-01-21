from random import randint
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

a = input('Nb of cycles ?')
try:
    a = int(a)
except:
    a = 1
if a > 1:
    plt.ioff()


def display_metrics(history_op, history_strat, history_float):
    fig = plt.figure(figsize=[30, 20])
    gs = gridspec.GridSpec(nrows = 3, ncols = 3)

    list_keys = []
    for i in history_op.history.keys():
        list_keys.append(i)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(history_op.history[list_keys[4]])
    ax0.plot(history_op.history[list_keys[5]])
    ax0.legend(['op_type', 'quality'], loc='upper left')
    ax0.set_title('Metrics on train set')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(history_op.history[list_keys[1]])
    ax1.plot(history_op.history[list_keys[2]])
    ax1.legend(['val_op_type',  'val_quality'], loc='upper left')
    ax1.set_title('Metrics on validation set')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(history_op.history[list_keys[3]])
    ax2.plot(history_op.history[list_keys[0]])
    ax2.legend(['loss', 'val_loss'], loc='upper left')
    ax2.set_title('OP Losses')

    list_keys = []
    for i in history_strat.history.keys():
        list_keys.append(i)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history_strat.history[list_keys[1]])
    ax5.plot(history_strat.history[list_keys[0]])
    ax5.legend(['loss', 'val_loss'], loc='upper left')
    ax5.set_title('Strat Losses')

    list_keys = []
    for i in history_float.history.keys():
        list_keys.append(i)

    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(history_float.history[list_keys[4]])
    ax6.plot(history_float.history[list_keys[5]])
    ax6.legend(['Allowances', 'Tool cond'], loc='upper left')
    ax6.set_title('Metrics on train set')

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(history_float.history[list_keys[1]])
    ax7.plot(history_float.history[list_keys[2]])
    ax7.legend(['val_allow',  'val_tool_cond'], loc='upper left')
    ax7.set_title('Metrics on validation set')

    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(history_float.history[list_keys[3]])
    ax8.plot(history_float.history[list_keys[0]])
    ax8.legend(['loss', 'val_loss'], loc='upper left')
    ax8.set_title('Allow & Tool Cond Losses')

    plt.tight_layout()

    return fig

def normalize_pocket_desc(dataset):
    dataset["l"] = dataset["l"]/dataset["L"]
    dataset["h"] = dataset["h"]/dataset["L"]

    dataset["Rcoin_act"] = dataset["Rcoin_act"] / dataset["L"]
    dataset["Rbot_act"] = dataset["Rbot_act"] / dataset["L"]

    dataset["Rcoin"] = dataset["Rcoin"]/dataset["L"]
    dataset["Rbot"] = dataset["Rbot"]/dataset["L"]

    # Tool cond are debatable
    dataset["Rcoin_rea"] = dataset["Rcoin_rea"]/dataset["L"]
    dataset["Rbot_rea"] = dataset["Rbot_rea"]/dataset["L"]

    # Allowances are debatable
    dataset["allow_bot"] = dataset["allow_bot"]/dataset["L"]
    dataset["Allow_side"] = dataset["Allow_side"]/dataset["L"]

    dataset["L_act"] = dataset["L_act"] / dataset["L"]
    dataset["l_act"] = dataset["l_act"] / dataset["L"]
    dataset["h_act"] = dataset["h_act"] / dataset["L"]

    dataset["L"] = dataset["L"]/150 # Max size of the pockets

    return dataset

# Metrics are defined for each 'atomic' choice :
def my_metric_op_type(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def my_metric_quality(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def my_metric_strategy(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def my_metric_allowances(y_true, y_pred):
    return K.mean(math_ops.square(y_pred - y_true))

def my_metric_tool_cond(y_true, y_pred):
    return K.mean(math_ops.square(y_pred - y_true))

# Open the csv file containing the dataset
for i in range(a):
    dataset_csv = pd.read_csv("app.csv")

    header = [
        "NAME","L","l","h","Rcoin","Rbot","Open","Al","Ti","Acier","isPlan","L_act","l_act","h_act","Rcoin_act","Rbot_act",
        "bot_side_mill","plane_mill","side_mill","drilling","finish","Unidir_mill","Bidir_mill","Contour_para",
        "Bidir_contour","Contour_spiral","Center_mill","Explicit","allow_bot","Allow_side",
        "Rcoin_rea","Rbot_rea"
    ]

    # Changing the columns types :

    # First, associate each column with its type :
    header_to_normalize = [
        "L","l","h","Rcoin","Rbot","L_act","l_act","h_act","Rcoin_act","Rbot_act",
        "allow_bot","Allow_side","Rcoin_rea","Rbot_rea"
    ]

    header_bool = [
        "Open","Al","Ti","Acier","isPlan",
        "bot_side_mill","plane_mill","side_mill","drilling","finish","Unidir_mill","Bidir_mill","Contour_para",
        "Bidir_contour","Contour_spiral","Center_mill","Explicit"
    ]

    header_numeric = header_to_normalize + header_bool

    # Actual conversion
    dataset_csv[header_numeric] = dataset_csv[header_numeric].astype('float32')

    dataset_csv = normalize_pocket_desc(dataset_csv)

    headerInput = [
        "NAME","L","l","h","Rcoin","Rbot","Open","Al","Ti","Acier","isPlan","L_act","l_act","h_act","Rcoin_act","Rbot_act"
    ]

    headerOutputOP = [
        "NAME", "bot_side_mill", "plane_mill", "side_mill", "drilling", "finish"
    ]

    headerOutputStrat = [
        "NAME", "Unidir_mill", "Bidir_mill","Contour_para","Bidir_contour",
        "Contour_spiral","Center_mill", "Explicit"
    ]

    headerOutputFloats = [
        "NAME", "allow_bot","Allow_side","Rcoin_rea","Rbot_rea"
    ]

    # For further treatment, ex when using the networks to predict on a new case
    op_list = ["bot_side_mill","plane_mill","side_mill","drilling"]
    strat_list = ["Unidir_mill", "Bidir_mill","Contour_para", "Bidir_contour", "Contour_spiral", "Center_mill", "Explicit"]
    allowances = ["allow_bot","Allow_side"]
    tool_cond = ["Rcoin_rea","Rbot_rea"]

    # 20% of the dataset is used for validation purposes
    index = []
    for i in range(int(len(dataset_csv) / 5)-1):
        index.append(randint(1, 5) + i*5)

    val_data = dataset_csv.loc[index]
    # Remove the cases used for validation
    dataset = dataset_csv.drop(index, axis = 0)
    # A Dataset item is used for the training
    dataset_OP = tf.data.Dataset.from_tensor_slices((dataset[headerInput].drop("NAME", axis = 1),
                                                  (
                                                       dataset[op_list],
                                                       dataset['finish']
                                                   )) # Shuffled and packed into batches of 10 OP
                                                 ).shuffle(7, reshuffle_each_iteration = True).batch(10).repeat()

    val_data_OP = tf.data.Dataset.from_tensor_slices((val_data[headerInput].drop("NAME", axis = 1),
                                                   (
                                                       val_data[op_list],
                                                       val_data['finish']
                                                   ))
                                                  ).shuffle(5, reshuffle_each_iteration = True).batch(5).repeat()

    # Same operation is performed for the strategy field...
    index = []
    for i in range(int(len(dataset) / 5)):
        index.append(randint(1, 5) + i*5)

    val_data = dataset_csv.loc[index]

    dataset = dataset_csv.drop(index, axis = 0)

    ind = headerInput + op_list + ['finish']

    dataset_strat = tf.data.Dataset.from_tensor_slices((dataset[ind].drop("NAME", axis = 1),
                                                       dataset[strat_list])
                                                       ).shuffle(7, reshuffle_each_iteration = True).batch(10).repeat()

    val_data_strat = tf.data.Dataset.from_tensor_slices((val_data[ind].drop("NAME", axis = 1),
                                                         val_data[strat_list])
                                                        ).shuffle(5, reshuffle_each_iteration = True).batch(5).repeat()

    # ... and for the tool_cond and allowances
    index = []
    for i in range(int(len(dataset) / 5)):
        index.append(randint(1, 5) + i*5)

    val_data = dataset_csv.loc[index]

    dataset = dataset_csv.drop(index, axis = 0)

    dataset_float = tf.data.Dataset.from_tensor_slices((dataset[headerInput].drop("NAME", axis = 1),
                                                  (
                                                       dataset[allowances],
                                                       dataset[tool_cond]
                                                   ))
                                                 ).shuffle(7, reshuffle_each_iteration = True).batch(10).repeat()

    val_data_float = tf.data.Dataset.from_tensor_slices((val_data[headerInput].drop("NAME", axis = 1),
                                                   (
                                                       val_data[allowances],
                                                       val_data[tool_cond]
                                                   ))
                                                  ).shuffle(5, reshuffle_each_iteration = True).batch(5).repeat()

    # Definition of the networks :
    # One input layer
    visible = Input(shape = (15,))
    # Two hidden layers with 8 neurons each
    hidden_1_op = layers.Dense(8, activation = 'tanh')(visible)

    hidden_2_op = layers.Dense(8, activation = 'tanh')(hidden_1_op)
    # And a two-fold output layer
    op_type = layers.Dense(4, activation = 'softmax')(hidden_2_op) # softmax : Gives 1 to the most probable output
    quality = layers.Dense(1, activation = 'tanh')(hidden_2_op) # tanh : Strong delimitation between outputs -1 and 1

    # My model has an input and an output ; the links between the layers are created when defining the layers
    model_OP = Model(inputs = visible, outputs = [op_type, quality])

    # Parametered optimizer
    my_opti_op = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # The model is instantiated by defining the optimiser, loss functions and metrics
    model_OP.compile(
                  optimizer = my_opti_op,
                  loss = [my_metric_op_type,
                          my_metric_quality],
                  metrics = {
                      'op_type' : my_metric_op_type,
                      'quality' : my_metric_quality
                  } )
    # And then trained on the dataset and validation set defined earlier.
    history_OP = model_OP.fit(dataset_OP, epochs = 50, steps_per_epoch = 20, validation_data = val_data_OP, validation_steps=5, shuffle=True)

    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################

    visible = Input(shape = (20,))

    hidden_1_strat = layers.Dense(8, activation = 'tanh')(visible)

    hidden_2_strat = layers.Dense(8, activation = 'tanh')(hidden_1_strat)

    op_strat = layers.Dense(7, activation = 'softmax')(hidden_2_strat)

    model_strat = Model(inputs = visible, outputs = op_strat)

    my_opti_strat = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model_strat.compile(
                  optimizer = my_opti_strat,
                  loss = my_metric_strategy,
                  metrics = {
                      'strat' : my_metric_strategy
                  })

    history_strat = model_strat.fit(dataset_strat, epochs = 50, steps_per_epoch = 20,
                                    validation_data = val_data_strat, validation_steps=5, shuffle=True)

    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################

    visible = Input(shape = (15,))

    hidden_1_float = layers.Dense(8, activation = 'tanh')(visible)

    hidden_2_float = layers.Dense(8, activation = 'tanh')(hidden_1_float)

    allow = layers.Dense(2, activation = 'relu')(hidden_2_float)
    tool_cond = layers.Dense(2, activation = 'relu')(hidden_2_float)

    model_float = Model(inputs = visible, outputs = [allow, tool_cond])

    my_opti_op = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model_float.compile(
                  optimizer = my_opti_op,
                  loss = [my_metric_allowances,
                          my_metric_tool_cond],
                  metrics = {
                      'allowances' : my_metric_allowances,
                      'tool_cond' : my_metric_tool_cond
                  })

    history_float = model_float.fit(dataset_float, epochs = 50, steps_per_epoch = 20,
                                 validation_data = val_data_float, validation_steps=5, shuffle=True)


    # Display the training history of the three networks
    myFig = display_metrics(history_OP, history_strat, history_float)

    # Save model and training history in graphical form
    directory = r'Saved_models\new_dir'
    i = 0
    while os.path.isdir(directory + str(i)):
        i = i + 1

    directory = directory + str(i)

    os.mkdir(directory)

    myFig.savefig(directory + r'\model.png')
    model_float_json = model_float.to_json()
    with open(directory +  r'\model_float.json', 'w') as json_file:
        json_file.write(model_float_json)


    model_float.save_weights(directory + r'\model_float.h5')

    model_strat_json = model_strat.to_json()
    with open(directory +  r'\model_strat.json', 'w') as json_file:
        json_file.write(model_strat_json)


    model_strat.save_weights(directory + r'\model_strat.h5')

    model_OP_json = model_OP.to_json()
    with open(directory +  r'\model_OP.json', 'w') as json_file:
        json_file.write(model_OP_json)


    model_OP.save_weights(directory + r'\model_OP.h5')
