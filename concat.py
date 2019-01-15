from random import randint
import tensorflow as tf                 # Tensorflow, developpé par Google, une excellente librairie ML
from tensorflow.keras import layers     # KERAS est une surcouche plus haut niveau, qui me permet de manipuler des
                                        # objets TF sans trop m'embêter - Esprit "Le ML n'est qu'un outil"
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
import numpy as np                      # NUMPY, manipulation de données, matrices, vecteurs, des maths quoi
import pandas as pd                     # PANDAS, manipulation de données, inclus un petit utilitaire TRES pratique pour ouvrir des CSV
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_util
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import os

a = input('Nb of cycles ?')
try:
    a = int(a)
except:
    a = 1
if a > 1:
    plt.ioff()

nice_string = """Machine this pocket using :

Operation : {} {}
Strategy  : {}

With allowances :

Radial    : {}
Axial     : {}

Requirements on the tool :

Mx Radius : {}
Mx R_beak : {}"""

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

def display_results(model, pocket_desc = [140.0, 44.0, 16.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    # Evaluation sur un cas test et affichage :
    print("Description de la poche :\n ", pocket_desc)
    my_X = tf.constant([pocket_desc])

    Y = model.predict(my_X, steps = 1)
    Y = K.eval(K.sum(Y, axis = 0))

    # Pour affichage des différentes valeurs de sorties :
    # for i in range(len(headerOutput)-1):
    #     print("{} : {}".format(headerOutput[i+1], Y[i]))

    for i in range(len(Y)):
        print("{} : {}".format(headerOutput[i+1], Y[i]))
    # Juste pour cosmétique, avec "nice_string" défini en tête de script.
    op_type_index = K.eval(math_ops.argmax(Y[0:3]))
    quality = 0 if Y[4]>0.5 else 1
    strategy_index = K.eval(math_ops.argmax(Y[5:11]))
    allowance_rad = Y[12]
    allowance_bot = Y[13]
    tool_description = Y[14:]

    etat_intermediaire = [0, 0, 0, 0, 0]

    etat_intermediaire[0] = Y[0] - 2*allowance_rad
    etat_intermediaire[1] = Y[1] - 2*allowance_rad
    etat_intermediaire[2] = Y[2] - 2*allowance_bot
    #etat_intermediaire[3] = math.floor(Y[14])+1
    #etat_intermediaire[4] = math.floor(Y[15])+1

    #etat_intermediaire.append(K.eval(my_X)[0:-5])

    #print("Etat intermédiaire : \n", etat_intermediaire)

    op_type = op_list[op_type_index]
    strat = strat_list[strategy_index]

    print('\n\n\t\t####################################################################\n\n')

    print(nice_string.format(op_type, ["finish", ""][quality], strat, allowance_rad, allowance_bot, tool_description[0], tool_description[1]))

    return(etat_intermediaire)

def normalize_pocket_desc(dataset):
    dataset["l"] = dataset["l"]/dataset["L"]
    dataset["h"] = dataset["h"]/dataset["L"]

    dataset["Rcoin_act"] = dataset["Rcoin_act"] / dataset["L"]
    dataset["Rbot_act"] = dataset["Rbot_act"] / dataset["L"]

    dataset["Rcoin"] = dataset["Rcoin"]/dataset["L"]
    dataset["Rbot"] = dataset["Rbot"]/dataset["L"]
    # "Rcoin_rea","Rbot_rea"
    dataset["Rcoin_rea"] = dataset["Rcoin_rea"]/dataset["L"]
    dataset["Rbot_rea"] = dataset["Rbot_rea"]/dataset["L"]
    # "allow_bot","Allow_side"
    dataset["allow_bot"] = dataset["allow_bot"]/dataset["L"]
    dataset["Allow_side"] = dataset["Allow_side"]/dataset["L"]

    dataset["L_act"] = dataset["L_act"] / dataset["L"]
    dataset["l_act"] = dataset["l_act"] / dataset["L"]
    dataset["h_act"] = dataset["h_act"] / dataset["L"]



    dataset["L"] = dataset["L"]/150

    return dataset

def my_metric_custom(y_true, y_pred):
    """
    L'objectif de cette métrique est de caractériser la prédiction - y_pred - par rapport à la caleur "réelle" - y_true
    On a une prédiction composée de différents éléments ; deux sets de booleens mutuellement exclusifs, un booleen simple
    et des réels.

    On se propose de séparer la métrique en 5 parties :
    >> Le type d'opération :
          "bot_side_mill","plane_mill","side_mill","drilling"
    >> La qualité :
          "finish"
    >> La stratégie :
          "Unidir_mill","Bidir_mill","Contour_para", "Bidir_contour","Contour_spiral","Center_mill","Explicit"
    >> Les surépaisseurs :
          "allow_bot","Allow_side"
    >> La caractérisation outil :
          "Rcoin_rea","Rbot_rea"
    """
    # "bot_side_mill","plane_mill","side_mill","drilling"
    op_type = 1 - K.cast(K.equal(
        K.argmax(y_true[:, 0:3], axis = -1),
        K.argmax(y_pred[:, 0:3], axis = -1))
        , K.floatx())

    # "finish"
    quality = 1 - K.cast(K.equal(
        K.argmax(y_true[:, 4:5], axis = -1),
        K.argmax(y_pred[:, 4:5], axis = -1))
        , K.floatx())

    # "Unidir_mill","Bidir_mill","Contour_para", "Bidir_contour","Contour_spiral","Center_mill","Explicit"
    strategy = 1 - K.cast(K.equal(
        K.argmax(y_true[:, 5:11], axis = -1),
        K.argmax(y_pred[:, 5:11], axis = -1))
        , K.floatx())

    # "allow_bot","Allow_side"
    # Check : mean_squared_logarithmic_error
    allowances = K.mean(math_ops.square(y_pred[:, 12:13] - y_true[:, 12:13]), axis=-1)
    # Mean Sq Err : Will give important loss, where the boolean will just give 0 or 1
    # >> Check something that'll be more "normalised"

    # "Rcoin_rea","Rbot_rea"
    tool_cond = K.mean(math_ops.square(y_pred[:, 14:15] - y_true[:, 14:15]), axis=-1)

    # https://www.tensorflow.org/api_docs/python/tf/concat
    value = K.mean(tf.stack([op_type, quality, strategy, allowances, tool_cond]), 1)
    print(value)
    return value


# Pour constater l'évolution suivant l'intégralité des catégories :
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

def my_metrics_no_treatment(y_true, y_pred):
    return K.cast(y_true - y_pred, K.floatx())


# Ouverture du fichier csv contenant le dataset, avec la bibliothèque pandas, que toutes les bonnes bibliothèques de ML comprennent ;)
for i in range(a):
    dataset_csv = pd.read_csv("venv/mats/app.csv")

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

    headerOutput = [
        "NAME", "bot_side_mill","plane_mill","side_mill","drilling","finish","Unidir_mill",
        "Bidir_mill","Contour_para","Bidir_contour","Contour_spiral","Center_mill",
        "Explicit","allow_bot","Allow_side","Rcoin_rea","Rbot_rea"
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

    # Pour affichage des opérations à réaliser :
    op_list = ["bot_side_mill","plane_mill","side_mill","drilling"]
    strat_list = ["Unidir_mill", "Bidir_mill","Contour_para", "Bidir_contour", "Contour_spiral", "Center_mill", "Explicit"]
    allowances = ["allow_bot","Allow_side"]
    tool_cond = ["Rcoin_rea","Rbot_rea"]

    # On extrait 10% des cas pour la validation :
    index = []
    for i in range(int(len(dataset_csv) / 5)-1):
        index.append(randint(1, 5) + i*5)

    val_data = dataset_csv.loc[index]

    dataset = dataset_csv.drop(index, axis = 0)

    dataset_OP = tf.data.Dataset.from_tensor_slices((dataset[headerInput].drop("NAME", axis = 1),
                                                  (
                                                       dataset[op_list],
                                                       dataset['finish']
                                                   ))
                                                 ).shuffle(7, reshuffle_each_iteration = True).batch(10).repeat()

    val_data_OP = tf.data.Dataset.from_tensor_slices((val_data[headerInput].drop("NAME", axis = 1),
                                                   (
                                                       val_data[op_list],
                                                       val_data['finish']
                                                   ))
                                                  ).shuffle(5, reshuffle_each_iteration = True).batch(5).repeat()

    # On extrait 10% des cas pour la validation :
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

    # On extrait 10% des cas pour la validation :
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


    visible = Input(shape = (15,))

    #normalisation_layer_op = layers.BatchNormalization()(visible)

    #hidden_1_op = layers.Dense(8, activation = 'tanh')(normalisation_layer_op)

    hidden_1_op = layers.Dense(8, activation = 'tanh')(visible)

    hidden_2_op = layers.Dense(8, activation = 'tanh')(hidden_1_op)

    op_type = layers.Dense(4, activation = 'softmax')(hidden_2_op) # softmax : Gives 1 to the most probable output
    quality = layers.Dense(1, activation = 'tanh')(hidden_2_op) # tanh : Strong delimitation between outputs -1 and 1


    # output_layer = concatenate([op_type, quality, op_strat, allow_and_tool_cond])

    model_OP = Model(inputs = visible, outputs = [op_type, quality])


    # my_opti = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    my_opti_op = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model_OP.compile(
    #              optimizer=tf.train.RMSPropOptimizer(0.001),
                  optimizer = my_opti_op,
                  loss = [my_metric_op_type,
                          my_metric_quality],
    #              loss=tf.keras.losses.categorical_crossentropy,
                  metrics = {
                      'op_type' : my_metric_op_type,
                      'quality' : my_metric_quality
                  } )

    history_OP = model_OP.fit(dataset_OP, epochs = 50, steps_per_epoch = 20, validation_data = val_data_OP, validation_steps=5, shuffle=True)

    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################
    ########        ########        ########        ########        ########        ########
    ########################################################################################

    # visible_input = Input(shape = (15,))
    # visible_op = Input(shape = (4,))
    # visible_quality = Input(shape=(1,))
    #
    # visible = concatenate([visible_input, visible_op, visible_quality])

    visible = Input(shape = (20,))

    # normalisation_layer_strat = layers.BatchNormalization()(visible)
    #
    # hidden_1_strat = layers.Dense(8, activation = 'tanh')(normalisation_layer_strat)

    hidden_1_strat = layers.Dense(8, activation = 'tanh')(visible)

    hidden_2_strat = layers.Dense(8, activation = 'tanh')(hidden_1_strat)

    op_strat = layers.Dense(7, activation = 'softmax')(hidden_2_strat)


    # output_layer = concatenate([op_type, quality, op_strat, allow_and_tool_cond])

    model_strat = Model(inputs = visible, outputs = op_strat)


    # my_opti = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    my_opti_strat = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model_strat.compile(
    #              optimizer=tf.train.RMSPropOptimizer(0.001),
                  optimizer = my_opti_strat,
                  loss = my_metric_strategy,
    #              loss=tf.keras.losses.categorical_crossentropy,
                  metrics = {
                      'strat' : my_metric_strategy
                  } )

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

    # normalisation_layer_float = layers.BatchNormalization()(visible)
    #
    # hidden_1_float = layers.Dense(8, activation = 'tanh')(normalisation_layer_float)

    hidden_1_float = layers.Dense(8, activation = 'tanh')(visible)

    hidden_2_float = layers.Dense(8, activation = 'tanh')(hidden_1_float)

    allow = layers.Dense(2, activation = 'relu')(hidden_2_float) # Allows for predictions in the [0;inf[ range
    tool_cond = layers.Dense(2, activation = 'relu')(hidden_2_float)


    # output_layer = concatenate([op_type, quality, op_strat, allow_and_tool_cond])

    model_float = Model(inputs = visible, outputs = [allow, tool_cond])


    # my_opti = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    my_opti_op = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model_float.compile(
    #              optimizer=tf.train.RMSPropOptimizer(0.001),
                  optimizer = my_opti_op,
                  loss = [my_metric_allowances,
                          my_metric_tool_cond],
    #              loss=tf.keras.losses.categorical_crossentropy,
                  metrics = {
                      'allowances' : my_metric_allowances,
                      'tool_cond' : my_metric_tool_cond
                  } )

    history_float = model_float.fit(dataset_float, epochs = 50, steps_per_epoch = 20,
                                 validation_data = val_data_float, validation_steps=5, shuffle=True)

    # a = input('Affichage du graphe :\n0 - Non\n1 - Oui')
    # if a == "1":
    myFig = display_metrics(history_OP, history_strat, history_float)
    #
    # # display_results(model)
    #
    # # Todo : Boucler s/ nouvel état "as-is" de la poche, après OP n°1
    # print("\nThat's all folks !")
    #
    directory = r'Saved_models\new_dir'
    i = 0
    while os.path.isdir(directory + str(i)):
        i = i + 1
    #
    directory = directory + str(i)
    #
    os.mkdir(directory)
    #
    # model_json = model.to_json()
    # with open(directory +  r'\model.json', 'w') as json_file:
    #     json_file.write(model_json)
    #
    #
    # model.save_weights(directory + r'\model.h5')
    #
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
