import io
import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras


def evaluate(ds_test, model):
    """
    Train the model.

    Input:
        1) ds_test: Tensor Dataframe
            Datatframe containing test data.
        2) model: saved model
            Keras/Tensorflow 2.0 model.
    Output:
        None
    """
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    # test_writer = tf.summary.create_file_writer("logs/test/")
    test_rec_metric = keras.metrics.Recall()
    test_prec_metric = keras.metrics.Precision()

    # Test Loop
    epoch_loss_test = 0
    test_step = 0
    f1 = p = r = 0
    class_names = ['False', 'True']
    confusion = np.zeros((len(class_names), len(class_names)))
    for x1, x2, y in ds_test:
        x2 = tf.expand_dims(x2, axis=1)  # add extra dimension
        x2 = tf.transpose(x2, perm=[0, 1, 3, 2])  # [1, 1, 700, 12]
        y_pred = model((x1, x2), training=False)
        loss = loss_fn(y, y_pred)
        test_acc_metric.update_state(y, y_pred)
        test_rec_metric.update_state(y, tf.argmax(y_pred, axis=1))
        test_prec_metric.update_state(y, tf.argmax(y_pred, axis=1))
        epoch_loss_test += loss
        test_step += 1
        confusion += get_confusion_matrix(y, y_pred, class_names)
    r = test_rec_metric.result()
    p = test_prec_metric.result()
    f1 = (2 * r * p) / (r + p)
    plot_confusion_matrix_for_test(confusion, class_names)
    print(f"Test | Accuracy: {test_acc_metric.result()} | Loss: {epoch_loss_test / test_step} | "
          f"Prec: {p} | Rec: {r} | F1 : {f1}")
    test_acc_metric.reset_states()
    test_rec_metric.reset_states()
    test_prec_metric.reset_states()
    print("=============================>")


def train(ds_train, ds_val, model, num_epochs, lr):
    """
    Train the model.

    Input:
        1) ds_train: Tensor Dataframe
            Datatframe containing train data.
        2) ds_val: Tensor Dataframe
            Datatframe containing validation data.
        3) model: TF 2.0
            Keras/Tensorflow 2.0 model.
        4) num_epochs: int
            Total number of training epochs.
        5) lr: float
            Learning rate for optimizer.
    Output:
        None
    """
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=lr, clipnorm=0.2)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    train_writer = tf.summary.create_file_writer("logs/train/")
    val_writer = tf.summary.create_file_writer("logs/val/")

    train_rec_metric = keras.metrics.Recall()
    train_prec_metric = keras.metrics.Precision()
    val_rec_metric = keras.metrics.Recall()
    val_prec_metric = keras.metrics.Precision()
    class_names = ['False', 'True']
    weights = None
    prev_loss = np.inf
    for epoch in range(num_epochs):

        # Train loop
        epoch_loss = 0
        epoch_acc = 0
        train_step = 0
        p = r = f1 = 0
        confusion = np.zeros((len(class_names), len(class_names)))
        for x1, x2, y in ds_train:
            # train here
            with tf.GradientTape() as tape:
                # x1 = tf.expand_dims(x1, axis=1)  # add extra dimension
                x2 = tf.expand_dims(x2, axis=1)  # add extra dimension
                # change to [BATCH_SIZE, 1, DAYS, CHANNELS]
                # here, we assume that channels refer to the 12 types of waves
                # think of it as obtaining a last row of an image with 12 channels
                x2 = tf.transpose(x2, perm=[0, 1, 3, 2])  # [1, 1, 700, 12]
                #        print(x1.shape)
                #        print(x2.shape)
                y_pred = model((x1, x2), training=True)
                #        print(f"Finalout : {out.shape}")
                loss = loss_fn(y, y_pred)
            # back propagation of gradients
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            # update metrics
            train_acc_metric.update_state(y, y_pred)
            train_rec_metric.update_state(y, tf.argmax(y_pred, axis=1))
            train_prec_metric.update_state(y, tf.argmax(y_pred, axis=1))
            # accumulate loss
            epoch_loss += loss
            # update confusion matrix at each batch
            confusion += get_confusion_matrix(y, y_pred, class_names)
            train_step += 1
        r = train_rec_metric.result()
        p = train_prec_metric.result()
        f1 = (2 * r * p) / (r + p)
        with train_writer.as_default():  # to tensorboard
            tf.summary.scalar("Loss", epoch_loss / train_step, step=epoch)
            tf.summary.scalar("Accuracy", train_acc_metric.result(), step=epoch)
            tf.summary.scalar("Precision", p, step=epoch)
            tf.summary.scalar("Recall", r, step=epoch)
            tf.summary.scalar("F1", f1, step=epoch)
            tf.summary.image(
                "Train Confusion Matrix",
                plot_confusion_matrix(confusion / train_step, class_names),
                step=epoch,
            )
        print(f"Epoch {epoch} | Acc: {train_acc_metric.result()} | Loss: {epoch_loss / train_step} | "
              f"Prec: {p} | Rec: {r} | F1 : {f1}")
        # Reset accuracy in between epochs (and for testing and test)
        train_acc_metric.reset_states()
        train_rec_metric.reset_states()
        train_prec_metric.reset_states()

        # Validation Loop
        epoch_loss_val = 0
        val_step = 0
        vf1 = vp = vr = 0
        vconfusion = np.zeros((len(class_names), len(class_names)))
        for x1, x2, y in ds_val:
            x2 = tf.expand_dims(x2, axis=1)  # add extra dimension
            x2 = tf.transpose(x2, perm=[0, 1, 3, 2])  # [1, 1, 700, 12]
            y_pred = model((x1, x2), training=False)
            loss = loss_fn(y, y_pred)
            val_acc_metric.update_state(y, y_pred)
            val_rec_metric.update_state(y, tf.argmax(y_pred, axis=1))
            val_prec_metric.update_state(y, tf.argmax(y_pred, axis=1))
            epoch_loss_val += loss
            val_step += 1
            vconfusion += get_confusion_matrix(y, y_pred, class_names)
        if epoch_loss_val < prev_loss:  # monitor loss to save best weights
            weights = model.get_weights()
            prev_loss = epoch_loss_val
        vr = val_rec_metric.result()
        vp = val_prec_metric.result()
        vf1 = (2 * vr * vp) / (vr + vp)

        with val_writer.as_default():
            tf.summary.scalar("Loss", epoch_loss_val / val_step, step=epoch)
            tf.summary.scalar("Accuracy", val_acc_metric.result(), step=epoch)
            tf.summary.scalar("Precision", vp, step=epoch)
            tf.summary.scalar("Recall", vr, step=epoch)
            tf.summary.scalar("F1", vf1, step=epoch)
            tf.summary.image(
                "Val Confusion Matrix",
                plot_confusion_matrix(vconfusion / val_step, class_names),
                step=epoch,
            )
        if epoch == num_epochs-1:  # retrieve and save the best weights at the end of training
            model.set_weights(weights)
            model.save_weights("saved_ckpt.tf")
        print(f"Validation | Accuracy: {val_acc_metric.result()} | Loss: {epoch_loss_val / val_step} | "
              f"Prec: {vp} | Rec: {vr} | F1 : {vf1}")
        val_acc_metric.reset_states()
        val_rec_metric.reset_states()
        val_prec_metric.reset_states()
        print("=============================>")


def train_val_test_split_indices(ix, y):
    """
       Split indices to test, val and train.

       Input:
           1) ix: List
               List of sample indices.
           2) y: List
               List of targets.
       Output:
           1) ix_train: List
               List containing train sample indices.
           2) ix_val: List
               List containing val sample indices.
           3) ix_test: List
               List containing test sample indices.
           4) y_train: List
               List containing train targets.
           5) y_val: List
               List containing val targets.
           6) y_test: List
               List containing test targets.

       """
    ix_train_val, ix_test, y_train_val, y_test = train_test_split(ix, y, test_size=0.1, random_state=0, stratify=y)
    ix_train, ix_val, y_train, y_val = train_test_split(ix_train_val, y_train_val, test_size=0.1, random_state=1,
                                                        stratify=y_train_val)

    return ix_train, ix_val, ix_test, y_train, y_val, y_test


def to_tensor(data_demos, data_waves, labels):
    """
    Convert to tensors.

    Input:
        1) data_demos: List
            List containing demographic info.
        2) data_waves: List
            List containing waves of drugs and indicators.
        3) labels: List
            List containing true labels.
    Output:
        1) data_demos: Tensors
            Tensors containing demographic info.
        2) data_waves: Tensors
            Tensors containing waves of drugs and indicators.
        3) labels: Tensors
            Tensors containing true labels.

    """
    data_demos = tf.convert_to_tensor(data_demos)
    data_waves = tf.convert_to_tensor(data_waves)
    labels = tf.convert_to_tensor(labels)
    return data_demos, data_waves, labels


def combine_format(df_ind, df_med, df_demo, df_stage):
    """
    To output the format required for training

    Input:
        1) df_ind: Dataframe
            Datatframe containing waves of indicators.
        2) df_med: Dataframe
            Datatframe containing waves of drugs.
        3) df_demo: Dataframe
            Datatframe containing demographics scalars.
        4) df_stage: Dataframe
            Datatframe containing labels.
    Output:
        1) data_demos: List
            List containing demographic info.
        2) data_waves: List
            List containing waves of drugs and indicators.
        3) labels: List
            List containing true labels.

    """

    # indicators
    # df_ind_ = df_ind.copy() #
    df_ind.drop(columns=['id', 'indicator'], inplace=True)
    # df_ind_['wave'] = df_ind.aggregate(list, axis=1)
    indwaves = df_ind.aggregate(list, axis=1).values.tolist()
    # drugs
    # df_med_ = df_med.copy() #
    df_med.drop(columns=['id', ' drug'], inplace=True)
    # df_med_['wave'] = df_med.aggregate(list, axis=1)
    medwaves = df_med.aggregate(list, axis=1).values.tolist()
    # combine drugs and indicators
    data_waves = []
    for i in range(0, len(indwaves), 6):
        collect = indwaves[i: i + 6]
        collect.extend(medwaves[i: i + 6])
        data_waves.append(collect)

    # demographics
    df_demo['gender'] = df_demo['gender'].apply(lambda x: 1 if 'Male' else 0)
    df_demo['new_race'] = df_demo['new_race'].apply(lambda x: 1 if 'White' else 0)
    data_demos = df_demo[['gender', 'age', 'new_race']].to_numpy().tolist()
    # labels
    df_stage['Stage_Progress'] = df_stage['Stage_Progress'].apply(lambda x: 1 if x else 0)
    labels = df_stage['Stage_Progress'].values.tolist()

    return data_demos, data_waves, labels


def generate_indicator_waves(df_cat, indicator_list, dict_mean_ind, dict_std_ind, max_end_day, max_patient_id=299):
    """
    To generate 1D wave of dosage intake over 700 days for all selected drugs
    -generates waves with 0s when drug is absent
    -includes patients who do not take drugs
    
    Input:
        1) df_aggnew: Dataframe
            Modified datatframe containing lists of lists regarding drug type, dosage levels for 
            each drug type and the corresponding start and end days.
        2) red_drug_list: List
            List of drugs with frequency more thab 10% of sample size.
        3) indicator: List
            List of indicator names.
        4) dict_mean_ind: Dict
            Dictionary with keys as indicators and values as mean values.
        5) dict_std_ind: Dict
            Dictionary with keys as indicators and values as std values.
        6) max_end_day: int
            Maximum end of day for dosage taken (have to be less than 700).
        7) max_patient_id: int
            Maximum value of patient id.
    Output:
        out: List
            List of lists containing patient id, drug name and the 1D wave.
            Format: [patient_id, drug name, 1D wave]
    """
    out = []
    # df_cat = df_cat.to_numpy()

    for i in range(max_patient_id + 1):  # for each patient id

        for indicator in indicator_list:
            collect = [i, indicator]
            [0] * (max_end_day + 1)

            val_col = 'value_' + indicator
            time_col = 'time_' + indicator
            val_list = df_cat.iloc[i][val_col]
            time_list = df_cat.loc[i][time_col]
            # generate wave for current indicator
            wave = linear_interpolate(val_list, time_list, dict_mean_ind, dict_std_ind, indicator, max_end_day)
            collect.extend(wave)  # collect wave for current drug
            out.append(collect)  # collect for each indicator in list

    return out


def linear_interpolate(val_list, time_list, dict_mean_ind, dict_std_ind, indicator, max_end_day=699):
    """
    Interpolate between every 2 adjacent values linearly in the list of indicator values.
    
    Input:
        1) val_list: List
            List of values in every indicator.
        2) time_list: List
            List of days at which lab measurements were taken corresponding to val_list.
        3) dict_mean_ind: Dict
            Dictionary with keys as indicators and values as mean values.
        4) dict_std_ind: Dict
            Dictionary with keys as indicators and values as std values.
        5) indicator: List
            List of indicator names.
        6) max_end_day: int
            Maximum end of day for dosage taken (have to be less than 700).
    
    Output:    
        wave: List
            Interpolated wave generated over 700 days.
    """
    wave = [0] * (max_end_day + 1)
    wave_index = 0
    # normalize the list first
    for val_index in range(len(val_list)):
        val_list[val_index] = (val_list[val_index] - dict_mean_ind[indicator]) / dict_std_ind[indicator]

    for val_index in range(len(val_list)):
        if wave_index == time_list[val_index]:
            wave[wave_index] = val_list[val_index]
            wave_index += 1
            continue
        time_diff = time_list[val_index] - time_list[val_index - 1]
        val_diff = val_list[val_index] - val_list[val_index - 1]
        val_step = val_diff / time_diff
        while wave_index < time_list[val_index]:
            wave[wave_index] = wave[wave_index - 1] + val_step
            wave_index += 1

    # if the last day of measure is not 700th day, then interpolate till average at the end
    if wave_index != max_end_day:
        time_diff = max_end_day - time_list[-1]
        val_diff = np.mean(val_list) - val_list[-1]
        val_step = val_diff / time_diff
        while wave_index <= max_end_day:
            wave[wave_index] = wave[wave_index - 1] + val_step
            wave_index += 1

    return wave


def linear_interpolate2(val_list, time_list, max_end_day=699):
    """
    Interpolate between every 2 adjacent values linearly in the list of indicator values.
    
    Input:
        1) val_list: List
            List of values in every indicator.
        2) time_list: List
            List of days at which lab measurements were taken corresponding to val_list.
        3) max_end_day: int
            Maximum end of day for dosage taken (have to be less than 700).
    
    Output:    
        wave: List
            Interpolated wave generated over 700 days.
    """
    wave = [0] * (max_end_day + 1)
    wave_index = 0
    for val_index in range(len(val_list)):
        if wave_index == time_list[val_index]:
            wave[wave_index] = val_list[val_index]
            wave_index += 1
            continue
        time_diff = time_list[val_index] - time_list[val_index - 1]
        val_diff = val_list[val_index] - val_list[val_index - 1]
        val_step = val_diff / time_diff
        while wave_index < time_list[val_index]:
            wave[wave_index] = wave[wave_index - 1] + val_step
            wave_index += 1
    return wave


def select_drugs(df_meds, thresh_sample=30):
    """
    Feaure selection of drugs that has more than thresh_sample number of patients taking
    
    Input:
        1) df_meds: Dataframe
            Dataframe imported from meds csv.
        2) thresh_sample: int
            Min % of sample size (for each drug) needed for the drug to be selected for modelling.
    
    Output:    
        red_drug_list: List
            List of selected drugs.
    """
    df_temp_1 = df_meds.groupby(['id', 'drug'], as_index=False).nunique()
    drug_freq_series = df_temp_1['drug'].value_counts()
    drug_freq_df = pd.DataFrame(drug_freq_series)
    drug_freq_df = drug_freq_df[drug_freq_df['drug'] > thresh_sample]  # only include if more than 10% of sample size
    red_drug_list = drug_freq_df['drug'].keys().tolist()

    return red_drug_list


def generate_drug_waves(df_aggnew, red_drug_list, dict_max_dosages, max_end_day, max_patient_id=299):
    """
    To generate 1D wave of dosage intake over 700 days for all selected drugs
    -generates waves with 0s when drug is absent
    -includes patients who do not take drugs
    
    Input:
        1) df_aggnew: Dataframe
            Modified datatframe containing lists of lists regarding drug type, dosage levels for 
            each drug type and the corresponding start and end days.
        2) red_drug_list: List
            List of drugs with frequency more thab 10% of sample size.
        3) dict_max_dosages: Dict
            Dictionary of max dosages for each drug
        4) max_end_day: int
            Maximum end of day for dosage taken (have to be less than 700).
        5) max_patient_id: int
            Maximum value of patient id.
    Output:
        out: List
            List of lists containing patient id, drug name and the 1D wave.
            Format: [patient_id, drug name, 1D wave]
    """
    out = []
    df_aggnew_np = df_aggnew.to_numpy()
    count = 0  # tracker to track absence of patients not taking drugs

    for i in range(max_patient_id + 1):  # for each patient id

        # if incremental patient id matches the current index of patient id -> patient takes meds
        if i == df_aggnew_np[count][0]:
            for drug in red_drug_list:  # should only include selected drugs
                collect = [i, drug]
                wave = [0] * (max_end_day + 1)  # init wave with all zeros
                dosage_list = df_aggnew_np[count][2]
                start_day_list = df_aggnew_np[count][3]
                end_day_list = df_aggnew_np[count][4]

                if drug in df_aggnew_np[count][1]:  # if drug is taken by current patient_id
                    ix_drug = df_aggnew_np[count][1].index(drug)
                    # generate wave for current drug
                    wave = generate_wave_for_drug(dosage_list, start_day_list, end_day_list, wave,
                                                  ix_drug, drug, dict_max_dosages)
                    collect.extend(wave)  # collect wave for current drug
                else:
                    collect.extend(wave)  # collect wave for current drug
                out.append(collect)  # collect for each drug in list
            count += 1  # only increment count tracker if drug is taken by current patient_id

        else:  # if patient does not take meds

            out = generate_empty_waves_all_drugs(i, red_drug_list, max_end_day, out)

    return out


def generate_wave_for_drug(dosage_list, start_day_list, end_day_list, wave, ix_drug,
                           drug, dict_max_dosages):
    """
    To generate a wave for a particular drug
    
    Input:
        1) dosage_list: List
            List of lists containing dosage during drug intake.
        2) start_day_list: List
            List of lists containing days when patients started taking drugs.
        3) end_day_list: List
            List of lists containing days when patients stopped taking drugs.
        4) wave: List
            Wave initialised with 0s.
        5) ix_drug: int
            Index of the drug for that particular patient in the collated dataframe.
        6) drug: str
            Name of current drug.
        7) dict_max_dosages: Dict
            Dictionary of max dosages for each drug
    
    Output:
        wave: List
            Modified wave
    """
    for n in range(len(dosage_list[ix_drug])):  # for each dosage level
        for ix in range(start_day_list[ix_drug][n] + 1, end_day_list[ix_drug][n] + 2):
            wave[ix] = dosage_list[ix_drug][n] / dict_max_dosages['max_dosage'][drug]
    return wave


def generate_empty_waves_all_drugs(patient_id, red_drug_list, max_end_day, out):
    """
    To generate waves filled with 0s for patients who do not take drugs
    
    Input:
        1) patient_id: int
            The index of patient.
        2) red_drug_list: List
            List containing selected drugs.
        3) max_end_day: int
            Maximum day from day 0 when the patient stops taking the drug
        4) out: List
            Containing the final collection of waves.
    
    Output:
        out: List
            Containing the final collection of waves.
    """
    for drug in red_drug_list:  # should only include selected drugs
        collect = [patient_id, drug]
        wave = [0] * (max_end_day + 1)  # init wave with all zeros
        collect.extend(wave)  # collect wave for current drug
        out.append(collect)  # collect for each drug in list
    return out


#####
# Obtained from tensorflow official guide:
# https://www.tensorflow.org/tensorboard/image_summaries and
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial17-tensorboard/4_tb_confusion.py

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def get_confusion_matrix(y_labels, probs, class_names):
    preds = np.argmax(probs, axis=1)
    cm = sk.metrics.confusion_matrix(
        y_labels, preds, labels=np.arange(len(class_names)),
    )
    return cm


def plot_confusion_matrix(cm, class_names):
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.get_cmap('Blues', 128))
    plt.title("Confusion Matrix")

    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45, fontsize=10)
    plt.yticks(indices, class_names, fontsize=10)

    # Normalize Confusion Matrix
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=3, )

    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = "red" if cm[i, j] < threshold else "green"
            # color = "black"
            plt.text(
                i, j, cm[i, j], horizontalalignment="center", color=color,
            )

    plt.tight_layout()
    plt.xlabel("True Label", fontsize=10)
    plt.ylabel("Predicted label", fontsize=10)

    cm_image = plot_to_image(figure)
    return cm_image


def plot_confusion_matrix_for_test(cm, class_names):
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.get_cmap('Blues', 128))
    plt.title("Confusion Matrix")

    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45, fontsize=10)
    plt.yticks(indices, class_names, fontsize=10)

    # Normalize Confusion Matrix
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=3, )

    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = "red" if cm[i, j] < threshold else "green"
            # color = "black"
            plt.text(
                i, j, cm[i, j], horizontalalignment="center", color=color,
            )

    plt.tight_layout()
    plt.xlabel("True Label", fontsize=10)
    plt.ylabel("Predicted label", fontsize=10)
    plt.savefig('saved_plots/test_cm.png')
    plt.close(figure)
