import os
import time
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.utils import class_weight

import models 
# Ensure this imports the NEW get_data function we wrote in the previous step
from preprocess_golden import get_data 

print(matplotlib.get_backend())

#%% Visualization Helper Functions
def draw_learning_curves(history, results_path):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + "/validation_accuracy.png")
    plt.show()
    plt.close()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + "/validation_loss.png")
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, sub, results_path, classes, normalize=True, cmap=plt.cm.Blues):
    accuracy = accuracy_score(y_true, y_pred)
    F1_note = f1_score(y_true, y_pred, average='macro')
    title = "Sub: " + str(sub) + " Macro F1: " + str(round(F1_note*100.0,2)) + "  " + "Accuracy: " + str(round(accuracy*100.0,2))
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    plt.savefig( results_path + "/cm_subject_" + str(sub) + '.png')
    plt.close(fig)
    return ax

def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])

def scheduler(epoch, lr):
    decay_rate = 0.5
    decay_step = 200
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

#%% Training Function
def train(dataset_conf, train_conf, results_path):
    in_exp = time.time()
    best_models = open(results_path + "/best models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')

    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub') # Should be 1 now
    data_path = dataset_conf.get('data_path')
  
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_train')

    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    # We treat the single dataset as "Subject 1"
    for sub in range(0, n_sub): 
        in_sub = time.time()
        print('\nTraining on Subject (Window-Split)...')
      
        BestSubjAcc = 0 
        bestTrainingHistory = [] 
        
        # --- NEW: Call get_data without 'sub' index, just path and dataset name ---
        X_train, y_train_labels, y_train_onehot, X_test, y_test_labels, y_test_onehot = get_data(data_path, dataset)
        
        # --- NEW: Automatically calculate class weights (Handling Imbalance) ---
        # This replaces the hardcoded dictionary
        y_integers = np.argmax(y_train_onehot, axis=1)
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_integers), 
            y=y_integers
        )
        class_weight_dict = dict(enumerate(class_weights))
        print("Computed Class Weights:", class_weight_dict)

        # Iteration over multiple runs 
        for train in range(n_train): 
            tf.random.set_seed(train+1)
            np.random.seed(train+1)

            in_run = time.time()
            filepath = results_path + '/saved models/run-{}'.format(train+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)        
            
            # Save as subject-1.weights.h5 just to keep file structure consistent
            filepath_save = filepath + '/subject-{}.weights.h5'.format(sub+1)
     
            model = getModel(dataset_conf)
           
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])          
            # model.summary()

            callbacks = [
                ModelCheckpoint(filepath_save, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max'),
                LearningRateScheduler(scheduler)
            ]
            
            # Train
            history = model.fit(
                X_train, y_train_onehot, 
                validation_data=(X_test, y_test_onehot), 
                epochs=epochs, 
                batch_size=batch_size, 
                class_weight=class_weight_dict, # Use auto-calculated weights
                callbacks=callbacks, 
                verbose=1
            )

            # Evaluate (Load best weights from this run)
            model.load_weights(filepath_save)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train]  = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)
              
            out_run = time.time()
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub+1, train+1, ((out_run-in_run)/60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train])
            print(info)
            log_write.write(info +'\n')
            
            if(BestSubjAcc < acc[sub, train]):
                 BestSubjAcc = acc[sub, train]
                 bestTrainingHistory = history
        
        # Save best run info
        best_run = np.argmax(acc[sub,:])
        filepath_best = '/saved models/run-{}/subject-{}.weights.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath_best)
        
        out_sub = time.time()
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info+'\n')
        
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory, results_path)
          
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) )
    print(info)
    log_write.write(info+'\n')
    
    np.savez(perf_allRuns, acc = acc, kappa = kappa)
    best_models.close()   
    log_write.close() 
    perf_allRuns.close() 

#%% Evaluation Function
def test(model, dataset_conf, results_path):
    test_dir_sub = results_path + "/test/Y_truth_Sub_"
    test_dir_pred = results_path + "/test/Y_pred_Sub_"
    if not os.path.exists(test_dir_sub):
        os.makedirs(test_dir_sub)
    if not os.path.exists(test_dir_pred):
        os.makedirs(test_dir_pred)

    log_write = open(results_path + "/log.txt", "a")
    
    # Read the best model path
    with open(results_path + "/best models.txt", "r") as f:
        best_model_paths = f.readlines()

    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    classes_labels = dataset_conf.get('cl_labels')
    
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)  
    
    labels_all = np.array([])
    predic_all = np.array([])

    for sub in range(0, n_sub):
        # Load Data (Single Subject Mode)
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, dataset)
        
        # Load the BEST model for this subject found in training
        # We strip the newline char from the path
        relative_path = best_model_paths[sub].strip() 
        filepath = results_path + relative_path
        print(f"Loading best model for evaluation: {filepath}")
        
        model.load_weights(filepath)
        
        y_pred = model.predict(X_test).argmax(axis=-1)
        labels = y_test_onehot.argmax(axis=-1)
        
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        
        plot_confusion_matrix(labels, y_pred, sub+1, results_path, classes_labels)

        np.save(results_path + "/test/Y_truth_Sub_" + str(sub+1) + ".npy", labels)
        np.save(results_path + "/test/Y_pred_Sub_" + str(sub+1) + ".npy", y_pred)
        labels_all = np.concatenate((labels_all, labels), axis = 0)
        predic_all = np.concatenate((predic_all, y_pred), axis = 0)

        info = 'Subject: {}   acc: {:.4f}   kappa: {:.4f}   '.format(sub+1, acc_bestRun[sub], kappa_bestRun[sub] )
        print(info)
        log_write.write('\n'+info)
      
    info = '\nAverage of {} subjects:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun)) 
    print(info)
    log_write.write(info)
    
    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy')
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score')
    plot_confusion_matrix(labels_all, predic_all, "All", results_path, classes_labels)
    log_write.close() 

#%% Model Definition Wrapper
def getModel(dataset_conf):
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    # Ensure your 'models.py' has the Post_Fusion class
    model = models.Post_Fusion(
        n_classes=n_classes,
        in_chans=n_channels,
        in_samples=in_samples,
        n_windows=4,  
        F1=32,
        D=4,
        kernelSize=20,
        dropout=0.1,
        di_kernelSize=3,
        di_filters=32,
        di_dropout=0.1,
        di_activation='elu'
    )
    return model

#%% Main Execution
def run():
    # Define dataset parameters
    dataset = "Golden" 
    in_samples = 80
    n_channels = 6
    n_sub = 1      # Fixed to 1 for single subject split
    n_classes = 2  # Fixed to 2 for bench_press and arm_curl
    
    classes_labels = ['bench_press', 'arm_curl']
    data_path = "/home/stephen/WHT_Team1A/golden_dataset.csv" 

    results_path = os.getcwd() + "/results_golden"
    print(f"Results will be saved to: {results_path}")

    if not os.path.exists(results_path):
      os.makedirs(results_path)
      
    dataset_conf = { 
        'name': dataset, 
        'n_classes': n_classes, 
        'cl_labels': classes_labels, 
        'n_sub': n_sub, 
        'n_channels': n_channels, 
        'in_samples': in_samples, 
        'data_path': data_path
    }
    
    # Reduced batch size slightly as total data volume might be smaller
    train_conf = { 'batch_size': 32, 'epochs': 100, 'lr': 0.0001, 'LearnCurves': True, 'n_train': 3}
           
    # Train
    train(dataset_conf, train_conf, results_path)

    # Test
    model = getModel(dataset_conf)
    test(model, dataset_conf, results_path)    
    
if __name__ == "__main__":
    run()