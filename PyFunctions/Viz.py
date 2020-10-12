import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc, confusion_matrix 
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle




def plot_loss_accuracy(model_history, theme, path = None): 
    '''This function will create a graph showing the change in loss throughout each epoch'''
    plt.style.use(theme)
    train_loss = model_history.history['loss']
    train_acc = model_history.history['acc']
    test_loss = model_history.history['val_loss']
    test_acc = model_history.history['val_acc']
    epochs = [i for i in range(1, len(test_acc)+1)]

    fig, ax = plt.subplots(1,2, figsize = (10,5))
    ax[0].plot(epochs, train_loss, label = 'Train Loss')
    ax[0].plot(epochs, test_loss, label = 'Test Loss')
    ax[0].set_title('Train/Test Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss (cateogircal_crossentropy)')
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label = 'Train Accuracy')
    ax[1].plot(epochs, test_acc, label = 'Test Accuracy')
    ax[1].set_title('Train/Test Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    
    if path: 
        plt.savefig(path)
    

    
def plot_roc_auc(model, x_test, y_test, theme, model_type, path = None):
    '''This function will create ROC curve given the model, x_test, y_test, and theme for the plot. '''
    plt.style.use(theme)
    plt.figure(figsize = (8,5))
    if model_type.upper() == 'mask':
        y_test = label_binarize(y_test, classes = [0,1])
    elif model_type.upper() == 'EMOTION':
        y_test = label_binarize(y_test, classes = [0,1,2])
    n_classes = y_test.shape[1]
    
    #AUC CURVE
    
    y_test_prob = model.predict(x_test)
    y_test_pred = [np.argmax(i) for i in y_test_prob]
    y_test_actual = [np.argmax(i) for i in y_test]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes): 
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_test_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
               
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
   
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    lw = 2
    # Plot all ROC curves

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC/AUC for Each Class (Test)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    f1 = f1_score(y_test_actual, y_test_pred, average = None)
    if model_type.upper() == 'MASK':
        statement = f'F1 Scores Test\n~~~~~~~~~~~~~~~~~~~~~~\nNo Mask(0): {f1[0]}\nMask(1): {f1[1]}'
    elif model_type.upper() == 'EMOTION':
        statement = f'F1 Scores Test\n~~~~~~~~~~~~~~~~~~~~~~\nAngry(0): {f1[0]}\nHappy(1): {f1[1]}\nNeutral(2): {f1[2]}'
    print(statement)
    
    if path: 
        plt.savefig(path)
    plt.show()
    
      
def plot_model_cm(test_cm, train_cm, classes,
                          theme, cmap=plt.cm.Blues, path = None, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    plt.style.use(theme)

    if normalize:
        test_cm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
        train_cm = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots(1,2, figsize = (8,8))
    
    #Test Set
    ax[0].imshow(test_cm, interpolation='nearest', cmap=cmap)
    ax[0].set_title('CM for Test')
    tick_marks = np.arange(len(classes))
    ax[0].set_xticks(tick_marks, classes)
    ax[0].set_yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = test_cm.max() / 2.
    for i, j in itertools.product(range(test_cm.shape[0]), range(test_cm.shape[1])):
        ax[0].text(j, i, format(test_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if test_cm[i, j] > thresh else "black")

    ax[0].set_ylabel('True label')
    ax[0].set_xlabel('Predicted label')
    ax[0].set_ylim(2.5, -.5)

    
    
    #Train Set
    ax[1].imshow(train_cm, interpolation='nearest', cmap=cmap)
    ax[1].set_title('CM for Train')
    tick_marks = np.arange(len(classes))
    ax[1].set_xticks(tick_marks, classes)
    ax[1].set_yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = train_cm.max() / 2.
    for i, j in itertools.product(range(train_cm.shape[0]), range(train_cm.shape[1])):
        ax[1].text(j, i, format(train_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if train_cm[i, j] > thresh else "black")

    ax[1].set_ylabel('True label')
    ax[1].set_xlabel('Predicted label')
    ax[1].set_ylim(2.5, -.5)
    
    plt.tight_layout()
    
    if path: 
        plt.savefig(path)
    plt.show()   