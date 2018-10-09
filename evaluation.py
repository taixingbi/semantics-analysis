from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn import metrics

def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='val')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='val')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def evaluate_model(model, X_test, y_true):
    scores = model.evaluate(X_test, y_true, verbose=0)

    y_prob = model.predict(X_test, batch_size=10, verbose=1)
    y_pred= [y[0]> 0.5 for y in y_prob]
    precision, recall, fscore, _= precision_recall_fscore_support(y_true, y_pred, average='binary')
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("precision: %.2f%%" % (precision*100))
    print("recall: %.2f%%" % (recall*100))
    print("fscore: %.2f%%" % (fscore*100))

    fpr, tpr, _ = metrics.roc_curve(y_true,  y_prob)
    auc = metrics.roc_auc_score(y_true, y_prob)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()