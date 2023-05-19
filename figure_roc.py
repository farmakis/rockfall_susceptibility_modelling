from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np

for fig, dataset in enumerate(['mile109', 'wcw', 'marsden']):
    plt.figure(fig)
    plt.plot([0, 1], [0, 1], 'k--')

    for model in ['pointnet++', 'pointcnn', 'dgcnn']:
        rates = np.load('./predictions/roc_{}_{}.npy'.format(model, dataset), allow_pickle=True)
        score = auc(rates[0], rates[1])
        plt.plot(rates[0], rates[1], label='{} (AUC = {:.2f})'.format(model, score))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('{} - ROC curve'.format(dataset))
    plt.legend(loc='lower right')
    plt.savefig('./predictions/roc_{}.png'.format(dataset))




