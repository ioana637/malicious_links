import enum
# Using enum class create enumerations
class Algorithms(enum.Enum):
    RF = 'rf'
    DT = 'dt'
    KNN = 'knn'
    LR = 'lr'
    BNB = 'bnb'
    GNB = 'gnb'
    LDA = 'lda'
    ADA = 'ada'
    XGB = 'xgb'
    SVM_poly = 'svm_poly'
    SVM_sigmoid = 'svm_sigmoid'
    SVM_linear = 'svm_linear'
    SVM_rbf = 'svm_rbf'
    MLP = 'mlp'
    GPC = 'gpc'

class SVM_Kernels(enum.Enum):
    poly = 'poly',
    rbf = 'rbf',
    linear = 'linear',
    sigmoid = 'sigmoid'
