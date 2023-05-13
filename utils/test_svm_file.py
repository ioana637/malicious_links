from joblib import Memory
from sklearn.datasets import load_svmlight_file

mem = Memory("./mycache")

@mem.cache
def get_data():
    data = load_svmlight_file("D:\\IdeaProjects\\malicious_links\\data\\Day120.svm")
    return data[0], data[1]

if __name__=="__main__":
    X, y = get_data()
    print(X)
    print(y)

