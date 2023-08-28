# first line: 6
@mem.cache
def get_data():
    data = load_svmlight_file("D:\\IdeaProjects\\malicious_links\\data\\Day120.svm")
    return data[0], data[1]
