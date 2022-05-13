#coding:UTF-8
import scipy.io as scio
import h5py
import pickle


def read_data(file):
    import numpy as np


    f = h5py.File(file)

    metadata = {}
    metadata['height'] = []
    metadata['label'] = []
    metadata['left'] = []
    metadata['top'] = []
    metadata['width'] = []


    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
                vals.append(int(obj[0][0]))
        else:
            for k in range(obj.shape[0]):
                vals.append(int(f[obj[k][0]][0][0]))
        metadata[name].append(vals)


    for item in f['/digitStruct/bbox']:
        f[item[0]].visititems(print_attrs)



    pickle.dump(metadata,open("train.pkl","wb"))


if __name__=='__main__':
    read_data("digitStruct.mat")
    # data = pickle.load(open("train.pkl","rb"))
    # print(data["label"][:10])

