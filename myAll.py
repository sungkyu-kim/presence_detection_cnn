import os
import csv
from data_preprocessing import *
from data_learning import *
from wifi_process_combo import *
from convert import *

def test(model_, subcarrier_spacing_, shape_conversion_):
    data_preprocessing_path_ = data_preprocessing_path +"_"+ model_ +"_"+ str(subcarrier_spacing_) +"_"+ str(shape_conversion_) +'/'
    if not os.path.isdir(data_preprocessing_path_):
        os.mkdir(data_preprocessing_path_)

    test_conf = dict()
    test_conf['nsubcarrier'] = int(conf.nsubcarrier_max / subcarrier_spacing_)

    test_conf['nshape_conversion'] = shape_conversion_
    test_conf['model'] = model_
    test_conf['data_shape_to_nn'] = model_
    test_conf['fft_shape'] = (conf.n_timestamps, test_conf['nsubcarrier'])
    test_conf['data_shape_to_nn'] = (shape_conversion_, test_conf['nsubcarrier'], conf.ntx*conf.nrx, 2)
    test_conf['abs_shape_to_nn'] = (shape_conversion_, test_conf['nsubcarrier'], conf.ntx*conf.nrx)
    test_conf['phase_shape_to_nn'] = (shape_conversion_, test_conf['nsubcarrier'], conf.ntx*(conf.nrx-1))

    d_list = ['day9', 'day10', 'day11']
    data_preprocessing(d_list, data_path, data_preprocessing_path_, test_conf)

    model_path = test_path+'model/'
    if not os.path.isdir(model_path):
        print(f'mkdir test_path : {model_path}')
        os.mkdir(model_path)
    model_path_name = model_path +"_"+ model_ +"_"+ str(subcarrier_spacing_) +"_"+ str(shape_conversion_)
    if conf.do_fft:
        model_name = model_path_name+'wifi_presence_model_fft.h5'
    else:
        model_name = model_path_name+'wifi_presence_model.h5'
        
    train_list = ['day9']
    test_list = ['day10']
    data_learning(train_list, test_list, data_preprocessing_path_, model_path_name, test_conf, model_name)

    test_list = ['day11']
    test_result = wifi_process_combo(test_list, data_preprocessing_path_, model_path_name, test_conf, model_name)

    if conf.do_fft:
        model_name = model_path_name+'wifi_presence_model_fft'
    else:
        model_name = model_path_name+'wifi_presence_model'
    tflite_size = convert_lite(model_name)

    print(f"===== ===== {model_} {subcarrier_spacing_} {shape_conversion_} : {tflite_size} , {test_result}")
    data = {"model":model_, "subcarrier_spacing":subcarrier_spacing_, "shape_conversion":shape_conversion_, "tflite_size":tflite_size, "test_result":test_result}
    fcsv = open(test_path+"result.csv","a")
    fieldnames = ["model", "subcarrier_spacing", "shape_conversion", "tflite_size", "test_result"]
    writer = csv.DictWriter(fcsv,fieldnames=fieldnames)
    writer.writerow(data)
    fcsv.close()
    
now = time.localtime()
folder_name = "%04d-%02d-%02d_%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

test_path = f"test/{folder_name}/"
print(f'test_path : {test_path}')
if not os.path.isdir(test_path):
    print(f'mkdir test_path : {test_path}')
    os.mkdir(test_path)

data_path = 'data/data/'

model_list = ["model1"]
subcarrier_spacing_list = [4, 7, 8]
shape_conversion_list = [50, 40, 30, 20]

data_preprocessing_path = test_path+'data_preprocessing/'
if not os.path.isdir(data_preprocessing_path):
    os.mkdir(data_preprocessing_path)

for model_ in model_list:
    for subcarrier_spacing_ in subcarrier_spacing_list:
        for shape_conversion_ in shape_conversion_list:
            print(f"\n\n<<< Test Start >>> {model_} {subcarrier_spacing_} {shape_conversion_}")
            test(model_, subcarrier_spacing_, shape_conversion_)

model_list = ["model2"]
subcarrier_spacing_list = [4, 7, 8, 14]
shape_conversion_list = [50, 40, 30, 20, 10]

data_preprocessing_path = test_path+'data_preprocessing/'
if not os.path.isdir(data_preprocessing_path):
    os.mkdir(data_preprocessing_path)

for model_ in model_list:
    for subcarrier_spacing_ in subcarrier_spacing_list:
        for shape_conversion_ in shape_conversion_list:
            print(f"\n\n<<< Test Start >>> {model_} {subcarrier_spacing_} {shape_conversion_}")
            test(model_, subcarrier_spacing_, shape_conversion_)