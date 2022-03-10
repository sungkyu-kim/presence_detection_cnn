import tensorflow as tf
def convert_lite(model_name):
    #model_name = 'model/wifi_presence_model_fft'
    model = tf.keras.models.load_model(model_name+'.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(model_name+'.tflite',"wb").write(tflite_model)

    '''
    def representative_dataset():
        for i in range(500):
            yield([x_train[i].reshape(1,1)])

    converter.optimizations = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset

    tflite_model_quan = converter.convert()
    open(model_name+'_quan.tflite',"wb").write(tflite_model_quan)
    '''