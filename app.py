import streamlit as st
import os
import tensorflow as tf
import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
# from keras.layers import TimeDistributed as TD
from Time_Distr import TimeDistributed as TD
import Memristor as mem
from SCNN import Integrator_layer, Reduce_sum, sparse_data_generator_non_spiking

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print('Num GPUs Available: ', tf.config.list_physical_devices('GPU'))
# st.success('This is a success message!', icon="✅")

if 'nn_type' not in st.session_state:
    st.session_state.nn_type = None
if 'snn' not in st.session_state:
    st.session_state.snn = False
if 'load' not in st.session_state:
    st.session_state.load = False
if 'upld' not in st.session_state:
    st.session_state.upld = False
if 'custom' not in st.session_state:
    st.session_state.custom = False
# Initialization session_state for added layers
if 'submittedLayers' not in st.session_state:
    st.session_state.submittedLayers = []

if 'descr' not in st.session_state:
    st.session_state.descr = {}
if 'x_train' not in st.session_state:
    st.session_state.x_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'x_test' not in st.session_state:
    st.session_state.x_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'ip_shape' not in st.session_state:
    st.session_state.ip_shape = None
if 'model' not in st.session_state:
    st.session_state.model = None


st.title("Build your Neural Network")

# Select box for neural network type
nn_type = st.selectbox("Please be specific about the Neural Network",("Hardware","Software"))
makeIt = st.button('Make It')

c1, c2, c3 = st.columns((8,1,1))
with c1:
    st.write('Are you going to build a SCNN?',st.session_state.snn)

with c2:
    snn = st.button('Yes')
with c3:
    No_snn = st.button('No')

if snn:
    st.session_state.snn = True
if No_snn:
    st.session_state.snn = False

if makeIt:
    st.session_state.nn_type = nn_type
    st.session_state.load = False


# Select box for selecting the dataset
st.session_state.dataset = st.sidebar.selectbox("Select and Load dataset",("mnist","cifar10","cifar100","Iris"))

# uploaded_file = st.sidebar.file_uploader("Choose a csv file")

# if uploaded_file is not None:

#     # Can be used wherever a "file-like" object is accepted:
#     dataframe = pd.read_csv(uploaded_file)
#     st.write(dataframe)


c1,c2 = st.sidebar.columns((1,2))
with c1:
    load = st.button('Load')
with c2:
    upld = st.button('Upload image dataset')

if load:
    st.session_state.load = True
    st.session_state.submittedLayers = []

if upld:
    if st.session_state.upld:
        st.session_state.upld = False
    else:
        st.session_state.upld = True

def custom_dataset(path,shape,test_size):
    shape = eval(shape)
    classes = []
    for p in os.listdir(path):
        if os.path.isdir(os.path.join(path,p)):
            classes.append(p)
    images = []
    label = []
    label_count = 0
    for clss in classes:
        trg_path = os.path.join(path,clss)
        for img in os.listdir(trg_path):
            img = cv2.imread(trg_path+'/'+img)
            img = cv2.resize(img,shape)
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_array)
            label.append(label_count)
        label_count += 1
    images = np.array(images)
    label = np.array(label)
    n_classes = len(classes)
    x_train, x_test, y_train, y_test = train_test_split(images, label, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test, n_classes


if st.session_state.upld:
    st.sidebar.warning('The Image folder should be in a format "Root folder--> class1 folder-->(images), class2 folder-->(images), etc"')
    # st.sidebar.caption('Root folder--> class1 folder-->(images), class2 folder-->(images), etc')
    rpath = st.sidebar.text_input('Give path of the Root folder')
    
    shape = st.sidebar.text_input('Target shape in tuple format')
    st.sidebar.caption('target shape is the shape in which all your images will be resized into. eg:(32,32)')

    test_size = st.sidebar.number_input('Test_size for splitting dataset',min_value=0.0,max_value=1.0,value=0.2)

    done = st.sidebar.button('Done')
    if done:
        st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test, n_classes = custom_dataset(rpath,shape,test_size)
        st.sidebar.success('Successfully uploaded')
        st.session_state.y_train = np.asarray(st.session_state.y_train).astype('float32').reshape((-1,1))
        st.session_state.y_test = np.asarray(st.session_state.y_test).astype('float32').reshape((-1,1))
        st.session_state.custom = True
        st.session_state.descr = {'Number of classes': n_classes,
                                'x_train shape ': st.session_state.x_train.shape,
                                'x_test shape ': st.session_state.x_test.shape,
                                'y_train shape ': st.session_state.y_train.shape,
                                'y_test shape ': st.session_state.y_test.shape}
        st.session_state.ip_shape = st.session_state.x_train.shape[1:]
        st.session_state.model = Sequential()
        st.session_state.model.add(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape))


if not st.session_state.load or  not st.session_state.custom:
    st.write('Load or upload the dataset from the sidebar')

# function for loading the selected dataset
def get_dataset(dataset):
    if dataset=="mnist":
        descr = {
                "Dataset" : "MNIST digits classification dataset",
                "About" : "This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.",
                "xTrain" : "uint8 NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data. Pixel values range from 0 to 255.",
                "yTrain" : "uint8 NumPy array of digit labels (integers in range 0-9) with shape (60000,) for the training data.",
                "xTest" : "uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data. Pixel values range from 0 to 255.",
                "yTest" : "uint8 NumPy array of digit labels (integers in range 0-9) with shape (10000,) for the test data."
            }
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Model / data parameters
        num_classes = 10
        ip_shape = (28, 28, 1)

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        st.sidebar.success("Dataset loaded",icon='🤩')
    
    elif dataset=="cifar10":
        descr = {
                "Dataset":"CIFAR10 small images classification dataset",
                "About":"This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories.",
                "xTrain": "uint8 NumPy array of grayscale image data with shapes (50000, 32, 32, 3), containing the training data. Pixel values range from 0 to 255.",
                "yTrain": "uint8 NumPy array of labels (integers in range 0-9) with shape (50000, 1) for the training data.",
                "xTest": "uint8 NumPy array of grayscale image data with shapes (10000, 32, 32, 3), containing the test data. Pixel values range from 0 to 255.",
                "yTest": "uint8 NumPy array of labels (integers in range 0-9) with shape (10000, 1) for the test data."
            }
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
        ip_shape = (32, 32, 3)

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        st.sidebar.success("Dataset loaded",icon='🤩')

    elif dataset=="cifar100":
        descr = {
                "Dataset":"CIFAR10 small images classification dataset",
                "About":"This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 100 fine-grained classes that are grouped into 20 coarse-grained classes.",
                "xTrain": "uint8 NumPy array of grayscale image data with shapes (50000, 32, 32, 3), containing the training data. Pixel values range from 0 to 255.",
                "yTrain": "uint8 NumPy array of labels (integers in range 0-9) with shape (50000, 1) for the training data.",
                "xTest": "uint8 NumPy array of grayscale image data with shapes (10000, 32, 32, 3), containing the test data. Pixel values range from 0 to 255.",
                "yTest": "uint8 NumPy array of labels (integers in range 0-9) with shape (10000, 1) for the test data."
            }
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        num_classes = 100
        ip_shape = (32, 32, 3)

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        st.sidebar.success("Dataset loaded",icon='🤩')

    elif dataset=='Iris':
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.model_selection import train_test_split

        iris_data = load_iris()
        x = iris_data.data
        y_ = iris_data.target.reshape(-1, 1)

        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y_)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        ip_shape = (4,)
        descr={'Dataset':'Iris dataset',
                'About':'This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray. The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.',
                'x_train' :  'x_train shape is (120, 4)',
                'x_test' :  'x_test shape is (30, 4)',
                'y_train' :  'y_train shape is (120, 1)',
                'y_test' :  'y_test shape is (30, 1)'
                }
        st.sidebar.success("Dataset loaded",icon='🤩')
    else:
        st.write("Please select a dataset")

    return descr, ip_shape, x_train, y_train, x_test, y_test

#loading the dataset
if load:
    descr,ip_shape, x_train, y_train, x_test, y_test = get_dataset(st.session_state.dataset)
    st.session_state.x_train = x_train
    st.session_state.y_train = y_train
    st.session_state.x_test = x_test
    st.session_state.y_test = y_test
    st.session_state.descr = descr
    st.session_state.ip_shape = ip_shape
    st.session_state.model = Sequential()
    if st.session_state.snn:
        st.session_state.model.add(TD(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape)))
    else:
        st.session_state.model.add(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape))

if (st.session_state.load or st.session_state.custom) and st.session_state.nn_type:
    if st.session_state.model == None:
        st.session_state.model = Sequential()
        st.session_state.model.add(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape))
        # st.write(st.session_state.ip_shape)
    # if st.session_state.nn_type == 'Hardware':
    #     st.session_state.Hmodel = Sequential()
    #     st.session_state.Hmodel.add(tf.keras.layers.InputLayer(input_shape=ip_shape))
    if (st.session_state.dataset == 'mnist' and st.session_state.load):
        st.sidebar.caption('The loaded dataset has shape (28,28,1). If you want to reshape it to (784,) please click the below button')
        reshape = st.sidebar.button('Reshape')
        if reshape:
            num_pixels = 784
            st.session_state.x_train = st.session_state.x_train.reshape(st.session_state.x_train.shape[0], num_pixels)
            st.session_state.x_test = st.session_state.x_test.reshape(st.session_state.x_test.shape[0], num_pixels)
            st.session_state.ip_shape = (784,)
            st.session_state.model = Sequential()
            st.session_state.model.add(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape))
            st.session_state.submittedLayers = []
            st.sidebar.success('Successfully reshaped')
            # st.sidebar.write(st.session_state.x_train.shape)

if load and not st.session_state.nn_type:
    st.sidebar.error("Are you sure that you selected the type of your Neural Network. If not make it and try loading again.....") 

# container showing loaded dataset discription
with st.container():
    if st.session_state.descr =={}:
        pass
    else:
        st.subheader('Loaded dataset')
        for i in st.session_state.descr.keys():
            st.write(i," :  ",st.session_state.descr[i])

        if st.session_state.custom:
            Norm = st.button('Normalize the dataset')
            st.caption('If Normalization shows error, try changing target shape to lower pixel sizes like (32,32) and upload again. Or you can skip normalization step and move on. But remember that this step will affect the accuracy of your model.')
            if Norm:
                st.session_state.x_train = st.session_state.x_train.astype("float32") / 255
                st.session_state.x_test = st.session_state.x_test.astype("float32") / 255
                st.success('Succesfully Normalized')

        if st.session_state.snn:
            c1,c2 = st.columns(2)
            with c1:
                b_size = st.number_input('batch_size', value = 32)
                n_steps = st.number_input('number of steps', value = 100)
            with c2:
                sh = st.selectbox('shuffle',(True,False))
                fl = st.selectbox('flatten',(False,True))
            timesteps = st.number_input('timesteps', value = 100)
            c1,c2,c3 = st.columns((1,1,1))
            with c2:
                spike = st.button('Generate spiking dataset')

            if spike:
                x_train_for_spiking = st.session_state.x_train
                x_test_for_spiking = st.session_state.x_test
                y_train_for_spiking = st.session_state.y_train
                y_test_for_spiking = st.session_state.y_test
                ip_shape_for_spiking = [st.session_state.ip_shape[0], st.session_state.ip_shape[1], st.session_state.ip_shape[2]]
                st.session_state.dataset_generator = tf.data.Dataset.from_generator(lambda: sparse_data_generator_non_spiking(input_images=x_train_for_spiking,
                                                                                            input_labels=y_train_for_spiking,
                                                                                            batch_size=b_size,
                                                                                            nb_steps=n_steps, shuffle=True,
                                                                                            flatten=fl),
                                                    output_shapes=((None, timesteps, ip_shape_for_spiking[0], ip_shape_for_spiking[1], ip_shape_for_spiking[2]), (None, 10)),
                                                    output_types=(tf.float64, tf.uint8))
                st.session_state.dataset_generator_test = tf.data.Dataset.from_generator(lambda: sparse_data_generator_non_spiking(input_images=x_test_for_spiking,
                                                                                            input_labels=y_test_for_spiking,
                                                                                            batch_size=b_size,
                                                                                            nb_steps=n_steps, shuffle=sh,
                                                                                            flatten=fl),
                                                    output_shapes=((None, timesteps, ip_shape_for_spiking[0], ip_shape_for_spiking[1], ip_shape_for_spiking[2]), (None, 10)),
                                                    output_types=(tf.float64, tf.uint8))
                
                st.success('Successfully generated')

# dict storing each layers and parameters
LAYERSandPARAMS={
    "Reshape":{
        "target_shape":'(28, 28, 1)',
        "name":"Reshape_1"
    },
    "Dense":{
        "units": 10,
        "activation":("relu","sigmoid","softmax","softplus","softsign","tanh","selu","elu","exponential",None),
        "kernel_initializer":("RandomUniform","RandomNormal","TruncatedNormal","Zeros","Ones","GlorotNormal","GlorotUniform","HeNormal","HeUniform","Identity","Orthogonal","Constant","VarianceScaling"),
        "bias_initializer":("zeros","RandomNormal","RandomUniform","TruncatedNormal","Ones","GlorotNormal","GlorotUniform","HeNormal","HeUniform","Identity","Orthogonal","Constant","VarianceScaling"),
        "name":"dense_1"
    },
    "Conv2D":{
        "filters": 32,
        "kernel_size":3,
        "strides":1,
        "activation":("relu","sigmoid","softmax","softplus","softsign","tanh","selu","elu","exponential",None),
        "padding":("valid","same","causal"),
        "kernel_initializer":("RandomUniform","RandomNormal","TruncatedNormal","Zeros","Ones","GlorotNormal","GlorotUniform","HeNormal","HeUniform","Identity","Orthogonal","Constant","VarianceScaling"),
        "bias_initializer":("zeros","RandomNormal","RandomUniform","TruncatedNormal","Ones","GlorotNormal","GlorotUniform","HeNormal","HeUniform","Identity","Orthogonal","Constant","VarianceScaling"),
        "name":"Conv2D_1"
    },
    "DepthwiseConv2D":{
       "kernel_size":3,
        "depth_multiplier":1,
        "depthwise_initializer":("glorot_uniform","RandomNormal","RandomUniform","TruncatedNormal","Zeros","Ones","GlorotNormal","HeNormal","HeUniform","Identity","Orthogonal","Constant","VarianceScaling"),
        "depthwise_constraint":(None,"MaxNorm","MinMaxNorm","NonNeg","UnitNorm","RadialConstraint"),
        "depthwise_regularizer":(None,"L1","L2","L1L2","OrthogonalRegularizer"),
        "name":"DepthwiseConv2D_1"
    },
    "MaxPooling1D":{
        "pool_size":2,
        "strides":1,
        "padding":("valid","same"),
        "data_format":("channels_last","channels_first"),
        "name":"MaxPooling1D_1"
    },
    "MaxPooling2D":{
        "pool_size":2,
        "strides":1,
        "padding":("valid","same"),
        "data_format":("channels_last","channels_first"),
        "name":"MaxPooling2D_1"
    },
    "AveragePooling1D":{
        "pool_size":2,
        "strides":1,
        "padding":("valid","same"),
        "data_format":("channels_last","channels_first"),
        "name":"AveragePooling1D_1"
    },
    "AveragePooling2D":{
        "pool_size":2,
        "strides":1,
        "padding":("valid","same"),
        "data_format":("channels_last","channels_first"),
        "name":"AveragePooling1D_1"
    },
    "Dropout":{
        "rate":0.5,
        "name":"Dropout_1"
    },
    "GaussianNoise":{
        "stddev":0.2
    },
    "GaussianDropout":{
        "rate":0.5
    },
    "AlphaDropout":{
        "rate":0.5,
        #"noise_shape":2,
        "seed":1
    },
    "LSTM":{
        "units":5,
        "return_sequences":True,
        "activation":("tanh","sigmoid","relu","softmax","softplus","softsign","selu","elu","exponential",None),
        "recurrent_activation":("sigmoid","relu","softmax","softplus","softsign","tanh","selu","elu","exponential",None),
        "use_bias":True,
        "kernel_initializer":("glorot_uniform","RandomNormal","RandomUniform","TruncatedNormal","Zeros","Ones","GlorotNormal","HeNormal","HeUniform","Identity","Orthogonal","Constant","VarianceScaling"),
        "recurrent_initializer":("Orthogonal","glorot_uniform","RandomNormal","RandomUniform","TruncatedNormal","Zeros","Ones","GlorotNormal","HeNormal","HeUniform","Identity","Constant","VarianceScaling"),
        "bias_initializer":("zeros","RandomNormal","RandomUniform","TruncatedNormal","Ones","GlorotNormal","GlorotUniform","HeNormal","HeUniform","Identity","Orthogonal","Constant","VarianceScaling"),
        "name":"LSTM_1"
    },
    "Flatten":{"name":"Flatten_1"},
    "Integrator_layer":{"name":"Integrator_layer_1"},
    "Reduce_sum":{"name":"Reduce_sum_1"},

}

# form for setting the parameters of the layer selected and Submit(Software)
if st.session_state.snn:
    with st.sidebar:
        layer = st.selectbox("Select a layer",('Conv2D', 'Integrator_layer', 'Flatten', 'Dense', 'Reduce_sum'))
        with st.form("SNNParams"):
            params = dict()
            if layer in LAYERSandPARAMS.keys():
                st.caption('Set the parameters below')
                for i in LAYERSandPARAMS[layer].keys():
                    if i=='units':
                        val =  st.number_input(i,min_value=0, max_value=None, value=LAYERSandPARAMS[layer][i])
                        params[i] = val
                    if i=='filters':
                        val =  st.number_input(i,min_value=0, max_value=None, value=LAYERSandPARAMS[layer][i])
                        params[i] = val
                    if i=='kernel_size':
                        val =  st.number_input(i,min_value=0, max_value=None, value=LAYERSandPARAMS[layer][i])
                        params[i] = val
                    if i=='name':
                        val =  st.text_input(i, value=LAYERSandPARAMS[layer][i])
                        st.caption('Please update name when each layer is added')
                        params[i] = val
             
            submitted = st.form_submit_button("Submit")
            st.caption('Submitted layers will be displayed in the main page under Added Layers.')
            if submitted:
                if st.session_state.descr =={}:
                    st.error("Please load a dataset first, then start adding layers",icon='💁‍♀️')
                else:
                    try:
                        if layer=='Dense':
                            st.session_state.model.add(TD(tf.keras.layers.Dense(
                                units=params['units'],
                                activation=None
                                ),name = params['name']))
                        if layer=='Conv2D':
                            st.session_state.model.add(TD(tf.keras.layers.Conv2D(
                                filters=params['filters'],
                                kernel_size=params['kernel_size'],
                                activation=None
                                ),name =params['name']))
                        if layer == 'Flatten':
                            st.session_state.model.add(TD(tf.keras.layers.Flatten(),name =params['name']))
                        if layer == 'Integrator_layer':
                            st.session_state.model.add(Integrator_layer(name=params['name']))
                        if layer == 'Reduce_sum':
                            st.session_state.model.add(Reduce_sum(name=params['name']))

                        st.session_state.submittedLayers.append([layer,params])
                        st.success('Submitted Successfully',icon='🎉')
                        st.write("Layer :", layer)
                        st.write("Parameters", params)
                    except Exception as ex:
                        st.error(ex,icon="🥺")

else:
    with st.sidebar:
            layer = st.selectbox("Select a layer",("Dense","Conv2D","DepthwiseConv2D","MaxPooling2D","Reshape","Flatten","Dropout","GaussianNoise","GaussianDropout","AlphaDropout"))
            with st.form("Params"):
                params = dict()
                if layer in LAYERSandPARAMS.keys():
                    st.caption('Set the parameters below')
                    for i in LAYERSandPARAMS[layer].keys():
                        if isinstance(LAYERSandPARAMS[layer][i], tuple) and i!='target_shape':
                            val = st.selectbox(i,LAYERSandPARAMS[layer][i])
                            params[i] = val
                        elif i=='target_shape':
                            val =  st.text_input(i, value=LAYERSandPARAMS[layer][i])
                            st.caption('Please enter in a tuple format, Eg:(28, 28, 1)')
                            params[i] = val
                        elif i=='rate' or i=='stddev':
                            val =  st.number_input(i,min_value=0.0, max_value=1.0, value=LAYERSandPARAMS[layer][i])
                            params[i] = val
                        elif i=='name':
                            val =  st.text_input(i, value=LAYERSandPARAMS[layer][i])
                            st.caption('Please update name when same layer is added')
                            params[i] = val
                        elif (i=="return_sequences") or (i =='use_bias'):
                            val =  st.selectbox(i, (True,False))
                            params[i] = val
                        else:
                            val =  st.number_input(i,min_value=0, max_value=None, value=LAYERSandPARAMS[layer][i])
                            params[i] = val
                submitted = st.form_submit_button("Submit")
                st.caption('Submitted layers will be displayed in the main page under Added Layers.')
                if submitted:
                    if st.session_state.descr =={}:
                        st.error("Please load a dataset first, then start adding layers",icon='💁‍♀️')
                    else:
                        try:
                            if layer=='Dense':
                                st.session_state.model.add(tf.keras.layers.Dense(
                                    units=params['units'],
                                    activation=params['activation'],
                                    kernel_initializer =params['kernel_initializer'],
                                    bias_initializer =params['bias_initializer'],
                                    name = params['name']
                                    ))
                            if layer=='Conv2D':
                                st.session_state.model.add(tf.keras.layers.Conv2D(
                                    filters=params['filters'],
                                    kernel_size=params['kernel_size'],
                                    activation=params['activation'],
                                    strides =params['strides'],
                                    padding =params['padding'],
                                    kernel_initializer =params['kernel_initializer'],
                                    bias_initializer =params['bias_initializer'],
                                    name =params['name']
                                    ))
                            if layer=='DepthwiseConv2D':
                                st.session_state.model.add(tf.keras.layers.DepthwiseConv2D(
                                    kernel_size=params['kernel_size'],
                                    depth_multiplier=params['depth_multiplier'],
                                    depthwise_initializer=params['depthwise_initializer'],
                                    depthwise_constraint=params['depthwise_constraint'],
                                    depthwise_regularizer=params['depthwise_regularizer'],
                                    name =params['name']
                                    ))
                            if layer=='MaxPooling1D':
                                st.session_state.model.add(tf.keras.layers.MaxPooling1D(
                                    pool_size=params['pool_size'],
                                    strides =params['strides'],
                                    padding =params['padding'],
                                    data_format =params['data_format'],
                                    name =params['name']
                                    ))
                            if layer=='MaxPooling2D':
                                st.session_state.model.add(tf.keras.layers.MaxPooling2D(
                                    pool_size=params['pool_size'],
                                    strides =params['strides'],
                                    padding =params['padding'],
                                    data_format =params['data_format'],
                                    name =params['name']
                                    ))
                            if layer=='AveragePooling1D':
                                st.session_state.model.add(tf.keras.layers.AveragePooling1D(
                                    pool_size=params['pool_size'],
                                    strides =params['strides'],
                                    padding =params['padding'],
                                    data_format =params['data_format'],
                                    name =params['name']
                                    ))
                            if layer=='AveragePooling2D':
                                st.session_state.model.add(tf.keras.layers.AveragePooling2D(
                                    pool_size=params['pool_size'],
                                    strides =params['strides'],
                                    padding =params['padding'],
                                    data_format =params['data_format'],
                                    name =params['name']
                                    ))
                            if layer=='Reshape':
                                ts = eval(params['target_shape'])
                                st.session_state.model.add(tf.keras.layers.Reshape(
                                    ts,name =params['name']
                                    ))
                            if layer=='Dropout':
                                rate = params['rate']
                                st.session_state.model.add(tf.keras.layers.Dropout(
                                    rate,name =params['name']
                                    ))
                            if layer=='GaussianNoise':
                                st.session_state.model.add(tf.keras.layers.GaussianNoise(
                                    stddev=params['stddev']
                                    ))
                            if layer=='GaussianDropout':
                                st.session_state.model.add(tf.keras.layers.GaussianDropout(
                                    rate=params['rate']
                                    ))
                            if layer=='AlphaDropout':
                                st.session_state.model.add(tf.keras.layers.AlphaDropout(
                                    rate=params['rate'],
                                    #noise_shape=params['noise_shape'],
                                    seed=params['seed']
                                    ))
                            if layer == 'LSTM' and st.session_state.ip_shape != (4,):
                                if st.session_state.model.layers == []:
                                    st.session_state.model = Sequential()
                                    st.session_state.model.add(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape[:-1]))
                                
                                    if st.session_state.ip_shape[:-1] == 3:
                                        st.session_state.x_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in st.session_state.x_train])
                                        st.session_state.x_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in st.session_state.x_test])
                                    
                                st.session_state.model.add(tf.keras.layers.LSTM(
                                    units=params['units'],
                                    name = params['name'],
                                    return_sequences=params['return_sequences']
                                ))
                            if layer == 'Flatten':
                                st.session_state.model.add(tf.keras.layers.Flatten())

                            st.session_state.submittedLayers.append([layer,params])
                            st.success('Submitted Successfully',icon='🎉')
                            st.write("Layer :", layer)
                            st.write("Parameters", params)
                        except Exception as ex:
                            st.error(ex,icon="🥺")

# if 'HardwareLayers' not in st.session_state:
#     st.session_state.HardwareLayers = []

# HardwareLayers = {
#     "Dense":{
#         "units":3,
#         "name":"Dense_1"
#     },
#     "LSTM":{
#         "units":5,
#         "return_sequences":True,
#         "name":"LSTM_1"
#     },
#     "Conv2D":{
#         "filters":3,
#         "kernel_size":3,
#         "name":"Conv2D_1"
#     },
#     "MaxPooling2D":{
#         "pool_size":2,
#         "name":"MaxPooling2D_1"
#     }
# }

# if st.session_state.nn_type == 'Hardware':
#     with st.sidebar:
#         layer = st.selectbox("Select a layer",("Dense","Conv2D","MaxPooling2D","Flatten","LSTM"))
#         with st.form("HParams"):
#             params={}
#             if layer in HardwareLayers.keys():
#                 for i in HardwareLayers[layer].keys():
#                     if i=="name":
#                         val =  st.text_input(i, value=HardwareLayers[layer][i])
#                         st.caption('Please update name when same layer is added')
#                         params[i] = val
#                     elif i=="return_sequences":
#                         val =  st.selectbox(i, (True,False))
#                         params[i] = val
#                     else:
#                         val =  st.number_input(i,min_value=0, max_value=None, value=HardwareLayers[layer][i])
#                         params[i] = val
                
#             submitted = st.form_submit_button("Submit")
#             if submitted:
#                 if st.session_state.descr =={}:
#                     st.error("Please load a dataset first, then start adding layers",icon='💁‍♀️')
#                 else:
#                     try:
#                         if layer=='Dense':
#                             st.session_state.Hmodel.add(tf.keras.layers.Dense(
#                                     units=params['units'],
#                                     name = params['name']
#                                     ))
#                         if layer=='Conv2D':
#                             st.session_state.Hmodel.add(tf.keras.layers.Conv2D(
#                                     filters=params['filters'],
#                                     kernel_size=params['kernel_size'],
#                                     name = params['name']
#                                     ))
#                         if layer == 'Flatten':
#                             st.session_state.Hmodel.add(tf.keras.layers.Flatten())

#                         if layer == 'MaxPooling2D':
#                             st.session_state.Hmodel.add(tf.keras.layers.MaxPooling2D(
#                                 pool_size=params['pool_size'],
#                                 name = params['name']
#                             ))
#                         if layer == 'LSTM' and st.session_state.ip_shape != (4,):
#                             if st.session_state.Hmodel.layers == []:
#                                 st.session_state.Hmodel = Sequential()
#                                 st.session_state.Hmodel.add(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape[:-1]))
                            
#                                 if st.session_state.ip_shape == (32,32,3):
#                                     st.session_state.x_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in st.session_state.x_train])
#                                     st.session_state.x_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in st.session_state.x_test])
                                
#                             st.session_state.Hmodel.add(tf.keras.layers.LSTM(
#                                 units=params['units'],
#                                 name = params['name'],
#                                 return_sequences=params['return_sequences']
#                             ))

#                         if layer == 'LSTM' and st.session_state.ip_shape == (4,):
#                             st.error('Please choose an appropriate dataset for the LSTM')
#                         else:
#                             st.session_state.HardwareLayers.append([layer,params])
#                             st.success('Submitted Successfully')
#                             st.write("Layer :", layer)
#                             st.write("Parameters", params)

#                     except Exception as ex:
#                         st.error(ex,icon="🥺")


if 'Store' not in st.session_state:
    st.session_state.Store = {"Dataset":[],"loss":[], "accuracy":[],"precision":[],"recall":[],"f1 score":[],"Neural network config":[]}


def show_layers(layer_list):
    for i in layer_list:
            layer_with_idx = str((layer_list.index(i))+1)+'  '+i[0]
            with st.expander(layer_with_idx):
                st.write(i[1])
    
def show_compile_fit():
    with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Compile')
                optimizer = st.selectbox('optimizer',('adam','sgd','rmsprop','nadam','adadelta','adagrad','adamax','ftrl'))
                loss = st.selectbox('loss',('categorical_crossentropy','binary_crossentropy','sparse_categorical_crossentropy','poisson'))
            with col2:
                st.subheader('Fit')
                epochs = st.number_input('epochs',max_value=None, min_value=1, value=2)
                if st.session_state.snn:
                    # batch_size = 0
                    # count = st.number_input('repeat count',max_value=None, min_value=0, value=1)
                    txt = 'repeat count'
                else:
                    txt = 'batch_size'
                    # count = 0
                batch_size = st.number_input(txt,max_value=None, min_value=0, value=10)
                # validation_split = st.number_input('validation_split',max_value=None, min_value=0.0, value=0.1)
    return optimizer,loss,epochs,batch_size

def run_model(model,loss,optimizer,epochs,batch_size):
    # print(model.summary())
    print("Initialize epochs:", epochs)
    try:
        if st.session_state.snn:
            if loss == 'categorical_crossentropy':
                model.compile(loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),     
                optimizer = optimizer, 
                metrics = ['accuracy'])
            if loss == 'binary_crossentropy':
                model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),     
                optimizer = optimizer, 
                metrics = ['accuracy'])
            if loss == 'sparse_categorical_crossentropy':
                model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),     
                optimizer = optimizer, 
                metrics = ['sparse_categorical_accuracy'])
            if loss == 'poisson':
                model.compile(loss = tf.keras.losses.Poisson(from_logits=True),     
                optimizer = optimizer, 
                metrics = ['accuracy'])

            model_fit = model.fit(st.session_state.dataset_generator.repeat(count=1),
                                  epochs=epochs,
                                  validation_data=st.session_state.dataset_generator_test.repeat(count=1))
        else:
            model.compile(loss = loss,     
                optimizer = optimizer, 
                metrics = ['accuracy'])

            model_fit = model.fit(st.session_state.x_train, st.session_state.y_train,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=(st.session_state.x_test, st.session_state.y_test))
                
        # if st.session_state.snn:
        #     print("Hey hey People!!!", len(st.session_state.x_train))
        #     print("I am at the sesion state")
        #     print("1122", max(st.session_state["x_train"]))
        #     print('SNN training epochs:', epochs)
        #     print(epochs)
        #     model_fit = model.fit(st.session_state.dataset_generator.repeat(count=1),
        #                           epochs=epochs,
        #                           validation_data=st.session_state.dataset_generator_test.repeat(count=1))
        # else:
        #     print("Initialize epochs non spike:", epochs)
        #     model_fit = model.fit(st.session_state.x_train, st.session_state.y_train,
        #             epochs = epochs,
        #             batch_size = batch_size,
        #             validation_data=(st.session_state.x_test, st.session_state.y_test))


        # st.snow()
        model.save_weights('Model_Weights.h5')
        return model_fit
    except Exception as ex:
        st.error(ex)

def cal_result(model):
    if st.session_state.snn:
       st.session_state.score = model.evaluate(st.session_state.dataset_generator_test, verbose=2)
    else:
        st.session_state.score = model.evaluate(st.session_state.x_test, st.session_state.y_test, verbose=0)
        y_test_class = np.argmax(st.session_state.y_test, axis=1)
        y_pred = np.argmax(model.predict(st.session_state.x_test, verbose=0),axis=1)

        # precision tp / (tp + fp)
        precision = precision_score(y_test_class, y_pred, average='weighted', labels=np.unique(y_pred))
        # recall: tp / (tp + fn)
        recall = recall_score(y_test_class, y_pred, average='weighted', labels=np.unique(y_pred))
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test_class, y_pred, average='weighted', labels=np.unique(y_pred))
        config = model.get_config()
        st.session_state.Store["Neural network config"].append(config)
        st.session_state.Store["loss"].append(st.session_state.score[0])
        st.session_state.Store["precision"].append(precision)
        st.session_state.Store["accuracy"].append(st.session_state.score[1])
        st.session_state.Store["recall"].append(recall)
        st.session_state.Store["f1 score"].append(f1)
        st.session_state.Store["Dataset"].append(st.session_state.dataset)

def show_results(model_fit):
    st.subheader('Results')
    st.write("Test loss:", st.session_state.score[0])
    st.write("Test accuracy:", st.session_state.score[1])

    col1, col2= st.columns([1,1])
    with col1:
        fig = plt.figure()
        plt.plot(model_fit.history['loss'], label='train')
        plt.plot(model_fit.history['val_loss'], label='val')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        st.pyplot(fig)

    with col2:
        fig = plt.figure()
        plt.plot(model_fit.history['accuracy'], label='train')
        plt.plot(model_fit.history['val_accuracy'], label='val')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        st.pyplot(fig)
 
if 'nn_submit' not in st.session_state:
    st.session_state.nn_submit = False

# if st.session_state.submittedLayers!=[] and st.session_state.nn_type == 'Software':-  
#     # container for showing added layers
#     with st.container():
#         st.subheader("Added Layers")
#         show_layers(st.session_state.submittedLayers) 
#         reset = st.button('Reset')

#         # resetting the submittedLayers and so the model too
#         if reset:
#             st.session_state.Smodel = Sequential(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape))
#             st.session_state.submittedLayers = []

#         optimizer,loss,epochs,batch_size = show_compile_fit()
        
#         col1, col2, col3 = st.columns([2,1,2])
#         with col2:
#             submitAll = st.button('Submit all')
                
#         # if submitAll:
#         #     show_results(st.session_state.Smodel)

#         if submitAll:
#             st.session_state.model_fit = run_model(st.session_state.Smodel,loss,optimizer,epochs,batch_size)
#             cal_result(st.session_state.Smodel)
#             st.session_state.nn_submit = True
    
#         if st.session_state.nn_submit:
#             show_results(st.session_state.model_fit)

#         if st.session_state.Store!={}:
#             df=pd.DataFrame(st.session_state.Store)
#             st.table(df)

if 'setup' not in st.session_state:
    st.session_state.setup = False
if 'csv' not in st.session_state:
    st.session_state.csv = None

def set_hardware_weights(model):
    st.text("")
    st.text("")
    col1,col2 = st.columns(2)
    with col1:
        mem_txt = "Select the memristor "#+str(mem)
        memristor_model = st.radio(mem_txt, ('Joglekar','Prodromakis','Biolek','Zha'),key=mem_txt)
        if memristor_model=='Joglekar' or memristor_model=='Biolek':
            p=st.number_input('Enter p value', value = 1)
            j=1
        if memristor_model=='Prodromakis' or memristor_model=='Zha':
            p=st.number_input('Enter p value', value=7)
            j=st.number_input('Enter j value', value=1)
        Amplitude = st.number_input('Amplitude', value = 1)
        freq = st.number_input('Frequency', value = 1)
    with col2:
        Ron_txt = "Ron"#+str(mem)
        Ron = st.number_input('Set Ron value', min_value=100,max_value=16000, value=100,key=Ron_txt)
        Roff_txt = "Roff"#+str(mem)
        Roff = st.number_input('Set Roff value', min_value=100, max_value=16000, value=16000, key=Roff_txt)
        part_txt = "part"#+str(mem)
        Rint = st.number_input('Set Rint value', min_value=100, max_value=16000, value=11000)
        partition = st.slider('Define the Quatization value here',2,64, key=part_txt)
        sample_rate = st.number_input('Sample Rate', value = 500)


    # st.write('Would you like to add some variabilities? Add them below...')
    # Ron_Roff_txt = "Ron_Roff"#+str(mem)
    Ron_Roff_aging = st.checkbox("Ron-Roff Aging")
    c1,c2,c3 = st.columns((1,2,1))
    if Ron_Roff_aging:
        with c2:
            st.caption('Aging value can be positive or negative')
            Ron_aging = st.number_input('Enter aging % (b/w 0-20)',key='ronAge',value=0)
            Roff_aging = st.number_input('Enter aging % (b/w 0-20)',key='roffAge',value=0)
    else:
        Ron_aging = 0
        Roff_aging = 0
    

    c1,c2,c3 = st.columns((1,1,1))
    with c2:
        setup = st.button('Set up Memristor')
    if setup:
        st.session_state.setup = True

    if setup:
        st.text("")
        st.text("")

        # Get the current weights of the neural network
        old_weights = model.get_weights()

        old_weight_array = np.concatenate([arr.flatten() for arr in old_weights])

        # Calculate the minimum and maximum values of the old weights
        old_weight_min = np.amin(np.abs(old_weight_array))
        old_weight_max = np.amax(np.abs(old_weight_array)) 

        lyr=0
        for layer in model.layers:
            lyr += 1
            if layer.__class__.__name__ == 'Dense' or layer.__class__.__name__ =='Conv2D' or layer.__class__.__name__ == 'LSTM':
                try:
                    shape = layer.get_weights()[0].shape
                    txt = "Weights for the layer "+layer.name+" of shape "+str(shape)
                    st.subheader(txt)

                    old_weights = list(layer.get_weights()[0])
                    st.session_state.old_weights = []
                    st.session_state.old_bias = []
                    idx = 0

                    if layer.__class__.__name__ == 'LSTM':
                        # old_weights = layer.trainable_weights[0]
                        # old_weights = old_weights.numpy()
                        # shape = layer.trainable_weights[0].shape
                        # old_bias = layer.trainable_weights[1]
                        st.session_state.old_weights = old_weights
                        st.session_state.new_weights = []
                        st.session_state.new_u = []
                        st.session_state.old_u = layer.get_weights()[1]
                        shape_u = st.session_state.old_u.shape
                        old_bias = layer.get_weights()[2]

                        for weight in list(old_weights):
                            Mem = mem.memristor_models(Roff,Ron,Rint,Amplitude,freq,1,sample_rate,p,j,memristor_model)
                            Mem.variability(partition,Ron_aging,Roff_aging)
                            weight = (list(weight))
                            Mem.neural_weight([weight], old_weight_max, old_weight_min) 
                            st.session_state.new_weights.append(Mem.new_weights())
                        
                        for weight in list(st.session_state.old_u):
                            Mem = mem.memristor_models(Roff,Ron,Rint,Amplitude,freq,1,sample_rate,p,j,memristor_model)
                            Mem.variability(partition,Ron_aging,Roff_aging)
                            weight = (list(weight))
                            Mem.neural_weight([weight], old_weight_max, old_weight_min) 
                            st.session_state.new_u.append(Mem.new_weights())
                    else:
                        old_bias = layer.get_weights()[1]
                    
                    if layer.__class__.__name__ == 'Conv2D':
                        st.session_state.old_weights = old_weights
                        st.session_state.new_weights = []
                        for row in old_weights:
                            # st.session_state.old_weights.append([])
                            st.session_state.new_weights.append([])
                            for weights in row:
                                for weight in weights:
                                    # st.session_state.old_weights[idx].append([weight])
                                    Mem = mem.memristor_models(Roff,Ron,Rint,Amplitude,freq,1,sample_rate,p,j,memristor_model)
                                    Mem.variability(partition,Ron_aging,Roff_aging)
                                    weight = (list(weight))
                                    Mem.neural_weight([weight], old_weight_max, old_weight_min) 
                                    st.session_state.new_weights[idx].append(Mem.new_weights())
                            idx += 1
                    if layer.__class__.__name__ == 'Dense':
                        for row in old_weights:
                            st.session_state.old_weights.append([])
                            for weight in row:
                                # new_w_txt = "Set new weight "+str(memW)+' for '+layer.__class__.__name__+' '+layer.name
                                # new_w = st.number_input(new_w_txt, key=new_w_txt)
                                # set_txt = "set"+str(memW)
                                # memW += 1
                                
                                st.session_state.old_weights[idx].append(weight)
                            idx += 1
                            # st.write('***')

                        Mem = mem.memristor_models(Roff,Ron,Rint,Amplitude,freq,1,sample_rate,p,j,memristor_model)
                        Mem.variability(partition,Ron_aging,Roff_aging)
                        
                        Mem.neural_weight(st.session_state.old_weights, old_weight_max, old_weight_min) 
                        st.session_state.new_weights = Mem.new_weights()

                    for bias in old_bias:
                        
                        # new_b_txt = "Set new bias "+str(memB)+' for '+layer.__class__.__name__+' '+layer.name
                        # new_b = st.number_input(new_b_txt, key=new_b_txt)
                        # set_txt = "setb"+str(memB)
                        # memB += 1
                        #st.write(":heavy_minus_sign:" * 30)
                        
                        st.session_state.old_bias.append(bias)
                    
                    Mem = mem.memristor_models(Roff,Ron,Rint,Amplitude,freq,1,sample_rate,p,j,memristor_model)
                    Mem.variability(partition,Ron_aging,Roff_aging)
                    
                    Mem.neural_weight([st.session_state.old_bias], old_weight_max, old_weight_min) 
                    st.session_state.new_bias = Mem.new_weights()[0]

                    C1,C2 = st.columns(2)
                    with C1:
                        st.write(layer.name,": Weights", np.array(st.session_state.old_weights))
                        if layer.__class__.__name__ == 'LSTM':
                            st.write(layer.name,":hidden Weights", np.array(st.session_state.old_u))
                        st.write(layer.name,": Biases", np.array(st.session_state.old_bias))

                    with C2:
                        st.session_state.new_weights = np.array(st.session_state.new_weights).reshape(shape)
                        st.write(layer.name,": mapped Weights", st.session_state.new_weights)
                        if layer.__class__.__name__ == 'LSTM':
                            st.session_state.new_u = np.array(st.session_state.new_u).reshape(shape_u)
                            st.write(layer.name,":mapped hidden Weights", st.session_state.new_u)
                        st.write(layer.name,": mapped Biases", np.array(st.session_state.new_bias))

                        
                    # apply = st.button("Apply mapped values",key=lyr)
                    # if apply:            
                    st.session_state.new_weights = np.array(st.session_state.new_weights).reshape(shape)
                    if layer.__class__.__name__ == 'LSTM':
                        layer.set_weights([st.session_state.new_weights, st.session_state.new_u, np.array(st.session_state.new_bias)])
                    else:
                        layer.set_weights([st.session_state.new_weights, np.array(st.session_state.new_bias)])
                    # st.success('Successfully applied new mapped wights and biases')

                except Exception as ex:
                    st.error(ex)
                    print(ex)


def get_weights_and_biases(model):

    # Get the current weights of the neural network
    old_weights = np.array(model.get_weights(), dtype=object)
    # print(len(old_weights))
    # print(old_weights)
    # for i in old_weights:
    #     print(len(i))
    df = pd.DataFrame(old_weights)
    
    return df


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


if st.session_state.submittedLayers!=[]:
    st.subheader('Added Layers')
    show_layers(st.session_state.submittedLayers)
    reset = st.button('Reset')

    # resetting the submittedLayers and so the model too
    if reset:
        if st.session_state.snn:
            st.session_state.model = Sequential(TD(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape)))
            st.session_state.submittedLayers = []
        else:
            st.session_state.model = Sequential(tf.keras.layers.InputLayer(input_shape=st.session_state.ip_shape))
            st.session_state.submittedLayers = []


    optimizer,loss,epochs,batch_size = show_compile_fit()
        
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        submitAll = st.button('Submit all')
            
    if submitAll:
        st.session_state.model_fit = run_model(st.session_state.model,loss,optimizer,epochs,batch_size)
        cal_result(st.session_state.model)
        st.session_state.nn_submit = True
        df = get_weights_and_biases(st.session_state.model)
        st.session_state.csv = convert_df(df)

    col1, col2, col3 = st.columns([2,2,2])   
    with col2:
        if st.session_state.csv:
            st.download_button(
                label="Download weights as CSV",
                data= st.session_state.csv,
                file_name='weights_df.csv',
                mime='text/csv',
            )
        
    if st.session_state.nn_submit:
        show_results(st.session_state.model_fit)
        restore = st.button('Restore trained weights')
        if restore:
            st.session_state.model.load_weights('Model_Weights.h5')
        
        if st.session_state.nn_type == 'Hardware':
            set_hardware_weights(st.session_state.model)

    c1,c2,c3 = st.columns(3)
    with c2:
        evaluate = st.button("Evaluate")
    if evaluate:
        cal_result(st.session_state.model)


    if st.session_state.Store!={}:
            df=pd.DataFrame(st.session_state.Store)
            st.table(df)

    