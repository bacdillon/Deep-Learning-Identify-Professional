import logging
import random
import time

from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import os
import uuid

### UNCOMMENT THESE 2 LINES IF YOU ARE RUNNING THIS ON MACOS AND HAS ERRORS WHEN UPLOAD BUTTON IS CLICKED ###
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

################### PARAMETERS ###################

# folder where the uploaded images are stored
UPLOAD_FOLDER = 'uploads'
# file extensions that are allowed to be uploaded
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

# location of the model file
MODEL_PATH = 'models/retrained_graph.pb'
# location of the label file
LABEL_PATH = 'models/retrained_labels.txt'
# name of the input node
INPUT_NAME = 'Mul'
# name of the output node
OUTPUT_NAME = "final_result"
INPUT_HEIGHT = 299
INPUT_WIDTH = 299
INPUT_MEAN = 128
INPUT_STD = 128

##################################################


################# SETUP the app ##################

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_PATH'] = MODEL_PATH
app.config['LABEL_PATH'] = LABEL_PATH

##################################################



########## FUNCTIONS #############################

def load_labels():
    # initialise empty array
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(app.config['LABEL_PATH']).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def getLabel(idx):
    result = labels[idx]

    return result

def load_graph():
    # Read the graph definition file
    with open(app.config['MODEL_PATH'], 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Load the graph stored in `graph_def` into `graph`
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    # Enforce that no new nodes are added
    graph.finalize()

    # Verify what we can access the list of operations in the graph
    #print(graph.get_operations())
    for op in graph.get_operations():
        print(op.name)

    return graph

def generate_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.




#########################################################
### function to set what are the allowed file extensions
### this helps to prevent unwanted files being uploaded
### e.g. non-image files which cannot be classified
#########################################################
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# All we need to classify an image is:
# `sess` : we will use this session to run the graph (this is thread safe)
# `input_tensor` : we will assign the image to this placeholder
# `output_tensor` : the predictions will be stored here

def classify(file_path):
    #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)#request.args['file_path']
    # file_path = "images/pic1.jpg"
    app.logger.info("Classifying image %s" % (file_path), )

    # Load in an image to classify and preprocess it
    file_reader = tf.read_file(file_path, "file_reader")
    if file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')

        image = imread(file_path)
        image = imresize(image, [INPUT_HEIGHT, INPUT_WIDTH])

        float_caster = image.astype(np.float32)#tf.cast(image, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)#np.expand_dims(float_caster, 0)#tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
        images = tf.divide(tf.subtract(resized, [INPUT_MEAN]), [INPUT_STD])

        with tf.Session() as gsess:
            timages = images.eval()

        # Get the predictions (output of the softmax) for this image
        t = time.time()
        preds = sess.run(output_tensor, {input_tensor: timages})
        dt = time.time() - t
        app.logger.info("Execution time: %0.2f" % (dt * 1000.))

        # Single image in this batch
        predictions = preds[0]
        app.logger.info(predictions)

        # The probabilities should sum to 1
        assert np.isclose(np.sum(predictions), 1)

        # get the index of the highest value
        class_label = np.argmax(predictions)
        # get the highest value
        class_confidence = np.amax(predictions)
        # get the label of the index
        lbl = getLabel(class_label)
        app.logger.info("Image %s classified as %d (%s - %s)" % (file_path, class_label, lbl, class_confidence))

        #return jsonify(predictions.tolist())
        return lbl + " (" + str(class_confidence) + ")"
    else:
        return ''


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get file from request stream
        file = request.files['file']

        # check that file exists in stream and has an allowed file extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # perform classification
            result = classify(file_path)
            # get the label for the result index
            #result_label = getLabel(result)

            # generate a new filename for the uploaded file
            filename = generate_random_string(6) + filename
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # display result on screen
            return render_template('template.html', label=result, imagesource='uploads/' + filename)
    else:
        return render_template('index.html')



########## app run #############################

graph = load_graph()
labels = load_labels()

# Create the session that we'll use to execute the model
sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=1
    )
)
sess = tf.Session(graph=graph, config=sess_config)

# Get the input and output operations
input_op = graph.get_operation_by_name(app.config['INPUT_NAME'])
input_tensor = input_op.outputs[0]
output_op = graph.get_operation_by_name(app.config['OUTPUT_NAME'])
output_tensor = output_op.outputs[0]


# setup the upload url to be visible
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})
# end setup

## start the whole app
if __name__ == '__main__':
    app.run(debug=True, port=8009)