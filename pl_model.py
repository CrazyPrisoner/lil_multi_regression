from __future__ import print_function
import os
import sys
import numpy
rng = numpy.random
import tensorflow as tf
from linear_input_data_exa import *

tf.app.flags.DEFINE_integer('training_iteration', 50000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    	print('Usage: mnist_export.py [--training_iteration=x] '
    	  '[--model_version=y] export_dir')
    	sys.exit(-1)
    if FLAGS.training_iteration <= 0:
    	print('Please specify a positive value for training iteration.')
    	sys.exit(-1)
    if FLAGS.model_version <= 0:
    	print('Please specify a positive value for version number.')
    	sys.exit(-1)

    # Hyperparameters
    learning_rate = 1e-5
    display_step = 1
    batch_size = 100
    sess = tf.InteractiveSession()
    # Data vizualization and train and test values
    train_x,test_x,train_y,test_y=input_data()
    # Trainin example, as requested (Issue #1)
    train_X = numpy.asarray(train_x)
    train_Y = numpy.asarray(train_y)
    n_samples = train_Y.shape[0]
    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray(test_x)
    test_Y = numpy.asarray(test_y)
    # tf Graph Input
    X = tf.placeholder(tf.float32, shape = [None, 3],name="first_placeholder")
    Y = tf.placeholder(tf.float32, shape = [None, 1],name="second_placeholder")
    # Set model weights
    W = tf.cast(tf.Variable(rng.randn(3, 1), name="weight"), tf.float32)
    b = tf.Variable(rng.randn(), name="bias")
    # Construct a linear model
    pred = tf.add(tf.matmul(X, W), b, name="prediction")
    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Initialize all variables
    sess.run(init)
    # Train model
    print('Training model...')
    # Fit all training data
    total_batch = int(len(train_X)/batch_size)
    for epoch in range(FLAGS.training_iteration):
        for i in range(total_batch):
            if(i == total_batch):
                break
            batch_xs = train_X[batch_size*i:batch_size*(i+1),:]
            batch_ys = train_Y[batch_size*i:batch_size*(i+1),:]
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    # Test our model
    predict_test = sess.run(pred, feed_dict={X: test_X})
    predict_test_arr = numpy.array(predict_test)
    print("Prediction :",predict_test)
    plt.plot(predict_test_arr,color='green') # Predicted line
    plt.ylabel('Parameter EGT')
    plt.plot(test_Y,color='blue') # Test line
    plt.show()
    # Path to save model
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(16)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # Build the signature_def_map.
    regression_inputs = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take prediction
    regression_outputs_prediction = tf.saved_model.utils.build_tensor_info(pred) # Save predcition function
    regression_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
    inputs={
        tf.saved_model.signature_constants.REGRESS_INPUTS:regression_inputs
    },
    outputs={
        tf.saved_model.signature_constants.REGRESS_OUTPUTS:regression_outputs_prediction,
    },
    method_name=tf.saved_model.signature_constants.REGRESS_METHOD_NAME
    ))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take prediction
    tensor_info_y = tf.saved_model.utils.build_tensor_info(cost) # Save cost function

    prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'input_value':tensor_info_x},
    outputs={'output_value':tensor_info_y},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_value':
            prediction_signature,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            regression_signature,
    },
    legacy_init_op = legacy_init_op)

    builder.save()
    print("Done exporting!")
    writer = tf.summary.FileWriter("/home/deka/Desktop/tensorflow-influxdb/tf-inf_model1") # Path to save tensorboard file
    writer.add_graph(sess.graph) # Save tensorboard
    merged_summary = tf.summary.merge_all()

if __name__ == '__main__':
    tf.app.run()
