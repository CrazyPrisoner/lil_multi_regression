<h1>  Multi Regression, Tensorflow Serving </h1>
<h2>  Script to visualization data and feed model. </h2>
<p>  Import packages. </p>

      import numpy
      import pandas
      import matplotlib.pyplot as plt
      from mlxtend.preprocessing import one_hot
      
<p>  Import data. </p>

      def import_data():
          dataframe = pandas.read_csv('/home/deka/Desktop/datasets/example.csv')
          return dataframe

<p>  Information about data. </p>

      def info_data():
          data = import_data()
          print(data.head(10))
          print(data.info())
          print(data.describe())
          print(data.corr())

<p>  Data visualization. </p>

          empty_df = pandas.DataFrame(columns=['gas_fuel_flow','hpc_eta','ngp','npt'])

          empty_df.gas_fuel_flow=numpy.array(data['unit3.value_gasflow'].rolling(window=207).mean())
          plt.plot(empty_df.gas_fuel_flow, 'red');

          plt.title("Gas Flow smoothing trend");
          plt.show()

          empty_df.ngp=numpy.array(data['unit3.value_ngp'].rolling(window=208).mean())
          plt.plot(empty_df.ngp, 'green');

          plt.title("NGP smoothing trend");
          plt.show()


          empty_df.hpc_eta=numpy.array(data['unit3.value_eta_a'].rolling(window=318).mean())
          plt.plot(empty_df.hpc_eta, 'blue');

          plt.title("HPC ETA smoothing trend");
          plt.show()

          empty_df.npt=numpy.array(data['unit3.value_npt'].rolling(window=1000).mean())
          plt.plot(empty_df.npt, 'pink');

          plt.title("NPT smoothing trend");
          plt.show()

          return empty_df

<p>  Preapare data for model.  </p>

      def input_data():
          df = info_data() # values
          divide = 14500 # value for divide data on train and test
          df.dropna(inplace=True)
          print(df)
          print(df.head(10),"\n") # print first 10 raws
          print(df.info(),"\n") # print info about dataframe
          print(df.shape,"\n") # print dataframe shape
          print(df.describe(),"\n") # print info about values
          print(df.corr(),"\n") # dataframe correlation

          # Divide data on train and test without shuffle
          train_X = numpy.array(df.values[:divide,0:3])
          train_Y = numpy.array(df.values[:divide,3:])
          test_X = numpy.array(df.values[divide:,0:3])
          test_Y = numpy.array(df.values[divide:,3:])

          return train_X, test_X, train_Y, test_Y

<h2>  Script to train model and save it. </h2>
<p>  Import packages  </p>

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

<p>  Graphics, we call from linear_input_data_exa.py </p>

![grph1](https://user-images.githubusercontent.com/37526996/44617214-df463280-a880-11e8-9cd3-d397e5d7e304.png)
![grph2](https://user-images.githubusercontent.com/37526996/44617215-e0775f80-a880-11e8-83c2-23b5f54fa0a6.png)
![grph3](https://user-images.githubusercontent.com/37526996/44617216-e1a88c80-a880-11e8-9cd8-c31cab2719e9.png)
![grph4](https://user-images.githubusercontent.com/37526996/44617217-e2412300-a880-11e8-8968-58ae7a41d9ff.png)

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

<p>  Train model </p>

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

![predicted-npt](https://user-images.githubusercontent.com/37526996/44617264-5d0a3e00-a881-11e8-9b73-447bc2725535.png)

          plt.plot(predict_test_arr,color='green') # Predicted line
          plt.ylabel('Parameter EGT')
          plt.plot(test_Y,color='blue') # Test line
          plt.show()

<p>  Saving path  </p>

          # Path to save model
          export_path_base = sys.argv[-1]
          export_path = os.path.join(
          tf.compat.as_bytes(export_path_base),
          tf.compat.as_bytes(str(16)))
          print('Exporting trained model to', export_path)
          builder = tf.saved_model.builder.SavedModelBuilder(export_path)

<p>  Signature map  </p>

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

<p>  Saving tensorboard file un folder output  </p>

          print("Done exporting!")
          writer = tf.summary.FileWriter("/home/deka/Desktop/tensorflow-influxdb/tf-inf_model1") # Path to save tensorboard file
          writer.add_graph(sess.graph) # Save tensorboard
          merged_summary = tf.summary.merge_all()

      if __name__ == '__main__':
          tf.app.run()


<h2>  Run server. </h2>
<p>  Run server like this: ```tensorflow_model_server --port=6660 --model_name=deka --model_base_path=/home/deka/Desktop/test_tensorflow_serving/test_serving_model3/```.  </p>
<p>  port="need_give_port", model_name="give_own_name_for_your_model", model_base_path="give_path_to_your_model".  </p> 
<p>  If you run successfully, you can see this, in your command line.  </p>

      2018-08-25 10:37:05.999545: I tensorflow_serving/model_servers/main.cc:157] Building single TensorFlow model file config:  model_name: deka model_base_path: /home/deka/Desktop/tensorflow-influxdb/tf-inf_model2/
      2018-08-25 10:37:05.999789: I tensorflow_serving/model_servers/server_core.cc:462] Adding/updating models.
      2018-08-25 10:37:05.999813: I tensorflow_serving/model_servers/server_core.cc:517]  (Re-)adding model: deka
      2018-08-25 10:37:06.101042: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: deka version: 16}
      2018-08-25 10:37:06.101169: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: deka version: 16}
      2018-08-25 10:37:06.101249: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: deka version: 16}
      2018-08-25 10:37:06.101515: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:360] Attempting to load native SavedModelBundle in bundle-shim from: /home/deka/Desktop/tensorflow-influxdb/tf-inf_model2/16
      2018-08-25 10:37:06.101705: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /home/deka/Desktop/tensorflow-influxdb/tf-inf_model2/16
      2018-08-25 10:37:06.175477: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
      2018-08-25 10:37:06.233216: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:113] Restoring SavedModel bundle.
      2018-08-25 10:37:06.435725: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:148] Running LegacyInitOp on SavedModel bundle.
      2018-08-25 10:37:06.523371: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:233] SavedModel load for tags { serve }; Status: success. Took 421632 microseconds.
      2018-08-25 10:37:06.523497: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:83] No warmup data file found at /home/deka/Desktop/tensorflow-influxdb/tf-inf_model2/16/assets.extra/tf_serving_warmup_requests
      2018-08-25 10:37:06.523718: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: deka version: 16}
      2018-08-25 10:37:06.558123: I tensorflow_serving/model_servers/main.cc:327] Running ModelServer at 0.0.0.0:6660 ...


<h2>  Test Server  </h2>
<p>  Test server with influx  </p>

      import time
      import numpy
      import pandas
      import tensorflow as tf

      from influxdb import InfluxDBClient, DataFrameClient
      from datetime import tzinfo, timedelta, datetime, date

      from grpc.beta import implementations
      from tensorflow.core.framework import types_pb2
      from tensorflow.python.platform import flags
      from tensorflow_serving.apis import predict_pb2
      from tensorflow_serving.apis import prediction_service_pb2

      USER = 'test'
      PASSWORD = '12345'
      DBNAME = 'example'
      HOST = '192.168.4.33'
      PORT = 8086

      tf.app.flags.DEFINE_string('server', 'localhost:6660',
                                 'inception_inference service host:port')
      FLAGS = tf.app.flags.FLAGS

      def main():
          # Connect to server
          client = InfluxDBClient(host=HOST, port=PORT, username=USER, password=PASSWORD, database=DBNAME)
          print("connect to Influxdb", DBNAME, HOST, PORT)
          # Time
          dt1 = datetime.now()
          dt1 = dt1 - timedelta(hours=6)
          dt = dt1.isoformat()
          # Query to database
          query = 'SELECT "value_gasflow", "value_eta_a", "value_ngp", "value_npt" FROM "example"."autogen"."unit3" WHERE time > now() - 4h'
          data = DataFrameClient(host=HOST, \
                                             username=USER, \
                                             password=PASSWORD, \
                                             database=DBNAME)
          dict_query = data.query(query)
          df = pandas.DataFrame(data=dict_query['unit3'])
          index = df.index
          empty_df = pandas.DataFrame(columns=['gas_fuel_flow','hpc_eta','ngp','npt','prediction'],index=index)
          empty_df.gas_fuel_flow = df['value_gasflow']
          empty_df.hpc_eta = df['value_eta_a']
          empty_df.ngp = df['value_ngp']
          empty_df.npt = df['value_npt']
          strafe = numpy.array(empty_df.values[:,0:3])
          out_pp = numpy.float32(strafe)
          # Prepare request
          request = predict_pb2.PredictRequest()
          request.model_spec.name = 'deka'
          request.inputs['inputs'].dtype = types_pb2.DT_FLOAT
          request.inputs['inputs'].CopyFrom(
              tf.contrib.util.make_tensor_proto(out_pp))
          request.output_filter.append('outputs')
          # Send request
          host, port = FLAGS.server.split(':')
          channel = implementations.insecure_channel(host, int(port))
          stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
          prediction = stub.Predict(request, 5.0)  # 5 secs timeout
          floats = prediction.outputs['outputs'].float_val
          predicted_array = numpy.array(floats)
          empty_df.prediction = predicted_array
          #print(empty_df)
          json_body = [{
                 "measurement": "prediction",
                 "tags": {"type": "npt predict"},
                 "time": dt,
                 "fields": {"gas_fuel_flow" : empty_df.gas_fuel_flow[-1],
                      "hpc_eta": empty_df.hpc_eta[-1],
                      "ngp": empty_df.ngp[-1],
                      "npt": empty_df.npt[-1],
                      "prediction": empty_df.prediction[-1]}
          }]
          client.write_points(json_body, database="example")
          #print(json_body)

          client.close()

      if __name__ == '__main__':
          main()

<p> Test Server with dataframe </p>

      from grpc.beta import implementations
      import tensorflow as tf
      import numpy
      import pandas
      from linear_input_data_exa import *

      from tensorflow.core.framework import types_pb2
      from tensorflow.python.platform import flags
      from tensorflow_serving.apis import predict_pb2
      from tensorflow_serving.apis import prediction_service_pb2


      tf.app.flags.DEFINE_string('server', 'localhost:6660',
                                 'inception_inference service host:port')
      FLAGS = tf.app.flags.FLAGS


      def main(_):
          x_tr,x_te,y_tr,y_te = input_data()
          x_test_arr = numpy.float32(x_te)
          # Prepare request
          request = predict_pb2.PredictRequest()
          request.model_spec.name = 'deka'
          request.inputs['inputs'].dtype = types_pb2.DT_FLOAT
          #request.inputs['inputs'].float_val.append(feed_value2)
          request.inputs['inputs'].CopyFrom(
              tf.contrib.util.make_tensor_proto(x_test_arr))
          request.output_filter.append('outputs')
          # Send request
          host, port = FLAGS.server.split(':')
          channel = implementations.insecure_channel(host, int(port))
          stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
          prediction = stub.Predict(request, 5.0)  # 5 secs timeout
          floats = prediction.outputs['outputs'].float_val
          pred_arr = numpy.array(floats)
          pred_df = pandas.DataFrame(data=pred_arr)
          print(pred_df)
          plt.plot(pred_arr, 'pink');
          plt.plot(y_te, 'blue')
          plt.title("NPT prediction");
          plt.show()


      if __name__ == '__main__':
          tf.app.run()

GG WP
