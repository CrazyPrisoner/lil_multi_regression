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
