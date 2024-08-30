import tensorflow as tf


#Create an iterator to read tfrecord files
dataset = tf.data.TFRecordDataset("/Users/wangping/Desktop/pythonProject/text_pretreatment/output_file.tfrecords")

#Define a parsing function to parse each sample in tfrecord
def parse_function(example_proto):
    max_seq_length = 128
    max_predictions_per_seq = 20
    features = {
        "input_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features

#Apply the parsing function to each sample in the dataset using the map function
parsed_dataset = dataset.map(parse_function)

#Create an iterator to traverse the parsed dataset and view its contents
iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)
next_element = iterator.get_next()

#Session version compatibility
tf.compat.v1.disable_eager_execution()
#Print the content of each sample
with tf.compat.v1.Session() as sess:
    while True:
        try:
            features = sess.run(next_element)
            print(features)
        except tf.errors.OutOfRangeError:
            break

