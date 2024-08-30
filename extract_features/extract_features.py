import tensorflow as tf
import modeling
import tokenization
import function
import codecs
import collections
import json

tf = tf.compat.v1
flags = tf.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_string("target", "embedding", "")  # embedding or attention
flags.DEFINE_string("layers", "11", "")  #从0开始，0-11
flags.DEFINE_string("bert_config_file", '/Users/wangping/Desktop/pythonProject/bert_config.json', "")
flags.DEFINE_string("vocab_file", '/Users/wangping/Desktop/pythonProject/vocab.txt', "")
flags.DEFINE_bool("do_lower_case", True, "")
flags.DEFINE_string("input_file", '/Users/wangping/Desktop/pythonProject/extract_features/Verified by existing alloys.txt', "")
flags.DEFINE_string("output_file_embedding",
                    '/Users/wangping/Desktop/pythonProject/extract_features/Verified by existing alloys.json', "")
flags.DEFINE_string("output_file_attention",
                    '/Users/wangping/Desktop/pythonProject/extract_features/zhang_em.json', "")
flags.DEFINE_integer("max_seq_length", 128, "")
flags.DEFINE_integer("batch_size",100, "Batch size for predictions.")
flags.DEFINE_string("init_checkpoint", '/Users/wangping/Desktop/pythonProject/extract_features/model.ckpt-107000',
                    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string("master", None, "If using a TPU, the address of the master.")
flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_bool("use_one_hot_embeddings", False, "If True, tf.one_hot will be used for embedding lookups, "
                                                   "otherwise tf.nn.embedding_lookup will be used. On TPUs, "
                                                   "this should be True since it is much faster.")


def main(_):
    #tf.logging.set_verbosity(tf.logging.INFO)
    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    #Parameter settings for tokenizer
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    # [<function.InputExample object>,<function.InputExample object>...]
    examples = function.read_examples(FLAGS.input_file)
    # <function.InputFeatures object> unique_id, tokens ,input_ids ,input_mask ,input_type_ids
    #print(examples)
    features = function.convert_examples_to_features(examples=examples, seq_length=FLAGS.max_seq_length,
                                                     tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    run_config = tf.estimator.RunConfig(
             #save_checkpoints_steps=100,
             #save_summary_steps=50,
             #session_config=tf.ConfigProto(log_device_placement=True)
        )

    model_fn = function.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=layer_indexes,
        target=FLAGS.target,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": FLAGS.batch_size})

    # <function input_fn_builder.<locals>.input_fn at 0x1c72079d0>
    input_fn = function.input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

    if FLAGS.target == 'embedding':
        with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file_embedding, "w")) as writer:  #Output the embedding results of each layer encoder
            for result in estimator.predict(input_fn, yield_single_examples=True):
              unique_id = int(result["unique_id"])
              feature = unique_id_to_feature[unique_id]
              output_json = collections.OrderedDict()
              output_json["linex_index"] = unique_id
              all_features = []
              for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                  layer_output = result["layer_output_%d" % j]
                  layers = collections.OrderedDict()
                  layers["index"] = layer_index
                  layers["values"] = [
                      round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                  ]
                  all_layers.append(layers)

                features = collections.OrderedDict()
                features["token"] = token
                features["layers"] = all_layers
                all_features.append(features)
                #print(all_features)
              output_json["features"] = all_features
              writer.write(json.dumps(output_json) + "\n")

    if FLAGS.target == 'attention':
        with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file_attention, "w")) as writer:  #Output attention weight matrix
            for result in estimator.predict(input_fn, yield_single_examples=True):
                unique_id = int(result["unique_id"])
                # feature = unique_id_to_feature[unique_id]
                # print(feature)
                output_json = collections.OrderedDict()  
                output_json["linex_index"] = unique_id
                all_ws = []

                for (i, layer_index) in enumerate(layer_indexes):
                    layer_output = result["layer_output_%d" % i]  # 
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    heads = collections.OrderedDict()
                    for (j, head) in enumerate(layer_output):  
                        heads['head_%s' % j] = layer_output[j].tolist()  # 128*128
                    layers["attention_ws"] = heads  # layer_output b*heads*128*128
                    all_ws.append(layers)

                output_json["attention_ws"] = all_ws
                writer.write(json.dumps(output_json) + "\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("target")
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_file_embedding")
    flags.mark_flag_as_required("output_file_attention")
    tf.app.run()
