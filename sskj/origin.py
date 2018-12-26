# coding = utf-8
import time, collections, sys, os
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import rewriter_config_pb2


TRAINING_STEPS = 20000  # 最大训练步数
SAVE_CHECKPOINT_SECS = 600  # 每隔600秒保存一次模型
DATA_PATH = "./data/"  # 数据位置


MODEL = "small"  # small, medium, large
SAVE_PATH = None


Py3 = sys.version_info[0] == 3


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y


def export_state_tuples(state_tuples, name):
  for state_tuple in state_tuples:
    tf.add_to_collection(name, state_tuple.c)
    tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state_tuples, name, num_replicas):
  restored = []
  for i in range(len(state_tuples) * num_replicas):
    c = tf.get_collection_ref(name)[2 * i + 0]
    h = tf.get_collection_ref(name)[2 * i + 1]
    restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
  return tuple(restored)


def with_prefix(prefix, name):
  return "/".join((prefix, name))


def with_autoparallel_prefix(replica_id, name):
  return with_prefix("AutoParallel-Replica-%d" % replica_id, name)


class UpdateCollection(object):

  def __init__(self, metagraph, model):
    self._metagraph = metagraph
    self.replicate_states(model.initial_state_name)
    self.replicate_states(model.final_state_name)
    self.update_snapshot_name("variables")
    self.update_snapshot_name("trainable_variables")

  def update_snapshot_name(self, var_coll_name):
    var_list = self._metagraph.collection_def[var_coll_name]
    for i, value in enumerate(var_list.bytes_list.value):
      var_def = variable_pb2.VariableDef()
      var_def.ParseFromString(value)
      if var_def.snapshot_name != "Model/global_step/read:0":
        var_def.snapshot_name = with_autoparallel_prefix(
            0, var_def.snapshot_name)
      value = var_def.SerializeToString()
      var_list.bytes_list.value[i] = value

  def replicate_states(self, state_coll_name):
    state_list = self._metagraph.collection_def[state_coll_name]
    num_states = len(state_list.node_list.value)
    for i in range(num_states):
      state_list.node_list.value.append(state_list.node_list.value[i])
    for i in range(num_states):
      state_list.node_list.value[i] = with_autoparallel_prefix(0, state_list.node_list.value[i])


def auto_parallel(metagraph, model):
  from tensorflow.python.grappler import tf_optimizer
  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers.append("autoparallel")
  rewriter_config.auto_parallel.enable = True
  rewriter_config.auto_parallel.num_replicas = 1
  optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, metagraph)
  metagraph.graph_def.CopyFrom(optimized_graph)
  UpdateCollection(metagraph, model)


class PTBInput(object):
  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    self._global_step=tf.train.get_or_create_global_step()
    size = config.hidden_size
    vocab_size = config.vocab_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=tf.float32)
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph(inputs, config, is_training)

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True)

    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=self._global_step)

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    return self._build_rnn_graph_lstm(inputs, config, is_training)


  def _get_lstm_cell(self, config, is_training):
   return tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, tf.float32)
    state = self._initial_state
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    self._name = name
    ops = {with_prefix(self._name, "cost"): self._cost}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = with_prefix(self._name, "initial")
    self._final_state_name = with_prefix(self._name, "final")
    export_state_tuples(self._initial_state, self._initial_state_name)
    export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(with_prefix(self._name, "cost"))[0]
    num_replicas = 1
    self._initial_state = import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name

  @property
  def m_global_step(self):
    return self._global_step


class SmallConfig(object):
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000



def run_epoch(session, model, eval_op=None, verbose=False):
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]

    costs += cost
    iters += model.input.num_steps
  return np.exp(costs / iters)


def get_config():
  config = None
  if MODEL == "small":
    config = SmallConfig()
  else:
    raise ValueError("Invalid model: %s", MODEL)
  return config


global m
global config
start_time = time.time()


# 实现创建模型接口，在这里，我们没有用到传入的参数is_chief（代表是否是分布式中的主节点）
def build_model(is_chief):
    global m
    global config
    raw_data = ptb_raw_data(DATA_PATH)
    train_data, valid_data, test_data, _ = raw_data
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)


# 实现训练接口，step是本地的step数，每调用一次train_model函数，该step会+1
def train_model(session, step):
    global m
    global config
    for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        yield "Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr))  # 用yield传出需要打印的信息

        state = session.run(m.initial_state)
        feed_dict = {}
        costs = 0.0
        iters = 0
        start_time = time.time()
        for step in range(m.input.epoch_size):
            for j, (c, h) in enumerate(m.initial_state):
                feed_dict[c] = state[j].c
                feed_dict[h] = state[j].h
                cost, state, _ = session.run([m.cost, m.final_state, m.train_op], feed_dict)
                costs += cost
                iters += m.input.num_steps

            if step % 10 == 0:
                current_info = "%.3f perplexity: %.3f speed: %.0f wps ,global_step is %s ,time count %s" % (
                step * 1.0 / m.input.epoch_size, np.exp(costs / iters),
                iters * m.input.batch_size /
                (time.time() - start_time),session.run(m.m_global_step),(time.time() - start_time))
                yield current_info  # 用yield传出需要打印的信息
