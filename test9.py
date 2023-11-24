import tensorflow as tf
# import datetime
from datetime import datetime
tf.compat.v1.disable_eager_execution()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train, y_train)
# print(x_test, y_test)
def create_mnist_dataset(data, labels, batch_size):
  def gen():
    for image, label in zip(data, labels):
        yield image, label
  ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28,28 ), ()))

  return ds.repeat().batch(batch_size)

#train and validation dataset with different batch size
train_dataset = create_mnist_dataset(x_train, y_train, 1024)
valid_dataset = create_mnist_dataset(x_test, y_test, 1024)
# print(type(train_dataset))
# print(type(valid_dataset))
# train_dataset.make_one_short_iterator().
iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
data_init = iterator.initializer()
next_batch = iterator.get_next()
xs = tf.cast(next_batch[0],tf.float32)
ys = tf.cast(next_batch[1],tf.float32)
xs = tf.reshape(xs, [-1, xs.shape[1]*xs.shape[2]])
w = tf.compat.v1.get_variable('weight',shape=[784,1],dtype=tf.float32)
b = tf.compat.v1.get_variable('bias',shape=(1,),dtype=tf.float32)
pred = tf.matmul(xs,w)+b
loss = tf.reduce_mean(tf.square(tf.subtract(ys, pred)))/100000
opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

cnt = 0
logdir = "./logs3"

file_writer = tf.summary.create_file_writer(logdir)
step = tf.compat.v1.get_variable('step',shape=[],dtype=tf.int64,initializer=tf.compat.v1.zeros_initializer())
step_update = step.assign_add(1)
with file_writer.as_default():
    tf.summary.scalar("my_metric", data=loss, step=step)
    all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()

var_init = tf.compat.v1.global_variables_initializer()
writer_flush = file_writer.flush()
with tf.compat.v1.Session() as sess:
    sess.run(file_writer.init())
    sess.run([var_init])
    for cnt in range(100):
      _ , loss_val = sess.run([opt, loss])
      sess.run([step_update])
      sess.run(all_summary_ops)
    sess.run(writer_flush)




