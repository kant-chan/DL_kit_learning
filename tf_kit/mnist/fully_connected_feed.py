'''
completed from scratch, mnist data load manually, use utils.py file.

load data form ubyte file, not pickle all to CPU only once, is by batch size
'''
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import os

import mnist
import utils

hidden1 = 256
hidden2 = 256
batch_size = 10
learning_rate = 0.1
max_step = 60000//100 # totals / batch_size
epochs = 10
log_dir = './log'

def do_eval(sess, eval_correct, imgs_placeholder, labels_placeholder, data_set):
    true_count = 0
    for step in range(10000//batch_size):
        images_feed, labels_feed = data_set.next_batch_shuffle()
        images_feed = images_feed.reshape((batch_size, -1))
        feed_dict = {
            imgs_placeholder: images_feed,
            labels_placeholder: labels_feed
        }
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / 10000
    return precision

def run_training():
    with tf.Graph().as_default():
        imgs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        logits = mnist.inference(imgs_placeholder, hidden1, hidden2)
        loss = mnist.loss(logits, labels_placeholder)
        loss = tf.reduce_mean(loss)
        train_op = mnist.training(loss, learning_rate)
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        sess.run(init)

        for epoch in range(epochs):
            print("##########", epoch, "##########")
            # load data class init
            batch = utils.Batch(utils.train_images_file, utils.train_labels_file, batch_size)

            for step in range(max_step):
                # load data by batch
                images_feed, labels_feed = batch.next_batch_shuffle()
                images_feed = images_feed.reshape((batch_size, -1))

                start_time = time.time()
                feed_dict = {
                    imgs_placeholder: images_feed,
                    labels_placeholder: labels_feed
                }
                _, loss_value = sess.run([train_op, loss],
                                        feed_dict=feed_dict)
                duration = time.time() - start_time

                if step % 100 == 0:
                    # evaluate accuracy
                    if step % 200 == 0 or (step + 1) == max_step:
                        eval_batch = utils.Batch(utils.train_images_file, utils.train_labels_file, batch_size)
                        precision = do_eval(sess, eval_correct, imgs_placeholder, labels_placeholder, eval_batch)
                        print("step: %5d, loss: %4.2f, precision: %0.04f" % (step, loss_value, precision))
                    else:
                        print("step: %5d, loss: %4.2f (%.3f sec)" % (step, loss_value, duration))

                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # save checkpoint
                # if (step + 1) % 1000 == 0 or (step) == max_step:
                #     checkpoint_file = os.path.join('./log', 'model.ckpt')
                #     saver.save(sess, checkpoint_file, global_step=step)
            

def main():
    run_training()

if __name__ == "__main__":
  main()