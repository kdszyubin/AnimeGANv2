from flask import Flask, request, jsonify
from net import generator
import tensorflow as tf
import numpy as np
from tools.utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
port = int(os.getenv("PORT", 5000))

# 创建一个新的 Flask web 服务器
app = Flask(__name__)

# 预加载模型
# 设置默认值
default_checkpoint_dir = 'checkpoint/generator_Hayao_weight'
# 从环境变量中获取值，如果环境变量不存在，将使用默认值
checkpoint_dir = os.getenv('CHECKPOINT_DIR', default_checkpoint_dir)
#checkpoint_dir = 'checkpoint/' + 'generator_Hayao_weight'
test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')

with tf.variable_scope("generator", reuse=False):
    test_generated = generator.G_net(test_real).fake
saver = tf.train.Saver()

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

# Load model
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
else:
    print(" [*] Failed to find a checkpoint")

@app.route('/transform', methods=['POST'])
def transform():
    input_path = request.json['input_path']
    output_path = request.json['output_path']
    img_size=[256,256]
    sample_image = np.asarray(load_test_data(input_path, img_size))
    fake_img = sess.run(test_generated, feed_dict = {test_real : sample_image})
    save_images(fake_img, output_path, None)

    return jsonify({'output_path': output_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

