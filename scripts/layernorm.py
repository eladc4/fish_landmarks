import numpy as np
import tensorflow as tf
import torch


_axis = -1
_input = np.arange(4*2*5*3).reshape((4, 2, 5, 3)).astype(np.float32)

# keras
ln = tf.keras.layers.LayerNormalization(axis=_axis)
out = ln(_input).numpy()
out2 = (_input-tf.reduce_mean(_input, axis=_axis, keepdims=True)
        )/tf.sqrt(tf.math.reduce_variance(_input, axis=_axis, keepdims=True)+ln.epsilon)

ln_torch = torch.nn.LayerNorm(_input.shape[-1], eps=ln.epsilon)
out_torch = ln_torch(torch.Tensor(_input)).cpu().detach().numpy()

print(out-out_torch)

#
# print('=============================================================================')
# print(_input)
# print(out)
# print(out-out2)
# print(f'axis={_axis} - input shape={_input.shape}, output shape={out.shape} gamma={ln.gamma.shape} beta={ln.beta.shape}')
#
# _axis = (1, 3)
# _input = np.arange(2*5*3).reshape((1, 2, 5, 3)).astype(np.float32)
# ln = tf.keras.layers.LayerNormalization(axis=_axis)
# out = ln(_input)
# out2 = (_input-tf.reduce_mean(_input, axis=_axis, keepdims=True)
#         )/tf.sqrt(tf.math.reduce_variance(_input, axis=_axis, keepdims=True)+ln.epsilon)
#
# print('=============================================================================')
# print(_input)
# print(out)
# print(out-out2)
# print(f'axis={_axis} - input shape={_input.shape}, output shape={out.shape} gamma={ln.gamma.shape} beta={ln.beta.shape}')
#
