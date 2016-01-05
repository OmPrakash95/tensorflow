from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_decoding_ops

@ops.RegisterShape("NBestListDecoding")
def _NBestListDecodingShape(op):
  state_shapes = [x.get_shape() for x in op.inputs[:-2]]

  return state_shapes + [op.inputs[-1].get_shape()] * 3

ops.NoGradient("NBestListDecoding")
