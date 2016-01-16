"""Adadelta for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class AdadeltaOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adadelta algorithm.

  @@__init__
  """

  def __init__(self, learning_rate, decay_rate=0.95, epsilon=1e-8,
               use_locking=False, name="Adadelta"):
    """Construct a new Adadelta optimizer.
    """
    super(AdadeltaOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._decay_rate = decay_rate
    self._epsilon = epsilon
    # Created in Initialize.
    self._learning_rate_tensor = None
    self._decay_rate_tensor = None
    self._epsilon_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "accum_grad", self._name)
      self._zeros_slot(v, "accum_update", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._decay_rate_tensor = ops.convert_to_tensor(self._decay_rate,
                                                    name="decay_rate")
    self._epsilon_tensor = ops.convert_to_tensor(self._epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    accum_grad = self.get_slot(var, "accum_grad")
    accum_update = self.get_slot(var, "accum_update")
    return training_ops.apply_adadelta(
        var, accum_grad, accum_update, self._learning_rate_tensor,
        self._decay_rate_tensor, self._epsilon_tensor, grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    accum_grad = self.get_slot(var, "accum_grad")
    accum_update = self.get_slot(var, "accum_update")
    return training_ops.sparse_apply_adadelta(
        var, accum_grad, accum_update, self._learning_rate_tensor,
        self._decay_rate_tensor, self._epsilon_tensor, grad.values,
        grad.indices, use_locking=self._use_locking)
