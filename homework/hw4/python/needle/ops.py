"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, (-lhs) * out_grad / (rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        curr_axes = list(range(len(a.shape)))
        if self.axes:
            curr_axes[self.axes[0]], curr_axes[self.axes[1]] = self.axes[1], self.axes[0]
            return a.permute(curr_axes)
        else:
            curr_axes[a.ndim - 2], curr_axes[a.ndim - 1] = a.ndim - 1, a.ndim - 2
            return a.permute(curr_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if node.inputs[0].shape == self.shape:
            return out_grad
        shape_len = len(self.shape)
        input_len = len(node.inputs[0].shape)
        sum_axis = [i for i in range(shape_len)]
        mv_axis = []
        for i in range(input_len):
            if node.inputs[0].shape[input_len - i - 1] == self.shape[shape_len - i - 1]:
                mv_axis.append(shape_len - 1 - i)
        sum_axis = tuple(filter(lambda x: x not in mv_axis, sum_axis))
        return out_grad.sum(sum_axis).reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not self.axes:
            return array_api.summation(a, axis=None)
        else:
            for axis in sorted(self.axes, reverse=True):
                a = array_api.summation(a, axis=axis)
            return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_shape = node.inputs[0].shape
        sum_shape = range(len(node.inputs[0].shape)) if self.axes is None else [self.axes]
        now_shape = list(new_shape)
        for i in sum_shape:
            now_shape[i] = 1
        return broadcast_to(out_grad.reshape(now_shape), new_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        if len(lgrad.shape) > len(lhs.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rgrad.shape) > len(rhs.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.realize_cached_data() > 0, dtype="float32", device=node.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max_dim = Z.max(self.axes, keepdims=True)
        Z_max = Z.max(self.axes)
        return array_api.log(array_api.summation(array_api.exp(Z + (-Z_max_dim).broadcast_to(Z.shape)), self.axes)) + Z_max
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        z_max_dim = Tensor(z.realize_cached_data().max(self.axes, keepdims=True), device=z.device)
        z_exp = exp(z + (-z_max_dim).broadcast_to(z.shape))
        z_exp_sum = summation(z_exp, axes=self.axes)
        grad_z_exp_sum = out_grad / z_exp_sum
        ori_shape = z.shape
        sum_shape = range(len(z.shape)) if self.axes is None else self.axes
        now_shape = list(ori_shape)
        for i in sum_shape:
            now_shape[i] = 1
        return reshape(grad_z_exp_sum, now_shape).broadcast_to(ori_shape) * z_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0]) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        arr_len = len(args)
        curr_shape = list(args[0].shape)
        curr_shape.insert(self.axis, arr_len)
        stack_arr = array_api.empty(curr_shape, device=args[0].device)
        slice_idx = [slice(0, len) for len in curr_shape]
        for i in range(arr_len):
            slice_idx[self.axis] = i
            stack_arr[tuple(slice_idx)] = args[i]
        return stack_arr
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        arr_len = A.shape[self.axis]
        curr_shape = list(A.shape)
        split_arr = []
        slice_idx = [slice(0, len) for len in curr_shape]
        curr_shape.pop(self.axis)
        for i in range(arr_len):
            slice_idx[self.axis] = i
            split_arr.append(array_api.reshape(A[tuple(slice_idx)].compact(), curr_shape))
        return tuple(split_arr)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        dilate_arr = array_api.full(tuple(new_shape), 0, device=a.device)
        slices = [slice(0, len) for len in new_shape]
        for axis in self.axes:
            slices[axis] = slice(0, slices[axis].stop, self.dilation + 1)
        dilate_arr[tuple(slices)] = a
        return dilate_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = int(new_shape[axis] / (self.dilation + 1))
        dilate_arr = array_api.full(tuple(new_shape), 0, device=a.device)
        a_slices = [slice(0, len) for len in a.shape]
        for axis in self.axes:
            a_slices[axis] = slice(0, a_slices[axis].stop, self.dilation + 1)
        dilate_arr = a[tuple(a_slices)]
        return dilate_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: N * H * W * C_in
        # B: K * K * C_in * C_out
        # out: N * (H + 2P - K + 1) // self.stride * (W + 2P - K + 1) // self.stride * C_out
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * C_in
        strided_A = A.as_strided(shape=(N, (H - K + 1) // self.stride, (W - K + 1) // self.stride, K, K, C_in),
                                 strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact(). \
                                reshape((N * (H - K + 1) // self.stride * (W - K + 1) // self.stride, inner_dim))
        out = strided_A @ B.compact().reshape((K * K * C_in, C_out))
        return out.compact().reshape((N, (H - K + 1) // self.stride, (W - K + 1) // self.stride, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        # out_grad: N * (H + 2P - K + 1) // self.stride * (W + 2P - K + 1) // self.stride * C_out
        # W: K * K * C_in * C_out
        # W_transpose: K * K * C_out * C_in
        # X_grad: N * H * W * C_in

        # X: N * H * W * C_in
        # out_grad: N * (H + 2P - K + 1) * (W + 2P - K + 1) * C_out
        # W_grad: K * K * C_in * C_out

        # X_grad = conv(out_grad, W)
        # W_grad = conv(X, out_grad)
        X = node.inputs[0]
        W = node.inputs[1]
        K, _, _, _ = W.shape
        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1) # N * (H + 2P - K + 1) * (W + 2P - K + 1) * C_out
        W_flip = flip(W, (0, 1)) # K * K * C_in * C_out
        W_transpose = transpose(W_flip, (2, 3)) # K * K * C_out * C_in
        X_grad = conv(out_grad, W_transpose, padding=K - 1 - self.padding)

        X_permute = transpose(X, (0, 3)) # C_in * H * W * N
        out_grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2)) # (H + 2P - K + 1) * (W + 2P - K + 1) * N * C_out
        W_grad_transpose = conv(X_permute, out_grad_permute, padding=self.padding) # C_in * K * K * C_out
        W_grad = transpose(transpose(W_grad_transpose, (0, 1)), (1, 2)) # K * K * C_in * C_out
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



