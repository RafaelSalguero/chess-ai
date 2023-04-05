from collections import OrderedDict
import numpy as np
from numba import njit, generated_jit, typeof, optional
from numba.typed import List
from numba.experimental import jitclass

type_input = -1
type_conv2d = 0
type_swish2d = 1
type_swish1d = 2
type_sigmoid1d = 3
type_flatten = 4
type_dense = 5
type_avg_pooling = 6
type_relu2d = 7
type_relu1d = 8

spec = OrderedDict()
spec['type'] = typeof(0)
spec['data1d'] = optional(typeof(List([np.array([0], np.float32)])))
spec['data2d'] = optional(typeof(List([np.array([[0]], np.float32)])))
spec['data3d'] = optional(typeof(List([np.array([[[0]]], np.float32)])))
spec['data4d'] = optional(typeof(List([np.array([[[[0]]]], np.float32)])))

@jitclass(spec)
class KerasNumbaLayer:
    def __init__(self, type: int, data1d, data2d, data3d, data4d):
        self.type = type
        self.data1d = data1d
        self.data2d = data2d
        self.data3d = data3d
        self.data4d = data4d

def _get_activation_layer(dims, activation, output_shape):
        output_array = np.empty(output_shape, np.float32)
        if(activation == 'swish'):
            if(dims == 2):
                return KerasNumbaLayer(type_swish2d, None, None, List([output_array]), None)
            elif(dims == 1):
                return KerasNumbaLayer(type_swish1d, List([output_array]), None, None, None)
            else:
                raise Exception(F"Unsupported dims")
        elif(activation == 'sigmoid'):
            if(dims == 1):
                return KerasNumbaLayer(type_sigmoid1d, List([output_array]), None, None, None)
            else:
                raise Exception(F"Unsupported dims")
        elif(activation == 'relu'):
            if(dims == 2):
                return KerasNumbaLayer(type_relu2d, None, None, List([output_array]), None)
            if(dims == 1):
                return KerasNumbaLayer(type_relu1d, List([output_array]), None, None, None)
            else:
                raise Exception(F"Unsupported dims")
        elif(activation =="linear"):
            return None
        else:
            raise Exception(f"Only swish and linear activations supported, activation: {activation}")

def _shape_equals(a, b):
    a_len = len(a)
    if(len(b) != a_len):
        return False
    for i in range(a_len):
        if(a[i] != b[i]):
            return False
        
    return True

def get_layer_data(layers):
    """
    Converts a list of keras layers to layer data
    """
    ret = List([KerasNumbaLayer(type_input, None, None, None, None)])
    for layer in layers:
        ret += _get_single_layer_data(layer)
    return ret

def _get_single_layer_data(layer):
    output_shape = layer.output.shape.as_list()[1:]
    output_array = np.empty(output_shape, np.float32)

    ret = List()
    if(layer.name.startswith("conv2d")):
        padding = layer.padding
        strides = layer.strides
        activation = layer.activation.__name__
        kernel = layer.kernel.numpy()
        bias = layer.bias.numpy()

        if(kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0):
            raise Exception(f"Even sized kernels for conv2d layers not supported")
        if(padding != "same"):
            raise Exception(f"Padding {padding} not supported")
        if(strides[0] != 1 or strides[1] != 1):
            raise Exception(f"Strides not supported")
        
        ret.append(KerasNumbaLayer(type_conv2d, List([bias]), None, List([output_array]), List([kernel])))

        a_layer = _get_activation_layer(2, activation, output_shape)
        if(a_layer is not None):
            ret.append(a_layer)
    elif(layer.name.startswith("flatten")):
        ret.append(KerasNumbaLayer(type_flatten, List([output_array]), None, None, None))
    elif(layer.name.startswith("dense")):
        activation = layer.activation.__name__
        kernel = layer.kernel.numpy()
        bias = layer.bias.numpy()

        ret.append(KerasNumbaLayer(type_dense, List([output_array, bias]), List([kernel]), None, None))

        a_layer = _get_activation_layer(1, activation, output_shape)
        if(a_layer is not None):
            ret.append(a_layer)
    elif(layer.name.startswith("input") or layer.name.startswith("dropout")):
        return ret
    elif(layer.name.startswith("average_pooling2d")):
        pool_size = np.array(layer.pool_size, dtype=np.float32)
        strides = np.array(layer.strides, dtype=np.float32)

        ret.append(KerasNumbaLayer(type_avg_pooling, List([pool_size, strides]), None, List([output_array]), None ))
    else:
        raise Exception(f"Layer type not supported {layer.name}")
    
    return ret

@generated_jit(nopython=True)
def _set_data(layer: KerasNumbaLayer, data):
    if data.ndim == 1:
        def x(layer: KerasNumbaLayer, data):
            layer.data1d = List([data])
        return x
    elif data.ndim == 3:
        def x(layer: KerasNumbaLayer, data):
            layer.data3d = List([data])
        return x
    else:
        raise Exception(f"Unsupported dims {data.ndim}")


@njit
def calc_layers(input, layers_data: list[KerasNumbaLayer]):
    first_layer = layers_data[0]
    _set_data(first_layer, input)

    for i in range(1, len(layers_data)):
        prev = layers_data[i - 1]
        curr = layers_data[i]

        _calc_layer(prev, curr)

    return layers_data[-1]

@njit
def _calc_layer(prev_layer: KerasNumbaLayer, layer: KerasNumbaLayer):
    layer_type = layer.type
    if(layer_type == type_conv2d):
        input = prev_layer.data3d[0]

        kernel = layer.data4d[0]
        bias = layer.data1d[0]
        output = layer.data3d[0]
        
        _conv2d(input, output, kernel, bias)
    elif(layer_type == type_dense):
        input = prev_layer.data1d[0]

        kernel = layer.data2d[0]
        bias = layer.data1d[1]
        output = layer.data1d[0]
        
        _dense(input, output, kernel, bias)
    elif(layer_type == type_swish2d):
        input = prev_layer.data3d[0]
        output = layer.data3d[0]
        _swish2d(input, output)
    elif(layer_type == type_swish1d):
        input = prev_layer.data1d[0]
        output = layer.data1d[0]
        _swish1d(input, output)
    elif(layer_type == type_sigmoid1d):
        input = prev_layer.data1d[0]
        output = layer.data1d[0]
        _sigmoid1d(input, output)
    elif(layer_type == type_flatten):
        input = prev_layer.data3d[0]
        output = layer.data1d[0]
        _flatten(input, output)
    elif(layer_type == type_relu2d):
        input = prev_layer.data3d[0]
        output = layer.data3d[0]
        _relu2d(input, output)
    elif(layer_type == type_relu1d):
        input = prev_layer.data1d[0]
        output = layer.data1d[0]
        _relu1d(input, output)
    elif(layer_type == type_avg_pooling):
        input = prev_layer.data3d[0]

        pool_size = layer.data1d[0]
        strides = layer.data1d[1]
        output = layer.data3d[0]
        _avgPool2d(input, output, pool_size, strides)
    else:
        raise Exception(f"Unsupported")

@njit
def _flatten(input, output):
    w = input.shape[0]
    h = input.shape[1]
    channels = input.shape[2]

    i = 0
    for x in range(0, w):
        for y in range(0, h):
            for c in range(0, channels):
                output[i] = input[x, y, c]
                i += 1
    
    return output
@njit
def _swish2d(input, output):
    w = input.shape[0]
    h = input.shape[1]
    channels = input.shape[2]
    for x in range(0, w):
        for y in range(0, h):
            for c in range(0, channels):
                input_val = input[x, y, c]
                output[x, y, c] = input_val/(1 + np.exp(-input_val))    
    
    return output

@njit
def _relu2d(input, output):
    w = input.shape[0]
    h = input.shape[1]
    channels = input.shape[2]
    for x in range(0, w):
        for y in range(0, h):
            for c in range(0, channels):
                input_val = input[x, y, c]
                output[x, y, c] = max(np.float32(0), input_val)
    
    return output

@njit
def _relu1d(input, output):
    for i in range(0, len(input)):
        input_val = input[i]
        output[i] = max(np.float32(0), input_val)
    
    return output

@njit
def _avgPool2d(input, output, pool_size, strides):
    in_w = input.shape[0]
    in_h = input.shape[1]
    channels = input.shape[2]
    
    out_w = np.floor_divide(in_w - pool_size[0], strides[0]) + 1
    out_h = np.floor_divide(in_h - pool_size[1], strides[1]) + 1

    count = pool_size[0] * pool_size[1]
    for out_x in range(0, out_w):
        for out_y in range(0, out_h):
               for c in range(0, channels):
                    sum = 0
                    for in_x in range(out_x * strides[0],out_x * strides[0] + pool_size[0]):
                        for in_y in range(out_y * strides[1],out_y * strides[1] + pool_size[1]):
                            sum += input[in_x, in_y, c]

                    avg = sum / count
                    output[out_x, out_y, c] = avg
    
    return output

@njit
def _swish1d(input, output):
    for i in range(0, len(input)):
        input_val = input[i]
        output[i] = input_val/(1 + np.exp(-input_val))    
    
    return output

@njit
def _sigmoid1d(input, output):
    for i in range(0, len(input)):
        input_val = input[i]
        output[i] = 1/(1 + np.exp(-input_val))    
    
    return output

@njit
def _conv2d(input, output, kernel, bias):
    w = input.shape[0]
    h = input.shape[1]
    
    # half kernel width and height
    hkw = kernel.shape[0] // 2
    hkh = kernel.shape[1] // 2

    i_channels = kernel.shape[2]
    o_channels = kernel.shape[3]

    for ox in range(0, w):
        for oy in range(0, h):
            for oc in range(0, o_channels):
                output[ox, oy, oc] = bias[oc]
                for ix in range(max(ox - hkw, 0), min(ox + hkw + 1, w)):
                    for iy in range(max(oy - hkh, 0), min(oy + hkh + 1, h)):
                        for ic in range(0, i_channels):
                            input_value = input[ix, iy, ic]
                            weight = kernel[ix - (ox - hkw), iy - (oy - hkh), ic, oc]
                            output[ox, oy, oc] += input_value * weight

    return output

@njit
def _dense(input, output, kernel, bias):
    for i in range(len(output)):
        output[i] = bias[i]
        for j in range(len(input)):
            output[i] += input[j] * kernel[j, i]
