from collections import OrderedDict
import numpy as np
from numba import njit, generated_jit, typeof, optional
from numba.typed import List
from numba.experimental import jitclass

type_input = -1
type_conv2d = 0
type_swish2d = 1

spec = OrderedDict()
spec['type'] = typeof(0)
spec['data1d'] = optional(typeof([np.array([0], np.float32)]))
spec['data2d'] = optional(typeof([np.array([[0]], np.float32)]))
spec['data3d'] = optional(typeof([np.array([[[0]]], np.float32)]))
spec['data4d'] = optional(typeof([np.array([[[[0]]]], np.float32)]))

@jitclass(spec)
class KerasNumbaLayer:
    def __init__(self, type: int, data1d, data2d, data3d, data4d):
        self.type = type
        self.data1d = data1d
        self.data2d = data2d
        self.data3d = data3d
        self.data4d = data4d

def _get_activation_layer(activation):
        if(activation == 'swish'):
            return KerasNumbaLayer(type_swish2d, None, None, None, None)
        elif(activation =="linear"):
            return None
        else:
            raise Exception(f"Only swish and linear activations supported, activation: {activation}")

def shape_equals(a, b):
    a_len = len(a)
    if(len(b) != a_len):
        return False
    for i in range(a_len):
        if(a[i] != b[i]):
            return False
        
    return True

def get_layer_data(layers):
    ret = [KerasNumbaLayer(type_input, None, None, None, None)]
    for layer in layers:
        ret += get_single_layer_data(layer)
    return ret

def get_single_layer_data(layer):
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
        
        ret.append(KerasNumbaLayer(type_conv2d, [bias], None, [output_array], [kernel]))

        a_layer = _get_activation_layer(activation)
        if(a_layer is not None):
            ret.append(a_layer)
    else:
        raise Exception(f"Layer type not supported {layer.name}")
    
    return ret


@njit
def calc_layers(input, layers_data: list[KerasNumbaLayer]):
    layers_data[0].data3d = [input]
    for i in range(1, len(layers_data)):
        prev = layers_data[i - 1]
        curr = layers_data[i]

        calc_layer(prev, curr)

    return layers_data[-1].data3d[0]

@njit
def calc_layer(prev_layer: KerasNumbaLayer, layer: KerasNumbaLayer):
    layer_type = layer.type
    if(layer_type == type_conv2d):
        input = prev_layer.data3d[0]

        kernel = layer.data4d[0]
        bias = layer.data1d[0]
        output = layer.data3d[0]
        
        conv2d(input, output, kernel, bias)
    elif(layer_type == type_swish2d):
        input = prev_layer.data3d[0]
        output = layer.data3d[0]
        swish2D(input, output)

    return output

@njit
def swish2D(input, output):
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
def conv2d(input, output, kernel, bias):
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