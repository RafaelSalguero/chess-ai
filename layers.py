from numba import njit
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