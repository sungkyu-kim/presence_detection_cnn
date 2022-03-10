import numpy as np


def append_array(array_a, array_b, axis=0):
    if array_a.size == 0:
        array_a = array_a.astype(array_b.dtype)
        array_a = array_a.reshape((0,) + array_b.shape[1:])
    array_a = np.concatenate([array_a, array_b], axis)
    return array_a


def reshape_func(d, subcarrier_spacing):
    print(f'd1 : {d.shape}')
    d = d[..., ::subcarrier_spacing]
    print(f'd2 : {d.shape}')
    d = np.transpose(d, [0, 1, 4, 3, 2])
    print(f'd3 : {d.shape}')
    d = d.reshape(d.shape[:-2] + (-1,))
    print(f'd4 : {d.shape}')
    return d


def shape_conversion(d, l):
    temp_d = d[:, int(d.shape[1] / 2 - l / 2):int(d.shape[1] / 2 + l / 2), ...]
    d = temp_d
    return d


def fft_func(data, fft_shape, num_dims):
    temp_data = data
    if num_dims == 1:
        temp_data = np.abs(np.fft.fft(temp_data, n=fft_shape[0], axis=1))
        # temp_data /= np.sum(temp_data, axis=(1,), keepdims=True)
        temp_data = np.fft.fftshift(temp_data, axes=(1,))
    else:
        temp_data = np.abs(np.fft.fft2(temp_data, s=fft_shape, axes=(1, 2)))
        # temp_data /= np.sum(temp_data, axis=(1,2), keepdims=True)
        temp_data = np.fft.fftshift(temp_data, axes=(1, 2))
    temp_data = np.log10(temp_data + 1)
    return temp_data


def obtain_angle(symbol_data):
    angle_data = np.zeros(symbol_data.shape[:-1] + (symbol_data.shape[-1] - 3,))
    for i in range(3):
        diff_data = symbol_data[..., i * 3 + 1:i * 3 + 3] / symbol_data[..., i * 3:i * 3 + 1]
        angle_data[..., 2 * i:2 * i + 2] = np.angle(diff_data)
    return angle_data


def sp_func(d, do_fft, fft_shape):
    print(f'sp_func d.shape {d.shape}')
    phase = obtain_angle(np.copy(d))
    print(f'sp_func phase.shape1 {phase.shape}')
    phase = phase.astype(np.float32)
    print(f'sp_func phase.shape2 {phase.shape}')
    ampl = np.abs(d)
    print(f'sp_func ampl.shape1 {ampl.shape}')
    # normalize amplitude along time axis
    ampl = ampl / ampl[:, :1, ...]
    print(f'sp_func ampl.shape2 {ampl.shape}')
    ampl = ampl.astype(np.float32)

    total_instance = phase.shape[0]
    if do_fft:
        out = np.zeros(((ampl.shape[0],) + fft_shape + (ampl.shape[-1], 2)), dtype=np.float32)
        print(f'sp_func fft out.shape1 {out.shape}')
        for i in range(0, total_instance, 5000):
            num = min(total_instance - i, 5000)
            # ampl 2D fft
            out[i:i + num, ..., 0] = fft_func(np.copy(ampl[i:i + num, ...]), fft_shape, 2)
            # phase 1D fft
            unwrap_phase = np.unwrap(phase[i:i + num, ...], axis=1)
            out[i:i + num, ..., :unwrap_phase.shape[-1], 1] = fft_func(unwrap_phase, fft_shape, 1)
        print(f'sp_func fft out.shape2 {out.shape}')
        return out
    else:
        out = np.zeros((ampl.shape + (2,)), dtype=np.float32)
        out[..., 0] = ampl
        print(f'sp_func out.shape1 {out.shape}')
        # phase unwrapping
        for i in range(0, total_instance, 5000):
            num = min(total_instance-i, 5000)
            unwrap_phase = np.unwrap(phase[i:i+num, ...],axis=1)
            out[i:i+num,...,:unwrap_phase.shape[-1], 1] = unwrap_phase
        print(f'sp_func out.shape2 {out.shape}')
        return out
