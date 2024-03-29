import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles

import numpy as np
from PIL import Image

def all_zeros(cocotb_array):
    return all ([n.is_resolvable and n.integer == 0 for n in cocotb_array.value])

def equal_of_greater(array1, array2):
    return all ([x >= y for x, y in zip(array1, array2)])

def toSigned8(n):
    n = n & 0xff
    return n | (-(n & 0x80))

def toSigned16(n):
    n = n & 0xffff
    return n | (-(n & 0x8000))

def toSigned32(n):
    n = n & 0xffffffff
    return n | (-(n & 0x80000000))

def accumulators_equal_to(cocotb_array, array):
    N = len(array)
    print ('accumulators: ', [toSigned32(n.integer) for n in cocotb_array.value[0:N]])
    print ('expected:     ', array.tolist())
    return np.all([toSigned32(n.integer) for n in cocotb_array.value[0:N]] == array)

def signals_to_str(cocotb_array):
    return [toSigned32(n.integer) if n.is_resolvable else n.binstr for n in cocotb_array.value]

def array_to_signals(array, bits_per_element):
    large_n = 0
    for n in reversed(array):
        large_n += n
        large_n <<= bits_per_element
    large_n >>= bits_per_element
    return large_n


# generated by ChatGPT v3.5
def load_hex(file_path):
    hex_values = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:
                try:
                    hex_bytes = bytes.fromhex(line)  # Convert hexadecimal string to bytes
                    hex_values.extend(hex_bytes)  # Extend the list with individual byte values
                except ValueError:
                    print(f"Ignoring invalid hexadecimal value(s) in line: {line}")
    return hex_values

### TESTS

# @cocotb.test()
async def blinky_test(dut):
    dut._log.info("start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    dut.BTN1.value = 0
    await ClockCycles(dut.clk, 100)

    dut.BTN1.value = 1
    await ClockCycles(dut.clk, 100)

    assert dut.LED1.value == 0;
    assert dut.LED4.value == 1;

    dut.BTN1.value = 0
    await ClockCycles(dut.clk, 100)

    assert dut.LED1.value == 1;
    assert dut.LED4.value == 0;


# @cocotb.test()
async def mac_test(dut):
    dut._log.info("start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    dut.reset.value = 0
    await ClockCycles(dut.clk, 10)

    dut._log.info("reset")
    dut.reset.value = 1
    await ClockCycles(dut.clk, 10)

    dut._log.info("calculate")
    dut.reset.value = 0
    print ('{:8d}'.format(int(dut.accumulator.value)))
    assert dut.accumulator.value == 0;

    for i in range (256+16):
        await ClockCycles(dut.clk, 1)
        print (dut.accumulator.value)
        print (f'%3d%8d' % (int(dut.progress.value),int(dut.accumulator.value)))

    await ClockCycles(dut.clk, 1)
    assert dut.accumulator.value == 5559680

# @cocotb.test()
async def gemm_processor_test(dut):
    weights_0 = load_hex('dense_weights.txt')
    weights_1 = load_hex('dense_1_weights.txt')
    weights_2 = load_hex('dense_2_weights.txt')
    weights_3 = load_hex('dense_3_weights.txt')
    bias_0    = load_hex('dense_biases.txt')
    bias_1    = load_hex('dense_1_biases.txt')
    bias_2    = load_hex('dense_2_biases.txt')
    bias_3    = load_hex('dense_3_biases.txt')

    inputs    = load_hex('encoded_pos.txt')
    act_0     = load_hex('activation_0.txt')
    act_1     = load_hex('activation_1.txt')
    act_2     = load_hex('activation_2.txt')
    act_3     = load_hex('activation_3.txt')
    act_3_12b = load_hex('activation_3_12bits.txt')

    weights_0_s = np.reshape(np.vectorize(toSigned8)(weights_0), (64, 64))
    weights_1_s = np.reshape(np.vectorize(toSigned8)(weights_1), (64, 64))
    weights_2_s = np.reshape(np.vectorize(toSigned8)(weights_2), (64, 32))
    weights_3_s = np.reshape(np.vectorize(toSigned8)(weights_3), (32, 4))
    bias_0_s = np.vectorize(toSigned8)(bias_0)
    inputs_s = np.reshape(np.vectorize(toSigned8)(inputs[0:64*64]), (64, 64))
    act_0_s = np.reshape(np.vectorize(toSigned8)(act_0[0:64*64]),   (64, 64))
    act_1_s = np.reshape(np.vectorize(toSigned8)(act_1[0:64*64]),   (64, 64))
    act_2_s = np.reshape(np.vectorize(toSigned8)(act_2[0:64*32]),   (64, 32))
    act_3_s = np.reshape(np.vectorize(toSigned8)(act_3[0:64*4]),    (64,  4))

    dut._log.info("start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    dut.reset.value = 0
    await ClockCycles(dut.clk, 4)

    dut._log.info("reset")
    dut.reset.value = 1
    await ClockCycles(dut.clk, 4)
    print (signals_to_str(dut.activations_as_array))
    assert all_zeros(dut.activations)
    assert all_zeros(dut.accumulators)

    dut._log.info("init")
    dut.reset.value = 0
    dut.enable.value = 1
    ### assert all([i == 0 for i in inputs[0:64]]) # first input row are all zeros
    dut.activations.value = array_to_signals(inputs[0:64], 8)

    print('WEIGHTS:')
    for i in range (2):
         print(weights_0[i*64:(i+1)*64])
    print('BIAS:', bias_0)
    print('INPUT:', inputs[0:64])
    # await ClockCycles(dut.clk, 1)

    

    weights = [format(byte, '02X') for byte in weights_0]
    print(weights[0], " ", weights[1], " ", weights[2], " ", weights[3], " ... ", weights[63]);
    print(weights[64], " ", weights[65], " ", weights[66], " ", weights[67], " ... ", weights[127]);
    print("...");
    print(weights[4032], " ", weights[4033], " ", weights[4034], " ", weights[4035], " ... ", weights[4095]);


    print('WEIGHTS:')
    print(weights_0_s[0])
    print(weights_0_s)
    print('BIAS:', bias_0_s)
    print('INPUT:', inputs_s[0:64])

    print('xxx:', (inputs_s   @ weights_0_s    + bias_0_s)[0])

    # print('OUT:', np.maximum((inputs_s   @ weights_0_s    + bias_0_s)[0] // 256 - 1, 0)) 
    # print('OUT:', np.maximum((inputs_s   @ weights_0_s.T  + bias_0_s)[0] // 256 - 1, 0)) 
    # print('OUT:', np.maximum((inputs_s.T @ weights_0_s    + bias_0_s)[0] // 256 - 1, 0))
    # print('OUT:', np.maximum((inputs_s.T @ weights_0_s.T  + bias_0_s)[0] // 256 - 1, 0))

    # print('GEMM:', np.maximum(np.matmul(inputs_s  , weights_0_s  )[0] // 256 - 1, 0)) 
    # print('GEMM:', np.maximum(np.matmul(inputs_s  , weights_0_s.T)[0] // 256 - 1, 0)) 
    # print('GEMM:', np.maximum(np.matmul(inputs_s.T, weights_0_s  )[0] // 256 - 1, 0)) 
    # print('GEMM:', np.maximum(np.matmul(inputs_s.T, weights_0_s.T)[0] // 256 - 1, 0)) 

    # print('GEM_:', np.maximum(np.matmul(inputs_s  , weights_0_s  ).T[0] // 256 - 1, 0)) 
    # print('GEM_:', np.maximum(np.matmul(inputs_s  , weights_0_s.T).T[0] // 256 - 1, 0)) 
    # print('GEM_:', np.maximum(np.matmul(inputs_s.T, weights_0_s  ).T[0] // 256 - 1, 0)) 
    # print('GEM_:', np.maximum(np.matmul(inputs_s.T, weights_0_s.T).T[0] // 256 - 1, 0)) 

    print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
    print(act_0[0:32])
    print('GEMM:', np.minimum(np.maximum(np.matmul(inputs_s, weights_0_s  )[0] // 256 - 1, 0), 63)[0:32])
    print('GEM*:', np.matmul(inputs_s, weights_0_s  )[0])

    ACCUM_DIV = 128
    CLAMP_MAX = 127

    verbose = False # True
    results = []
    for i in range(8):
        acc = 0
        for n in range(64):
            x = inputs_s.flatten()[n]
            w = weights_0_s.flatten()[n*64+i]
            acc = acc + x * w
            if verbose: print(hex(x&0xff), "*", hex(w&0xff), "=", acc, format(acc&0xffff_ffff, '04X'))
        if verbose: print(acc, acc//256, acc//256-1, np.minimum(np.maximum(acc//256-1, 0), 63), "-------------------------------------")
        results.append(np.minimum(np.maximum(acc//256-1, 0), 63))
        results.append(format((acc//256)&0xff, '02X'))
        results.append(acc//256)
    print(results[::3])
    print(results[1::3])
    print(results[2::3])

    new_act_0 = np.minimum(np.maximum(np.matmul(inputs_s, weights_0_s) // ACCUM_DIV, 0), CLAMP_MAX)
    # new_act_0 = (np.maximum(np.matmul(inputs_s, weights_0_s), 0) // 256) * 2
    # print(new_act_0.flatten()[0:32])
    print("================================================================================================")
    print(act_0[0:32])
    print(new_act_0.flatten()[0:32])
    print(act_1[0:32])
    print('GEM^2:', np.minimum(np.maximum(np.matmul(act_0_s*4, weights_1_s  )[0] // 256 - 1, 0), 63)[0:32])
    print('GEMM2:', np.minimum(np.maximum(np.matmul(new_act_0*4, weights_1_s  )[0] // 256 - 1, 0), 63)[0:32])
    print('GEMM2:', np.minimum(np.maximum(np.matmul(new_act_0, weights_1_s  )[0] // 64 - 1, 0), 63)[0:32])
    print('GEM*2:', np.matmul(new_act_0, weights_1_s  )[0])

    verbose = False # True
    results = []
    for i in range(4):
        acc = 0
        for n in range(64):
            x = new_act_0.flatten()[n]
            w = weights_1_s.flatten()[n*64+i]
            acc = acc + x * w
            if verbose: print(hex(x&0xff), "*", hex(w&0xff), "=", acc, format(acc&0xffff_ffff, '04X'))
        if verbose: print(acc, acc//256, acc//256-1, np.minimum(np.maximum(acc//256-1, 0), 63), "-------------------------------------")
        results.append(np.minimum(np.maximum(acc//256-1, 0), 63))
        results.append(format((acc//256)&0xff, '02X'))
        results.append(acc//256)
    print(results[::3])
    print(results[1::3])
    print(results[2::3])

    new_act_1 = np.minimum(np.maximum(np.matmul(new_act_0, weights_1_s) // ACCUM_DIV, 0), CLAMP_MAX)
    print("================================================================================================")

    new_act_2 = np.minimum(np.maximum(np.matmul(new_act_1, weights_2_s) // ACCUM_DIV, 0), CLAMP_MAX)
    print("================================================================================================")

    new_act_3 = np.minimum(np.maximum(np.matmul(new_act_2, weights_3_s) // ACCUM_DIV, 0), CLAMP_MAX)
    print("================================================================================================")

    dut._log.info("calculate")
    dut.enable.value = 1
    ACT_MUL = 1

    # 0th layer
    for i in range (64):
        await ClockCycles(dut.clk, 1)
        if i < 2:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array), 'activations:', signals_to_str(dut.activations_as_array))
        else:
            # print ('pc:', int(dut.pc.value), ' activations:', signals_to_str(dut.activations_as_array)[64:64+32])
            # pass
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
        if i == 0:
            assert all_zeros(dut.accumulators)
        if i >= 0:
            assert [n.integer for n in dut.activations_as_array.value[0:64]] == inputs[0:64]
        assert dut.pc.value == 64*0+i

    for i in range (64):
        await ClockCycles(dut.clk, 1)

        if i == 1: # check previous results while they are still in accumulators!!!
            assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(inputs_s, weights_0_s  )[0][0:32])

        if i == 2: # check previous results here!!!
            print('GEM*1:', np.matmul(inputs_s, weights_0_s  )[0][0:32])
            print ('expected activations:', act_0[0:32])
            print ('         activations:', signals_to_str(dut.activations_as_array)[64:96])
            print ([n.integer for n in dut.accumulators_as_array.value])
            print (np.matmul(inputs_s, weights_0_s  )[0][0:32])
            assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[64:64+32]], act_0[0:32])

        if i < 2:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array), 'activations:', signals_to_str(dut.activations_as_array))
        else:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
        if i == 2:
            assert all_zeros(dut.accumulators)
        if i >= 0:
            assert [n.integer for n in dut.activations_as_array.value[0:64]] == inputs[0:64]
        assert dut.pc.value == 64*1+i

    # for i in range (3):
    #     await ClockCycles(dut.clk, 1)
    #     if i == 1: # check previous results while they are still in accumulators!!!
    #         assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(inputs_s, weights_0_s  )[0][32:64])


    # print ('expected activations:', act_0[32:64])
    # print ('         activations:', signals_to_str(dut.activations_as_array)[96:128])
    # assert [n.integer for n in dut.activations_as_array.value[96:128]] >= act_0[32:64]

    # 1st layer
    for i in range (64):
        await ClockCycles(dut.clk, 1)

        if i == 1: # check previous results while they are still in accumulators!!!
            assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(inputs_s, weights_0_s  )[0][32:64])

        if i == 2: # check previous results here!!!
            print('GEM*1:', np.matmul(inputs_s, weights_0_s  )[0][32:64])
            print ('expected activations:', act_0[32:64])
            print ('         activations:', signals_to_str(dut.activations_as_array)[96:128])
            assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[96:128]], act_0[32:64])

        if i < 2:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array), 'activations:', signals_to_str(dut.activations_as_array))
        else:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
        # if i == 2:
        #     assert all_zeros(dut.accumulators)
        assert dut.pc.value == 64*2+i

    ACT_MUL = 1
    for i in range (64):
        await ClockCycles(dut.clk, 1)

        if i == 1: # check previous results while they are still in accumulators!!!
            assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(new_act_0, weights_1_s  )[0][0:32])

        if i == 2: # check previous results here!!!
            print('GEM*2:', np.matmul(new_act_0, weights_1_s  )[0][0:32])
            print ('expected activations:', act_1[0:32])
            print ('         activations:', signals_to_str(dut.activations_as_array)[0:32])
            # assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[0:32]], act_1[0:32])
            # print ('expected activations:', act_0[32:64])
            # print ('         activations:', signals_to_str(dut.activations_as_array)[96:128])
            # assert [n.integer for n in dut.activations_as_array.value[96:128]] >= act_0[32:64]

        if i < 2:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array), 'activations:', signals_to_str(dut.activations_as_array))
        else:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
        # if i == 2:
        #     assert all_zeros(dut.accumulators)
        # if i >= 0:
        #   assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[64:128]], act_0[0:64])
        assert dut.pc.value == 64*3+i

    # 2nd layer
    for i in range (64):
        await ClockCycles(dut.clk, 1)

        if i == 1: # check previous results while they are still in accumulators!!!
            assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(new_act_0, weights_1_s  )[0][32:64])

        if i == 2: # check previous results here!!!
            print('GEM*2:', np.matmul(new_act_0, weights_1_s  )[0][0:32])
            print ('expected activations:', act_1[0:32])
            print ('         activations:', signals_to_str(dut.activations_as_array)[0:32])
            # assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[0:32]], act_1[0:32])
            # print ('expected activations:', act_0[32:64])
            # print ('         activations:', signals_to_str(dut.activations_as_array)[96:128])
            # assert [n.integer for n in dut.activations_as_array.value[96:128]] >= act_0[32:64]

        if i < 2:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array), 'activations:', signals_to_str(dut.activations_as_array))
        else:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
        # if i == 2:
        #     assert all_zeros(dut.accumulators)
        # if i >= 0:
        #   assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[64:128]], act_0[0:64])
        assert dut.pc.value == 64*4+i

    # 3rd layer
    for i in range (32):
        await ClockCycles(dut.clk, 1)

        if i == 1: # check previous results while they are still in accumulators!!!
            assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(new_act_1, weights_2_s  )[0][0:32])

        if i == 2: # check previous results here!!!
            print('GEM*3', np.matmul(new_act_1, weights_2_s  )[0][0:32])
            print ('expected activations:', act_2[0:32])
            print ('         activations:', signals_to_str(dut.activations_as_array)[64:64+32])
            # assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[0:32]], act_1[0:32])
            # print ('expected activations:', act_0[32:64])
            # print ('         activations:', signals_to_str(dut.activations_as_array)[96:128])
            # assert [n.integer for n in dut.activations_as_array.value[96:128]] >= act_0[32:64]

        if i < 2:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array), 'activations:', signals_to_str(dut.activations_as_array))
        else:
            print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
        # if i == 2:
        #     assert all_zeros(dut.accumulators)
        # if i >= 0:
        #   assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[64:128]], act_0[0:64])
        assert dut.pc.value == 64*5+i

    for i in range (3):
        await ClockCycles(dut.clk, 1)
        if i == 1: # check previous results while they are still in accumulators!!!
            print()
            assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(new_act_2, weights_3_s  )[0][0:4])

        if i == 2: # check previous results here!!!
            print('GEM*4', np.matmul(new_act_2, weights_3_s  )[0][0:4])
            print ('expected activations:', act_3[0:4])
            print ('         activations:', signals_to_str(dut.activations_as_array)[0:4])

    print (np.matmul(new_act_2, weights_3_s  )[0][0:3] // 128)
    print ([toSigned8(n) for n in act_3[0:3]])
    assert np.all(np.matmul(new_act_2, weights_3_s  )[0][0:3] // 128 == [toSigned8(n) for n in act_3[0:3]])

    # print ('expected activations:', act_1[32:64])
    # print ('         activations:', signals_to_str(dut.activations_as_array)[32:64])
    # assert equal_of_greater([n.integer*ACT_MUL for n in dut.activations_as_array.value[32:64]], act_1[32:64])

# @cocotb.test()
async def gemm_processor_test_first_layer_with_array_of_inputs(dut):
    verbose = False

    weights_0 = load_hex('dense_weights.txt')
    bias_0    = load_hex('dense_biases.txt')

    inputs    = load_hex('encoded_pos.txt')
    act_0     = load_hex('activation_0.txt')

    weights_0_s = np.reshape(np.vectorize(toSigned8)(weights_0), (64, 64))
    bias_0_s = np.vectorize(toSigned8)(bias_0)
    inputs_s = np.reshape(np.vectorize(toSigned8)(inputs[0:64*64]), (64, 64))
    act_0_s = np.reshape(np.vectorize(toSigned8)(act_0[0:64*64]),   (64, 64))

    ACCUM_DIV = 128
    CLAMP_MAX = 127
    new_act_0 = np.minimum(np.maximum(np.matmul(inputs_s,  weights_0_s) // ACCUM_DIV, 0), CLAMP_MAX)

    dut._log.info("start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    dut.reset.value = 0
    await ClockCycles(dut.clk, 4)

    dut._log.info("reset")
    dut.reset.value = 1
    await ClockCycles(dut.clk, 4)
    dut.reset.value = 0
    assert all_zeros(dut.activations)
    assert all_zeros(dut.accumulators)

    for n in range (0, 8):
        dut._log.info("calculate pixel " + str(n))
        dut.enable.value = 0
        dut.restart_program = 1;
        if verbose: print ("pixel input:", inputs[64*n:64*(n+1)])
        dut.activations.value = array_to_signals(inputs[64*n:64*(n+1)], 8)
        await ClockCycles(dut.clk, 4)
        dut.restart_program = 0;
        dut.enable.value = 1

        if verbose:
            for i in range (128):
                await ClockCycles(dut.clk, 1)
                if n == 0 and i == 0:
                    assert all_zeros(dut.accumulators)
                if i%64 == 1:
                    print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
                if i%64 == 2:
                    print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, 'activations:', signals_to_str(dut.activations_as_array))
        elif n == 0:
            await ClockCycles(dut.clk, 1)
            assert all_zeros(dut.accumulators)
            await ClockCycles(dut.clk, 127)
        else:
            await ClockCycles(dut.clk, 128)

        for i in range (4):
            await ClockCycles(dut.clk, 1)
            if i == 2: # check previous results here!!!
                if verbose:
                    print ('expected  activations:', act_0_s[n][0:64])
                    print ('simulated activations:', new_act_0[n][0:64])
                    print ('          activations:', signals_to_str(dut.activations_as_array)[64:128])
                assert np.all(new_act_0[n][0:64] == [n.integer for n in dut.activations_as_array.value[64:128]])

# @cocotb.test()
async def gemm_processor_test_array_of_inputs_(dut):
    verbose = True
    weights_0 = load_hex('dense_weights.txt')
    weights_1 = load_hex('dense_1_weights.txt')
    weights_2 = load_hex('dense_2_weights.txt')
    weights_3 = load_hex('dense_3_weights.txt')
    bias_0    = load_hex('dense_biases.txt')
    bias_1    = load_hex('dense_1_biases.txt')
    bias_2    = load_hex('dense_2_biases.txt')
    bias_3    = load_hex('dense_3_biases.txt')

    inputs    = load_hex('encoded_pos.txt')
    act_0     = load_hex('activation_0.txt')
    act_1     = load_hex('activation_1.txt')
    act_2     = load_hex('activation_2.txt')
    act_3     = load_hex('activation_3.txt')
    act_3_12b = load_hex('activation_3_12bits.txt')

    weights_0_s = np.reshape(np.vectorize(toSigned8)(weights_0), (64, 64))
    weights_1_s = np.reshape(np.vectorize(toSigned8)(weights_1), (64, 64))
    weights_2_s = np.reshape(np.vectorize(toSigned8)(weights_2), (64, 32))
    weights_3_s = np.reshape(np.vectorize(toSigned8)(weights_3), (32, 4))
    bias_0_s = np.vectorize(toSigned8)(bias_0)
    inputs_s = np.reshape(np.vectorize(toSigned8)(inputs[0:64*64]), (64, 64))
    act_0_s = np.reshape(np.vectorize(toSigned8)(act_0[0:64*64]),   (64, 64))
    act_1_s = np.reshape(np.vectorize(toSigned8)(act_1[0:64*64]),   (64, 64))
    act_2_s = np.reshape(np.vectorize(toSigned8)(act_2[0:64*32]),   (64, 32))
    act_3_s = np.reshape(np.vectorize(toSigned8)(act_3[0:64*4]),    (64,  4))

    ACCUM_DIV = 128
    CLAMP_MAX = 127
    new_act_0 = np.minimum(np.maximum(np.matmul(inputs_s,  weights_0_s) // ACCUM_DIV, 0), CLAMP_MAX)
    new_act_1 = np.minimum(np.maximum(np.matmul(new_act_0, weights_1_s) // ACCUM_DIV, 0), CLAMP_MAX)
    new_act_2 = np.minimum(np.maximum(np.matmul(new_act_1, weights_2_s) // ACCUM_DIV, 0), CLAMP_MAX)
    new_act_3 = np.minimum(np.maximum(np.matmul(new_act_2, weights_3_s) // ACCUM_DIV, 0), CLAMP_MAX)

    print ([format(x&0xff, '02X') for x in new_act_2[4]])

    dut._log.info("start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    dut.reset.value = 0
    await ClockCycles(dut.clk, 4)

    dut._log.info("reset")
    dut.reset.value = 1
    await ClockCycles(dut.clk, 4)
    dut.reset.value = 0
    assert all_zeros(dut.activations)
    assert all_zeros(dut.accumulators)

    for n in range (0, 16):
        dut._log.info("calculate pixel " + str(n))
        dut.enable.value = 0
        dut.restart_program = 1;
        if verbose: print ("pixel input:", inputs[64*n:64*(n+1)])
        dut.activations.value = array_to_signals(inputs[64*n:64*(n+1)], 8)
        await ClockCycles(dut.clk, 4)
        dut.restart_program = 0;
        dut.enable.value = 1

        for i in range (128+128+64):
            await ClockCycles(dut.clk, 1)
            if n == 0 and i == 0:
                assert all_zeros(dut.accumulators)
            if i%32 == 1:
                print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
            if i%32 == 2:
                print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, 'activations:', signals_to_str(dut.activations_as_array))

        for i in range (4):
            await ClockCycles(dut.clk, 1)
            if i == 1: # check previous results while they are still in accumulators!!!
                assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(new_act_1, weights_2_s  )[n][0:32])
            if i == 2: # check previous results here!!!
                print ('expected  activations:', act_2_s[n][0:32])
                print ('simulated activations:', new_act_2[n][0:32])
                print ('          activations:', signals_to_str(dut.activations_as_array)[64:96])
                assert np.all(new_act_2[n][0:32] == [n.integer for n in dut.activations_as_array.value[64:96]])

        # for i in range (4):
        #     await ClockCycles(dut.clk, 1)
        #     if i == 1: # check previous results while they are still in accumulators!!!
        #         assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(new_act_2, weights_3_s  )[n][0:3])

        #     if i == 12: # check previous results here!!!
        #         print ('expected  activations:', act_3_s[n][0:3])
        #         print ('simulated activations:', new_act_3[n][0:3])
        #         print ('          activations:', signals_to_str(dut.activations_as_array)[0:3])
        #         assert np.all(new_act_3[n][0:3] == [n.integer for n in dut.activations_as_array.value[0:3]])


# @cocotb.test()
async def gemm_processor_test_array_of_inputs(dut):
    verbose = True
    weights_0 = load_hex('dense_weights.txt')
    weights_1 = load_hex('dense_1_weights.txt')
    weights_2 = load_hex('dense_2_weights.txt')
    weights_3 = load_hex('dense_3_weights.txt')
    bias_0    = load_hex('dense_biases.txt')
    bias_1    = load_hex('dense_1_biases.txt')
    bias_2    = load_hex('dense_2_biases.txt')
    bias_3    = load_hex('dense_3_biases.txt')

    inputs    = load_hex('encoded_pos.txt')
    act_0     = load_hex('activation_0.txt')
    act_1     = load_hex('activation_1.txt')
    act_2     = load_hex('activation_2.txt')
    act_3     = load_hex('activation_3.txt')
    act_3_12b = load_hex('activation_3_12bits.txt')

    weights_0_s = np.reshape(np.vectorize(toSigned8)(weights_0), (64, 64))
    weights_1_s = np.reshape(np.vectorize(toSigned8)(weights_1), (64, 64))
    weights_2_s = np.reshape(np.vectorize(toSigned8)(weights_2), (64, 32))
    weights_3_s = np.reshape(np.vectorize(toSigned8)(weights_3), (32, 4))
    bias_0_s = np.vectorize(toSigned8)(bias_0)
    inputs_s = np.reshape(np.vectorize(toSigned8)(inputs[0:64*64]), (64, 64))
    act_0_s = np.reshape(np.vectorize(toSigned8)(act_0[0:64*64]),   (64, 64))
    act_1_s = np.reshape(np.vectorize(toSigned8)(act_1[0:64*64]),   (64, 64))
    act_2_s = np.reshape(np.vectorize(toSigned8)(act_2[0:64*32]),   (64, 32))
    act_3_s = np.reshape(np.vectorize(toSigned8)(act_3[0:64*4]),    (64,  4))

    ACCUM_DIV = 128
    CLAMP_MAX = 127
    new_act_0 = np.minimum(np.maximum(np.matmul(inputs_s,  weights_0_s) // ACCUM_DIV, 0), CLAMP_MAX)
    new_act_1 = np.minimum(np.maximum(np.matmul(new_act_0, weights_1_s) // ACCUM_DIV, 0), CLAMP_MAX)
    new_act_2 = np.minimum(np.maximum(np.matmul(new_act_1, weights_2_s) // ACCUM_DIV, 0), CLAMP_MAX)
    new_act_3 = np.minimum(np.maximum(np.matmul(new_act_2, weights_3_s) // ACCUM_DIV, 0), CLAMP_MAX)

    print ([format(x&0xff, '02X') for x in new_act_2[4]])

    dut._log.info("start")
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    dut.reset.value = 0
    await ClockCycles(dut.clk, 4)

    dut._log.info("reset")
    dut.reset.value = 1
    await ClockCycles(dut.clk, 4)
    dut.reset.value = 0
    assert all_zeros(dut.activations)
    assert all_zeros(dut.accumulators)

    for n in range (0, 16):
        dut._log.info("calculate pixel " + str(n))
        dut.enable.value = 0
        dut.restart_program = 1;
        if verbose: print ("pixel input:", inputs[64*n:64*(n+1)])
        dut.activations.value = array_to_signals(inputs[64*n:64*(n+1)], 8)
        await ClockCycles(dut.clk, 4)
        dut.restart_program = 0;
        dut.enable.value = 1

        for i in range (128+128+64+2+32): # 2 read hazard cycles before the last layer
            await ClockCycles(dut.clk, 1)
            if n == 0 and i == 0:
                assert all_zeros(dut.accumulators)
            if i%32 == 1:
                print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, signals_to_str(dut.accumulators_as_array))
            if i%32 == 2:
                print ('pc:', int(dut.pc.value), 'cmd:', dut.command.value, 'activations:', signals_to_str(dut.activations_as_array))

        # for i in range (4):
        #     await ClockCycles(dut.clk, 1)
        #     if i == 1: # check previous results while they are still in accumulators!!!
        #         assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(new_act_1, weights_2_s  )[n][0:32])
        #     if i == 2: # check previous results here!!!
        #         print ('expected  activations:', act_2_s[n][0:32])
        #         print ('simulated activations:', new_act_2[n][0:32])
        #         print ('          activations:', signals_to_str(dut.activations_as_array)[64:96])
        #         assert np.all(new_act_2[n][0:32] == [n.integer for n in dut.activations_as_array.value[64:96]])

        for i in range (4):
            await ClockCycles(dut.clk, 1)
            if i == 1: # check previous results while they are still in accumulators!!!
                print (np.matmul(new_act_2, weights_3_s  ).shape)
                print ([format(x&0xff, '02X') for x in new_act_2[n]])
                print ([format(acc&0xffff_ffff, '04X') for acc in np.matmul(new_act_2, weights_3_s  )[n][0:3]])
                assert accumulators_equal_to(dut.accumulators_as_array, np.matmul(new_act_2, weights_3_s  )[n][0:3])

            if i == 12: # check previous results here!!!
                print ('expected  activations:', act_3_s[n][0:3])
                print ('simulated activations:', new_act_3[n][0:3])
                print ('          activations:', signals_to_str(dut.activations_as_array)[0:3])
                assert np.all(new_act_3[n][0:3] == [n.integer for n in dut.activations_as_array.value[0:3]])

@cocotb.test()
async def test_pos_encoding(dut):
    inputs    = load_hex('encoded_pos.txt')
    inputs_s  = np.vectorize(toSigned8)(inputs)

    embedding_size = 32
    seed = 127
    np.random.seed(seed)
    rnd = np.random.random_integers(0, 127, embedding_size*4)
    # print (rnd)

    def encode_positions_sawtooth(rnd, embedding_size, x_coords, y_coords):
        # periodic_fn = lambda x : np.modf(x)[0]*2-1 if int(np.modf(x)[1]) % 1 == 0 else (1-np.modf(x)[0])*2-1
        # return np.array([
        #     [periodic_fn(x * rnd[i*2+0] + y * rnd[i*2+1]) for i in range(embedding_size*2)]
        #         for x in x_coords for y in y_coords])

        periodic_fn = lambda x : toSigned8(x&0xFF if (x&0x100) == 0 else 0xFF-(x&0xFF)) / 127.0
        return np.array([
            [periodic_fn(int(x*128) * rnd[i*2+0] + int(y*128) * rnd[i*2+1]) for i in range(embedding_size*2)]
                for x in x_coords for y in y_coords])

    def quantize_array(values):
        d = 127.0
        values = np.clip(values, -1.0, 1.0)
        values *= d
        values = np.floor(values)
        values /= d
        return values

    xs = np.arange(64) / 64
    ys = np.arange(64) / 64
    new_pos_encoding = encode_positions_sawtooth(rnd, embedding_size, xs, ys)
    new_pos_encoding = quantize_array(new_pos_encoding)
    new_pos_encoding = (new_pos_encoding * 127.0).astype(int).flatten()

    assert new_pos_encoding.shape == inputs_s.shape
    for n in range(len(inputs_s)):
        if inputs_s[n] != new_pos_encoding[n]:
            print(n, inputs_s[n], new_pos_encoding[n])
            break
    assert np.all(new_pos_encoding == inputs_s)

@cocotb.test()
async def test_pos_encoding_fixed_point(dut):
    inputs    = np.array(load_hex('encoded_pos.txt'))

    embedding_size = 32
    seed = 127
    np.random.seed(seed)
    rnd = np.random.random_integers(0, 127, embedding_size*4)
    # print (rnd)

    def encode_positions_sawtooth_fixed(rnd, embedding_size, x_coords, y_coords):
        periodic_fn = lambda x : x&0xFF if (x&0x100) == 0 else 0xFF-(x&0xFF)

        return np.array([
            [periodic_fn(x * rnd[i*2+0] + y * rnd[i*2+1]) for i in range(embedding_size*2)]
                for x in x_coords for y in y_coords])

    xs = np.arange(64)*2 # 0..128
    ys = np.arange(64)*2 # 0..128
    new_pos_encoding = encode_positions_sawtooth_fixed(rnd, embedding_size, xs, ys).flatten()
    new_pos_encoding[new_pos_encoding == 128] = 129
    print (inputs[0:128])
    print (new_pos_encoding[0:128])
    for n in range(len(inputs)):
        if inputs[n] != new_pos_encoding[n]:
            print(n, inputs[n], new_pos_encoding[n])
            break
    assert new_pos_encoding.shape == inputs.shape
    assert np.all(new_pos_encoding == inputs)


@cocotb.test()
async def test_net(dut):
    verbose = True
    weights_0 = load_hex('dense_weights.txt')
    weights_1 = load_hex('dense_1_weights.txt')
    weights_2 = load_hex('dense_2_weights.txt')
    weights_3 = load_hex('dense_3_weights.txt')
    bias_0    = load_hex('dense_biases.txt')
    bias_1    = load_hex('dense_1_biases.txt')
    bias_2    = load_hex('dense_2_biases.txt')
    bias_3    = load_hex('dense_3_biases.txt')

    inputs    = load_hex('encoded_pos.txt')
    act_0     = load_hex('activation_0.txt')
    act_1     = load_hex('activation_1.txt')
    act_2     = load_hex('activation_2.txt')
    act_3     = load_hex('activation_3.txt')
    act_4     = load_hex('activation_4.txt')

    weights_0_s = np.reshape(np.vectorize(toSigned8)(weights_0), (64, 64))
    weights_1_s = np.reshape(np.vectorize(toSigned8)(weights_1), (64, 64))
    weights_2_s = np.reshape(np.vectorize(toSigned8)(weights_2), (64, 32))
    weights_3_s = np.reshape(np.vectorize(toSigned8)(weights_3), (32, 4))
    inputs_s = np.reshape(np.vectorize(toSigned8)(inputs[0:4096*64]), (4096, 64))
    act_0_s = np.reshape(np.vectorize(toSigned8)(act_0[0:4096*64]),   (4096, 64))
    act_1_s = np.reshape(np.vectorize(toSigned8)(act_1[0:4096*64]),   (4096, 64))
    act_2_s = np.reshape(np.vectorize(toSigned8)(act_2[0:4096*32]),   (4096, 32))
    act_3_s = np.reshape(np.vectorize(toSigned8)(act_3[0:4096*4]),    (4096,  4))
    act_4_s = np.reshape(np.vectorize(toSigned8)(act_4[0:4096*4]),    (4096,  4))

    ACCUM_DIV = 128
    CLAMP_MAX = 127

    new_act_0 = np.clip(np.matmul(inputs_s,    weights_0_s) // (ACCUM_DIV *  4),          0, CLAMP_MAX)
    new_act_1 = np.clip(np.matmul(new_act_0,   weights_1_s) // (ACCUM_DIV // 1),          0, CLAMP_MAX)
    new_act_2 = np.clip(np.matmul(new_act_1,   weights_2_s) // (ACCUM_DIV // 1),          0, CLAMP_MAX)
    new_act_3 = np.clip(np.matmul(new_act_2,   weights_3_s) // (ACCUM_DIV // 1), -CLAMP_MAX, CLAMP_MAX) * 4

    act_3_s *= 4
    # print (new_act_3)
    # print (act_3_s)
    # print (np.sum(np.abs(new_act_3 - act_3_s)))

    print(f'expected quantized activations before sigmoid min: {np.min(act_3_s/127.0)}, max: {np.max(act_3_s/127.0)}, mean: {np.mean(act_3_s/127.0)}, std: {np.std(act_3_s/127.0)}')
    print(f'         quantized activations before sigmoid min: {np.min(new_act_3/127.0)}, max: {np.max(new_act_3/127.0)}, mean: {np.mean(new_act_3/127.0)}, std: {np.std(new_act_3/127.0)}')
    outputs_hard = np.clip(0.5 + (new_act_3/CLAMP_MAX)*0.2, 0, 1)
    outputs_sigm = 1.0 / (1.0 + np.exp(-new_act_3/CLAMP_MAX))

    for n in range(64*64):
        outputs_sigm[n][3] = outputs_hard[n][3] = 1

    image = Image.fromarray((outputs_hard.reshape((64, 64, 4))*255).astype(np.uint8), 'RGBA')
    image.save("output_sigmoid_hard.png")

    image = Image.fromarray((outputs_sigm.reshape((64, 64, 4))*255).astype(np.uint8), 'RGBA')
    image.save("output_sigmoid.png")

    image = Image.fromarray((outputs_sigm.reshape((64, 64, 4))*255).astype(np.uint8) & 0xF0, 'RGBA')
    image.save("output_sigmoid_12bpp.png")

    image = Image.fromarray((act_4_s.reshape((64, 64, 4))*2).astype(np.uint8), 'RGBA')
    image.save("output_o.png")
