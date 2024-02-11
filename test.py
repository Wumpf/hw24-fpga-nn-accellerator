import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles


### TESTS

@cocotb.test()
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


@cocotb.test()
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

@cocotb.test()
async def gemm_processor_test(dut):
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
    print ([int(n) for n in dut.activations.value])

    for i in range (512):
        await ClockCycles(dut.clk, 1)
        print ([int(n) for n in dut.activations.value], int(dut.pc.value), dut.command.value)
