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