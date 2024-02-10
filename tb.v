`default_nettype none
`timescale 1ns/1ps

/*
this testbench just instantiates the module and makes some convenient wires
that can be driven / tested by the cocotb test.py
*/

// testbench is controlled by test.py
module tb ();

    // this part dumps the trace to a vcd file that can be viewed with GTKWave
    initial begin
        $dumpfile ("tb.vcd");
        $dumpvars (0, tb);
        #1;
    end

    // wire up the inputs and outputs
    wire clk;
    //wire rst_n;

    wire LED1;
    wire LED2;
    wire LED3;
    wire LED4;
    wire LED5;

    wire BTN_N;
    wire BTN1;
    wire BTN2;
    wire BTN3;

    // tt_hw23_nn_accelerator
    blinky top
    (
        .CLK(clk),

        .LED1(LED1),
        .LED2(LED2),
        .LED3(LED3),
        .LED4(LED4),
        .LED5(LED5),

        .BTN_N(BTN_N),
        .BTN1(BTN1),
        .BTN2(BTN2),
        .BTN3(BTN3)
    );

endmodule
