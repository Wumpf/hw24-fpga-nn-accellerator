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
    wire reset;
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
    blinky blinky
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


    reg [31:0] accumulator;
    reg [15:0] progress;
    mac_grid mac_grid
    (
        .clk(clk),
        .reset(reset),
        .out(accumulator),
        .progress(progress)
    );

    // // reg [15:0] activations [0:63];
    // wire [7:0] activations [0:127];
    // wire [7:0] accumulators [0:31];
    // reg [15:0] pc;
    // reg [7:0] command;
    // gemm_processor gemm_processor
    // (
    //     .clk(clk),
    //     .reset(reset),
    //     .activations_out(activations),
    //     .accumulator_out(accumulators),
    //     .progress(pc),
    //     .command(command)
    // );


    genvar ii;

    reg [128*8-1:0] activations;
    wire [7:0] activations_as_array [0:128-1];
    for (ii = 0; ii < 128; ii = ii + 1)
        assign activations_as_array[ii] = activations[ii*8 +: 8];

    wire [32*32-1:0] accumulators;
    wire [31:0] accumulators_as_array [0:32-1];
    for (ii = 0; ii < 32; ii = ii + 1)
        assign accumulators_as_array[ii] = accumulators[ii*32 +: 32];

    reg enable;
    reg restart_program;
    reg [15:0] pc;
    reg [7:0] command;
    gemm_processor2 gemm_processor2
    (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .restart_program(restart_program),
        .activations_in(activations),
        .activations_out(activations),
        .accumulators(accumulators),
        .progress(pc),
        .command(command)
    );



endmodule
