
module multiply_add_accumulator_with_relu 
#(
    parameter PARAMS_WIDTH = 16
) (
    input clk,
    input reset,
    input clear_accumulator,
    input [PARAMS_WIDTH-1:0] a,
    input [PARAMS_WIDTH-1:0] b,
    output wire [ACC_WIDTH-1:0] out,
    output wire [PARAMS_WIDTH-1:0] out_relu
);
    localparam ACC_WIDTH = PARAMS_WIDTH * 2;

    reg [ACC_WIDTH-1:0] accumulator = 0;
    always @(posedge clk) begin
        if (reset)
            accumulator <= 0;
        else if (clear_accumulator)
            accumulator <= a * b;
        else
            accumulator <= accumulator + a * b;
    end

    assign out = accumulator;
    assign out_relu = accumulator[ACC_WIDTH-1] == 1 ? 0 : 
                      (|accumulator[ACC_WIDTH-2 : PARAMS_WIDTH-1] == 1) ? {1'b0,{(PARAMS_WIDTH-1){1'b1}}} : accumulator[PARAMS_WIDTH-1:0];

    // assign out_relu = accumulator[PARAMS_WIDTH-1:0]; 

endmodule


// [2] -> [64] x [64x64] -> [64] x [64x64] ->... [64x4]
// [64] x [64x64] :: 4 times
// (row[64] x column[64] ) x64 --> [row[64] x sub_column[32] + row[64] x sub_column[32])


// 32 MAC
// [64] x [64x64] => [64]
// [64] x [64x64] => [64]
// [64] x [64x64] => [64]
// [64] x [64x4] =>   [4]


// [64] x [64x32] => [0..31]
// [64] x [64x32] => [31..63]
module gemm
#(
    parameter NEURONS_IN = 64,
    parameter NEURONS_OUT = 32,
    parameter PARAMS_WIDTH = 8,
    parameter ACTIVATIONS_WIDTH = PARAMS_WIDTH
) (
    input clk,
    input reset,
    input clear_accumulators,
    // input      [PARAMS_WIDTH-1:0] weights [0:WEIGHTS-1],
    // input      [PARAMS_WIDTH-1:0] biases [0:BIASES-1],
    // input      [ACTIVATIONS_WIDTH-1:0] activations_in [NEURONS_IN-1:0],
    // output reg [ACTIVATIONS_WIDTH-1:0] activations_out [NEURONS_OUT-1:0],
    input      [7:0] weights [0:WEIGHTS-1],
    input      [7:0] biases [0:BIASES-1],
    input      [7:0] activations_in [NEURONS_IN-1:0],
    output reg [7:0] activations_out [NEURONS_OUT-1:0],
    output [15:0] progress
);
    localparam MAC_COUNT = NEURONS_OUT;
    localparam WEIGHTS = NEURONS_IN * NEURONS_OUT;
    localparam BIASES = NEURONS_OUT;

    localparam ACCUMULATOR_WIDTH = PARAMS_WIDTH * 2;

    reg [PARAMS_WIDTH-1:0] mac_a[MAC_COUNT-1:0];
    reg [PARAMS_WIDTH-1:0] mac_b[MAC_COUNT-1:0];
    reg [ACCUMULATOR_WIDTH-1:0] mac_out[MAC_COUNT-1:0];

    genvar i;
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : element
            multiply_add_accumulator_with_relu #(.PARAMS_WIDTH(PARAMS_WIDTH)) mac
            (
                .clk(clk),
                .reset(reset),
                .clear_accumulator(clear_accumulators),
                .a(mac_a[i]),
                .b(mac_b[i]),
                .out(mac_out[i])
            );
        end
    endgenerate

    reg [7:0] n;
    reg [15:0] counter = 0;
    always @(posedge clk) begin
        if (reset) begin
            counter <= 0;
            for (n = 0; n < MAC_COUNT; n = n + 1) begin
                mac_a[n] <= 0;
                mac_b[n] <= 0;
            end
        end else if (counter < NEURONS_IN | clear_accumulators) begin
            if (clear_accumulators)
                counter <= 0;
            else
                counter <= counter + MAC_COUNT;

            for (n = 0; n < MAC_COUNT; n = n + 1) begin
                mac_a[n] <= weights        [counter + n]; // [N_IN, N_OUT]
                mac_b[n] <= activations_in [counter];
                
                activations_out[n] <= biases[n] + mac_out[n];
            end            
        end else begin
            for (n = 0; n < MAC_COUNT; n = n + 1) begin
                mac_a[n] <= 0;
                mac_b[n] <= 0;
            end
        end
    end

    assign progress = counter;
endmodule


module network
(
    input clk,
    input reset,
    // output [ACTIVATIONS_WIDTH-1:0] out [0:NEURONS_OUT-1],
    output [7:0] out [0:NEURONS_OUT-1],
    output [15:0] progress
);
    localparam MAC_COUNT = 32;
    localparam PARAMS_WIDTH = 8;
    localparam ACTIVATIONS_WIDTH = PARAMS_WIDTH;
    
    localparam NEURONS_IN = 64;
    localparam NEURONS_OUT = 64;
    localparam WEIGHTS = NEURONS_IN * NEURONS_OUT;
    localparam BIASES = NEURONS_OUT;

    // reg [PARAMS_WIDTH-1:0] weights_memory        [0:WEIGHTS-1];
    // reg [PARAMS_WIDTH-1:0] biases_memory         [0:BIASES-1];
    // reg [ACTIVATIONS_WIDTH-1:0] activations_in   [0:NEURONS_IN-1];
    // reg [ACTIVATIONS_WIDTH-1:0] activations_out  [0:NEURONS_OUT-1];
    reg [7:0] weights_memory        [0:WEIGHTS-1];
    reg [7:0] biases_memory         [0:BIASES-1];
    reg [7:0] activations_in        [0:NEURONS_IN-1];
    reg [7:0] activations_out       [0:NEURONS_OUT-1];

    initial begin
        $readmemh("dense_weights.txt", weights_memory);
        $readmemh("dense_biases.txt",  biases_memory);
        $readmemh("activations.memh", activations_in);
    end

    wire block = counter >= NEURONS_IN; // @TODO: NEURONS_OUT/MAC_COUNT
    reg [7:0] counter;
    gemm #(.NEURONS_IN(NEURONS_IN), .NEURONS_OUT(MAC_COUNT), .PARAMS_WIDTH(PARAMS_WIDTH), .ACTIVATIONS_WIDTH(ACTIVATIONS_WIDTH)) gemm
    (
        .clk(clk),
        .reset(reset),
        .clear_accumulators(counter == 0 || counter == NEURONS_IN),
        .weights(weights_memory), //block ? weights_memory[WEIGHTS/2:WEIGHTS-1] : weights_memory[0:WEIGHTS/2-1]),
        .biases(biases_memory),   // .biases(block ? biases_memory[BIASES/2:BIASES-1] : biases_memory[0:BIASES/2-1]),
        .activations_in(activations_in),
        .activations_out(activations_out), // .activations_out(block ? activations_out[NEURONS_OUT/2:NEURONS_OUT-1] : activations_out[0:NEURONS_OUT/2-1]),
        .progress(progress)
    );

    always @(posedge clk) begin
        if (reset)
            counter <= 0;
        else
            counter <= counter + 1;
    end

    assign out = activations_out;

endmodule


module gemm_processor
#(
    parameter MAC_COUNT = 32,
    parameter PARAMS_WIDTH = 8
) (
    input clk,
    input reset,
    output [7:0] out [0:NEURONS*2-1],
    output reg [7:0] out2,
    output [15:0] progress,
    output [7:0] command
);
    localparam NEURONS = 64;
    localparam WEIGHTS = 4096*3;
    localparam BIASES = 256;

    localparam ACTIVATIONS_WIDTH = PARAMS_WIDTH;
    localparam ACCUMULATOR_WIDTH = PARAMS_WIDTH * 2;

    localparam ITERATIONS_PER_COMMAND = NEURONS;

    reg [PARAMS_WIDTH-1:0] weights               [0:WEIGHTS-1];
    reg [PARAMS_WIDTH-1:0] biases                [0:BIASES-1];
    reg [ACTIVATIONS_WIDTH-1:0] activations      [0:NEURONS*2-1];

    localparam COMMAND_WIDTH = 1+2+3+2;
    reg [COMMAND_WIDTH] command_stream [0:511];

    integer cmd, ii;
    initial begin
        for (ii = 0; ii < WEIGHTS; ii = ii + 1)
            weights[ii] = 0;
        for (ii = 0; ii < BIASES; ii = ii + 1)
            biases[ii] = 0;

        $readmemh("dense_weights.txt",   weights, 4096*0, 4096*1-1);
        $readmemh("dense_biases.txt",    biases,    64*0,   64*1-1);
        $readmemh("dense_1_weights.txt", weights, 4096*1, 4096*2-1);
        $readmemh("dense_1_biases.txt",  biases,    64*1,   64*2-1);
        $readmemh("dense_2_weights.txt", weights, 4096*2, 4096*3-1);
        $readmemh("dense_2_biases.txt",  biases,    64*2,   64*3-1);
        // $readmemh("dense_3_weights.txt", weights, 4096*3); // @TODO:
        // $readmemh("dense_3_biases.txt",  biases, 64*3);
        // $readmemh("activations.memh", activations);

        // $readmemh("dense_biases.txt",    activations,    64*0,   64*1-1);    
        // $readmemh("dense_biases.txt",    activations,    64*1,   64*2-1);    
        for (ii = 0; ii < NEURONS*2; ii = ii + 1)
            activations[ii] = 1;//ii[7:0];

        // {clear_accumulators, enable_macs, advance_weight_addr, activations_addr, bias_addr}
        // 0th layer
        // act[0..64] x w[64x32] top    +  b[0..32]  => act[64..96]
        for (cmd = 64*0; cmd < 64*1; cmd = cmd + 1)
            command_stream[cmd] = {(cmd == 64*0 ? 1'b1 : 1'b0), 2'd0, 3'd0, 2'd2};
        // act[0..64] x w[64x32] bottom +  b[32..64] => act[96..128]
        for (cmd = 64*1; cmd < 64*2; cmd = cmd + 1)
            command_stream[cmd] = {(cmd == 64*1 ? 1'b1 : 1'b0), 2'd0, 3'd1, 2'd3};

        // 1th layer
        // act[64..128] x w[64x32] top    +  b[64..96]  => act[0..32]
        for (cmd = 64*2; cmd < 64*3; cmd = cmd + 1)
            command_stream[cmd] = {(cmd == 64*2 ? 1'b1 : 1'b0), 2'd2, 3'd2, 2'd0};
        // act[64..128] x w[64x32] bottom +  b[96..128] => act[32..64]
        for (cmd = 64*3; cmd < 64*4; cmd = cmd + 1)
            command_stream[cmd] = {(cmd == 64*3 ? 1'b1 : 1'b0), 2'd2, 3'd3, 2'd1};

        // 2nd layer
        // act[0..64] x w[64x32]          +  b[128..160] => act[64..96]
        for (cmd = 64*4; cmd < 64*5; cmd = cmd + 1)
            command_stream[cmd] = {(cmd == 64*4 ? 1'b1 : 1'b0), 2'd0, 3'd4, 2'd2};

        // 3rd layer
        // act[64..96] x w[32x4]          +  b[160..164] => act[0..4]
        for (cmd = 64*5; cmd < 64*5 + 32; cmd = cmd + 1)
            command_stream[cmd] = {(cmd == 64*5 ? 1'b1 : 1'b0), 2'd2, 3'd5, 2'd0};

        // 64*6 = 384 cmd, dummy write to                => act[96..128]
        for (cmd = 64*5 + 32; cmd < 512; cmd = cmd + 1)
            command_stream[cmd] = {1'b1, 2'd0, 3'd7, 2'd3};
    end

    reg [PARAMS_WIDTH-1:0]  mac_a[MAC_COUNT-1:0];
    reg [PARAMS_WIDTH-1:0]  mac_b[MAC_COUNT-1:0];
    wire [PARAMS_WIDTH-1:0] mac_out_relu[MAC_COUNT-1:0];
    wire [ACCUMULATOR_WIDTH-1:0] mac_out[MAC_COUNT-1:0];

    genvar i;
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : element
            multiply_add_accumulator_with_relu #(.PARAMS_WIDTH(PARAMS_WIDTH)) mac
            (
                .clk(clk),
                .reset(reset),
                .clear_accumulator(clear_accumulators),
                .a(mac_a[i]),
                .b(mac_b[i]),
                .out(mac_out[i]),
                .out_relu(mac_out_relu[i])
            );
        end
    endgenerate


    reg clear_accumulators = 0;
    reg [2:0] bias_addr = 0;
    reg [1:0] activations_in_addr = 0;
    reg [1:0] activations_out_addr = 0;

    reg [7:0] n; // mac iterator
    reg [8:0] pc = 0;
    reg [15:0] weight_addr = 0;
    always @(posedge clk) begin
        if (reset) begin
            pc <= 0;
            weight_addr <= 0;
            clear_accumulators <= 1;
        end else begin
            {clear_accumulators, activations_in_addr, bias_addr, activations_out_addr} <= command_stream[pc];

            for (n = 0; n < MAC_COUNT; n = n + 1) begin
                mac_a[n] <= weights    [weight_addr + n];
                mac_b[n] <= activations[activations_in_addr*32 + pc[5:0]];
                
                activations[activations_out_addr*32 + n] <= biases[bias_addr*32 + n] + mac_out_relu[n];
            end

            if (weight_addr < WEIGHTS - 32)
                weight_addr <= weight_addr + 32;
            pc <= pc + 1;

            // out2 <= activations[0];
        end
    end

    assign progress = pc;
    assign command = {clear_accumulators, activations_in_addr, bias_addr, activations_out_addr};
    assign out = activations;
    assign out2 = activations[0];

    // reg [8:0] counter = 0;
    // reg [8:0] pc = 0;
    // always @(posedge clk) begin
    //     if (reset) begin
    //         counter <= ITERATIONS_PER_COMMAND;
    //     end else begin
    //         if (counter == ITERATIONS_PER_COMMAND) begin
    //             {clear_accumulators, clear_arguments, weight_addr, bias_addr} <= command_stream[pc];
    //             pc <= pc + 1;
    //             counter <= 0;
    //         end else begin
    //             counter <= counter + 1;
    //             clear_accumulators <= 0;
    //             clear_arguments <= 0;
    //         end

    //         if (clear_arguments) begin
    //             for (n = 0; n < MAC_COUNT; n = n + 1) begin
    //                 mac_a[n] <= 0;
    //                 mac_b[n] <= 0;
    //             end
    //         end else begin
    //             for (n = 0; n < MAC_COUNT; n = n + 1) begin
    //                 mac_a[n] <= weights        [weight_addr + counter + ITERATIONS_PER_COMMAND*n];
    //                 mac_b[n] <= activations_in [counter];
                    
    //                 activations_out[n] <= biases[bias_addr + n] + mac_out[n];
    //             end
    //         end

    //     end







    // wire clear_accumulators = (|counter[$clog2(MAC_COUNT)-1:0] == 0);
    // wire clear_arguments = (reset | );



    // reg [7:0] n;
    // reg [15:0] counter = 0;
    // wire [11:0] command = 


    // always @(posedge clk) begin
    //     if (reset) begin
    //         counter <= 0;
    //         for (n = 0; n < MAC_COUNT; n = n + 1) begin
    //             mac_a[n] <= 0;
    //             mac_b[n] <= 0;
    //         end
    //     end else if (counter < NEURONS_IN | clear_accumulators) begin
    //         if (clear_accumulators)
    //             counter <= 0;
    //         else
    //             counter <= counter + MAC_COUNT;

    //         for (n = 0; n < MAC_COUNT; n = n + 1) begin
    //             mac_a[n] <= weights        [counter + n]; // [N_IN, N_OUT]
    //             mac_b[n] <= activations_in [counter];
                
    //             activations_out[n] <= biases[n] + mac_out[n];
    //         end            
    //     end else begin
    //         for (n = 0; n < MAC_COUNT; n = n + 1) begin
    //             mac_a[n] <= 0;
    //             mac_b[n] <= 0;
    //         end
    //     end
    // end

    // assign progress = counter;
endmodule


// module gemm_grid
//     #(  parameter BLOCK_W = 4,
//         parameter BLOCK_H = 8,
//         parameter GRID_W = 64,
//         parameter GRID_H = 64
// (
//     input clk,
//     input reset,
//     output reg [31:0] out [BLOCK_W-1:0],
//     output [15:0] progress
// );

//     // localparam NEURON_COUNT = BLOCK_W;
//     localparam STEP = BLOCK_H;
//     localparam WEIGHTS = GRID_W * GRID_H;
//     localparam BIASES = GRID_W;

//     reg [15:0] weights_memory  [0:WEIGHTS-1];
//     reg [15:0] biases_memory   [0:BIASES-1];
//     reg [15:0] activations_in  [0:GRID_H-1];
//     reg [15:0] activations_out [0:GRID_W-1];

//     initial begin
//         $readmemh("dense_weights.txt", weights_memory);
//         $readmemh("dense_biases.txt",  biases_memory);
//         $readmemh("activations.memh", activations_in);
//     end

//     localparam MAC_COUNT = BLOCK_W * BLOCK_H;
