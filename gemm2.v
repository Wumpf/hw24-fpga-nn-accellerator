
module signed_adder_with_clamp_18bit (
    input signed [17:0] a,
    input signed [17:0] b,
    output reg signed [17:0] out
);

    wire signed [18:0] sum_temp;
    wire carry;

    assign {carry, sum_temp} = a + b; // 18-bit + 16-bit addition
    wire overflow = (a[17] == b[17]) & (a[17] != sum_temp[17]);

    // Clamp result if overflow
    always @(*) begin
        //  if (carry == 1 || (a[17] == b[17] && a[17] != sum_temp[17])) begin
        if (overflow) begin
            if (a[17] == 1) // If both inputs are negative, clamp to the minimum negative value
                out = 18'sh20000;// -32768*4;
            else // Otherwise, clamp to the maximum positive value
                out = 18'sh1ffff;//32768*4-1;
        end
        else
            out = sum_temp[17:0];
    end
endmodule


module multiply_add_accumulator_18bit
#(
    parameter PARAMS_WIDTH = 8
) (
    input clk,
    input reset,
    input enable,
    input clear_accumulator,
    input signed [7:0] a, // weight
    input signed [7:0] b, // activations
    output wire signed [17:0] out
);
    wire signed [15:0] a_mul_b = a * b; //$signed({1'b0, b[7:1]});// $signed(a * b);
    wire signed [17:0] a_mul_b_plus_accumulator_clamped;
    signed_adder_with_clamp_18bit signed_adder_with_clamp(
        .a(accumulator),
        .b(a_mul_b),
        .out(a_mul_b_plus_accumulator_clamped)
    );

    reg signed [17:0] accumulator;
    always @(posedge clk) begin
        if (reset)
            accumulator <= 0;
        else if (clear_accumulator)
            accumulator <= a_mul_b;
        else if (enable)
            accumulator <= a_mul_b_plus_accumulator_clamped;
    end

    assign out = accumulator;
endmodule

module div4_rrelu_18bit
(
    input signed [17:0] in,
    output wire signed [7:0] out
);
    // assign out = in < 0 ? 0 : in[16 -: 8];
    assign out = in < 0 ? 0 : in[17 -: 8];
endmodule


module multiply_add_accumulator_32bit
#(
    parameter PARAMS_WIDTH = 8
) (
    input clk,
    input reset,
    input enable,
    input clear_accumulator,
    input signed [7:0] a, // weight
    input signed [7:0] b, // activations
    output wire signed [31:0] out
);
    wire signed [15:0] a_mul_b = a * b;
    reg signed [31:0] accumulator;
    always @(posedge clk) begin
        if (reset)
            accumulator <= 0;
        else if (clear_accumulator)
            accumulator <= a_mul_b;
        else if (enable)
            accumulator <= accumulator + a_mul_b;
    end

    assign out = accumulator;
endmodule

module div4_relu_32bit
(
    input signed [31:0] in,
    output wire signed [7:0] out
);
    // assign out = in < 0 ? 0 : in[17 : 10];
    assign out = in < 0 ? 0 : in / (256*4);
endmodule

module relu_32bit
(
    input signed [31:0] in,
    output wire signed [7:0] out
);
    // assign out = in < 0 ? 0 : in[17 : 10];
    // assign out = in < 0 ? 0 : in / 256;
    // assign out = in < 0 ? 0 : {1'b0, in[14:8]};

    assign out = in < 0 ? 0 : ((in / 128) <= 127 ? (in / 128) : 127);

    // assign out = in < 0 ? 0 : ((in / 256) <= 127 ? (in / 256) : 127);
    // assign out = in < 0 ? 0 : ((in / 512) <= 127 ? (in / 512) : 127);

    // assign out = in < 0 ? 0 : ((in / 512) <= 63 ? (in / 512) : 63);
    // assign out = in < 0 ? 0 : ((in / 256) <= 63 ? (in / 256) : 63);
endmodule

    // assign accumulator_ = accumulator;
    // assign out = a_mul_b_plus_accumulator_clamped < 0 : 0 ? (a_mul_b_plus_accumulator_clamped >> 2);

// module multiply_add_accumulator_with_relu
// #(
//     parameter PARAMS_WIDTH = 8
// ) (
//     input clk,
//     input reset,
//     input clear_accumulator,
//     input [PARAMS_WIDTH-1:0] a, // weight
//     input [PARAMS_WIDTH-1:0] b, // activations
//     output wire [ACC_WIDTH-1:0] out,
//     output wire [PARAMS_WIDTH-1:0] out_relu
// );
//     localparam ACC_WIDTH = PARAMS_WIDTH * 2 + 2 + 1; // range -4.0 .. 4.0

//     reg signed [ACC_WIDTH-1:0] accumulator = 0;
//     always @(posedge clk) begin
//         if (reset)
//             accumulator <= 0;
//         else if (clear_accumulator)
//             accumulator <= $signed(a * b);
//         else
//             accumulator <= accumulator + $signed(a * b); // handle overflows
//     end

//     // assign out = accumulator;
//     // wire is_accumulator_positive = accumulator[ACC_WIDTH-1] == 0;
//     // wire does_accumulator_overflow_positive_range = |accumulator[ACC_WIDTH-2:ACC_WIDTH-3] == 1;
//     // wire [PARAMS_WIDTH-1:0] accumulator_div4 = accumulator[ACC_WIDTH-4 -: PARAMS_WIDTH];
//     // assign out_relu = is_accumulator_positive ?
//     //                     (does_accumulator_overflow_positive_range ?
//     //                         {PARAMS_WIDTH{1'b1}} :
//     //                         accumulator_div4)
//     //                     : 0;                                            // # if x < 0: return 0

//     // assign out = accumulator;
//     // wire is_accumulator_positive = accumulator[18] == 0;
//     // wire does_accumulator_overflow_positive_range = |accumulator[17:16] == 1;
//     // wire [7:0] accumulator_div4 = accumulator[15:PARAMS_WIDTH];
//     // assign out_relu = is_accumulator_positive ?
//     //                     (does_accumulator_overflow_positive_range ?
//     //                         {PARAMS_WIDTH{1'b1}} :
//     //                         accumulator_div4)
//     //                     : 0;                                            // # if x < 0: return 0



//     // assign out_relu = accumulator[ACC_WIDTH-1] == 1 ? 0 : 
//     //                   (|accumulator[ACC_WIDTH-2 : PARAMS_WIDTH-1] == 1) ? {1'b0,{(PARAMS_WIDTH-1){1'b1}}} : accumulator[PARAMS_WIDTH-1:0];

//     // assign out_relu = accumulator[PARAMS_WIDTH-1:0]; 

// endmodule

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

module gemm_processor2
#(
    parameter MAC_COUNT = 32,
    parameter PARAMS_WIDTH = 8
) (
    input clk,
    input reset,
    input enable,
    input restart_program,
    input [NEURONS*2*ACTIVATIONS_WIDTH-1:0] activations_in,
    output reg [NEURONS*2*ACTIVATIONS_WIDTH-1:0] activations_out,
    // for debugging purposes
    output reg [MAC_COUNT*ACCUMULATOR_WIDTH-1:0] accumulators,
    output reg [MAC_COUNT*ACTIVATIONS_WIDTH-1:0] accumulators_afrer_nonlinearity,
    output wire [15:0] progress,
    output wire [7:0] command
);
    localparam NEURONS = 64;
    localparam WEIGHTS = 4096*3;
    localparam BIASES = 256;

    localparam ACTIVATIONS_WIDTH = PARAMS_WIDTH;
    localparam ACCUMULATOR_WIDTH = 32; // 18, 16

    localparam ITERATIONS_PER_COMMAND = NEURONS;

    reg signed [PARAMS_WIDTH-1:0] weights               [0:WEIGHTS-1];
    reg signed [PARAMS_WIDTH-1:0] biases                [0:BIASES-1];
    // reg [ACTIVATIONS_WIDTH-1:0] activations      [0:NEURONS*2-1];
    reg signed [NEURONS*2*ACTIVATIONS_WIDTH-1:0] activations;

    localparam COMMAND_WIDTH = 1+2+3+2; // @TODO: find 1 bit for store_activations flag, maybe instead of bias_addr?
    reg [COMMAND_WIDTH-1:0] command_stream [0:511];

    integer cmd, ii, jj, nn;
    byte x, y;
    initial begin
        for (ii = 0; ii < WEIGHTS; ii = ii + 1)
            weights[ii] = 0;
        for (ii = 0; ii < BIASES; ii = ii + 1)
            biases[ii] = 0;

        $readmemh("dense_weights.txt",   weights, 4096*0, 4096*1-1);
        // $readmemh("dense_biases.txt",    biases,    64*0,   64*1-1);
        $readmemh("dense_1_weights.txt", weights, 4096*1, 4096*2-1);
        // $readmemh("dense_1_biases.txt",  biases,    64*1,   64*2-1);
        $readmemh("dense_2_weights.txt", weights, 4096*2, 2048*5-1);
        // $readmemh("dense_2_biases.txt",  biases,    64*2,   32*5-1);
        $readmemh("dense_3_weights.txt", weights, 2048*5, 2048*5+32*4-1);
        // $readmemh("dense_3_biases.txt",  biases,    32*5,   32*5+   4-1);

        // matrix transpose for testing
        // for (ii = 0; ii < 64; ii = ii + 1)
        //     for (jj = ii + 1; jj < 64; jj = jj + 1) begin
        //         x = weights[ii*64+jj];
        //         y = weights[jj*64+ii];
        //         weights    [ii*64+jj] = y;
        //         weights    [jj*64+ii] = x;
        //     end

        // display weight values for testing
        // $display(weights[0], " ", weights[1], " ", weights[2], " ", weights[3], " ... ", weights[63]);
        // $display(weights[64], " ", weights[65], " ", weights[66], " ", weights[67], " ... ", weights[127]);
        // $display(weights[128], " ", weights[129], " ", weights[130], " ", weights[131], " ... ", weights[191]);
        // $display(weights[192], " ", weights[193], " ", weights[194], " ", weights[195], " ... ", weights[255]);
        // $display("...");
        // $display(weights[4032-192], " ", weights[4033-192], " ", weights[4034-192], " ", weights[4035-192], " ... ", weights[4095-192]);
        // $display(weights[4032-128], " ", weights[4033-128], " ", weights[4034-128], " ", weights[4035-128], " ... ", weights[4095-128]);
        // $display(weights[4032-64], " ", weights[4033-64], " ", weights[4034-64], " ", weights[4035-64], " ... ", weights[4095-64]);
        // $display(weights[4032], " ", weights[4033], " ", weights[4034], " ", weights[4035], " ... ", weights[4095]);
        // $display(weights[4096]);

        // 01234567 =transpose=> 02461357                   0123456789abcdef =transpose=> 02468ace13579bdf
        // 0.12.34.56.7 -> 0.21.43.65.7                     0.12.34.56.78.9a.bc.de.f -> 0.21.43.65.87.a9.cb.ed.f
        // 02.14.36.57  -> 02.41.63.57                      02.14.36.58.7a.9c.be.df  -> 02.41.63.85.a7.c9.eb.df 
        // 024.16.357   -> 024.61.357                       024.16.38.5a.7c.9e.bdf   -> 024.61.83.a5.c7.e9.bdf
        // 02461357                                         0246.18.3a.5c.7e.9bdf    -> 0246.81.a3.c5.e7.9bdf
        //                                                  02468.1a.3c.5e.79bdf     -> 02468.a1.c3.e5.79bdf
        //                                                  02468a.1c.3e.579bdf      -> 02468a.c1.e3.579bdf
        //                                                  02468ac.1e.3579bdf       -> 02468ac.e1.3579bdf
        //                                                  02468ace13579bdf

        // transpose layer 0 (64,64) -> (128, 32)
        for (nn = 1; nn < 128/2; nn = nn + 1) begin
            for (ii = nn*32; ii < (128-nn)*32; ii = ii + 64) begin
                for (jj = ii; jj < ii + 32; jj = jj + 1) begin
                    x = weights[jj   ];
                    y = weights[jj+32];
                    weights    [jj   ] = y;
                    weights    [jj+32] = x;
                end
            end
        end

        // transpose layer 1 (64,64) -> (128, 32)
        for (nn = 1; nn < 128/2; nn = nn + 1) begin
            for (ii = nn*32; ii < (128-nn)*32; ii = ii + 64) begin
                for (jj = ii; jj < ii + 32; jj = jj + 1) begin
                    x = weights[4096*1 + jj   ];
                    y = weights[4096*1 + jj+32];
                    weights    [4096*1 + jj   ] = y;
                    weights    [4096*1 + jj+32] = x;
                end
            end
        end


        // pad layer 3 with zeroes (32,4) -> (32,32)
        for (ii = 31; ii >= 0; ii = ii - 1)
            for (jj = 0; jj < 32; jj = jj + 1) begin
                if (jj < 4)
                    weights[2048*5+ii*32+jj] = weights[2048*5+ii*4+jj];
                else
                    weights[2048*5+ii*32+jj] = 0;
            end

        // display weight values for testing
        //
        // $display(weights[0], " ", weights[1], " ", weights[2], " ", weights[3], " ... ", weights[31]);
        // $display(weights[32+0], " ", weights[32+1], " ", weights[32+2], " ", weights[32+3], " ... ", weights[32+31]);
        // $display(weights[64+0], " ", weights[64+1], " ", weights[64+2], " ", weights[64+3], " ... ", weights[64+31]);
        // $display(weights[96+0], " ", weights[96+1], " ", weights[96+2], " ", weights[96+3], " ... ", weights[96+31]);
        // $display("...");
        // $display(weights[4064-96], " ", weights[4065-96], " ", weights[4066-96], " ", weights[4067-96], " ... ", weights[4095-96]);
        // $display(weights[4064-64], " ", weights[4065-64], " ", weights[4066-64], " ", weights[4067-64], " ... ", weights[4095-64]);
        // $display(weights[4064-32], " ", weights[4065-32], " ", weights[4066-32], " ", weights[4067-32], " ... ", weights[4095-32]);
        // $display(weights[4064], " ", weights[4065], " ", weights[4066], " ", weights[4067], " ... ", weights[4095]);
        // $display(weights[4096]);
        // for (ii = 0; ii < 32; ii = ii + 1) begin
        //     nn = 2048*5+ii*32;
        //     $display(weights[nn+0], " ", weights[nn+1], " ", weights[nn+2], " ", weights[nn+3], " ... ", weights[nn+30], weights[nn+31]);
        // end


        // {clear_accumulators, activations_in_addr, bias_addr, activations_out_addr} 
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

    reg signed [PARAMS_WIDTH-1:0]  mac_a[MAC_COUNT-1:0];
    // reg [PARAMS_WIDTH-1:0]  mac_b[MAC_COUNT-1:0];
    reg signed [PARAMS_WIDTH-1:0] mac_b_shared;
    // wire [PARAMS_WIDTH-1:0] mac_out_relu[MAC_COUNT-1:0];
    // wire [ACCUMULATOR_WIDTH-1:0] mac_out[MAC_COUNT-1:0];
    wire signed [MAC_COUNT*ACTIVATIONS_WIDTH-1:0] mac_out_relu;
    // wire [MAC_COUNT*ACCUMULATOR_WIDTH-1:0] mac_out;
    wire signed [MAC_COUNT*ACCUMULATOR_WIDTH-1:0] mac_out;

    genvar i;
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : element
            // multiply_add_accumulator_with_relu #(.PARAMS_WIDTH(PARAMS_WIDTH)) mac
            //multiply_add_accumulator_18bit mac
            multiply_add_accumulator_32bit mac
            (
                .clk(clk),
                .reset(reset),
                .enable(enable),
                .clear_accumulator(clear_accumulators_in_sync_with_mac_input),
                .a(mac_a[i]),
                .b(mac_b_shared),
                .out(mac_out[i*ACCUMULATOR_WIDTH +: ACCUMULATOR_WIDTH])
            );

            // div4_relu_32bit activation
            relu_32bit activation
            (
                .in(mac_out[i*ACCUMULATOR_WIDTH +: ACCUMULATOR_WIDTH]),
                .out(mac_out_relu[i*8 +: 8])
            );
        end
    endgenerate

    wire clear_accumulators;
    reg clear_accumulators_in_sync_with_mac_input;
    reg [1:0] activations_out_addr_in_sync_with_mac_output [0:1]; // runs along with MAC pipeline
    wire [2:0] bias_addr;
    wire [1:0] activations_in_addr;
    wire [1:0] activations_out_addr;
    assign {clear_accumulators, activations_in_addr, bias_addr, activations_out_addr} = command_stream[pc];

    wire read_hazard =
            activations_in_addr == activations_out_addr_in_sync_with_mac_output[0] ||
            activations_in_addr == activations_out_addr_in_sync_with_mac_output[1];

    reg [7:0] n; // mac iterator
    reg [8:0] pc = 0;
    reg [15:0] weight_addr = 0;
    always @(posedge clk) begin
        if (reset) begin
            pc <= 0;
            weight_addr <= 0;
            activations_out <= 0;
            clear_accumulators_in_sync_with_mac_input <= 1;
            activations_out_addr_in_sync_with_mac_output[0] <= -1;
            activations_out_addr_in_sync_with_mac_output[1] <= -1;
            for (n = 0; n < MAC_COUNT; n = n + 1)
                mac_a[n] <= 0;
            mac_b_shared <= 0;
        end else if (restart_program) begin
            pc <= 0;
            weight_addr <= 0;
            clear_accumulators_in_sync_with_mac_input <= 1;
            activations_out_addr_in_sync_with_mac_output[0] <= -1;
            activations_out_addr_in_sync_with_mac_output[1] <= -1;
        end else if (enable) begin
            clear_accumulators_in_sync_with_mac_input <= clear_accumulators;
            activations_out_addr_in_sync_with_mac_output[1] <= activations_out_addr;
            activations_out_addr_in_sync_with_mac_output[0] <= activations_out_addr_in_sync_with_mac_output[1];
            mac_b_shared <= activations_in[(activations_in_addr*32 + pc[5:0])*ACTIVATIONS_WIDTH +: ACTIVATIONS_WIDTH];
            activations_out <= activations_in;
            for (n = 0; n < MAC_COUNT; n = n + 1) begin
                mac_a[n] <= weights    [weight_addr + n];
                activations_out[(activations_out_addr_in_sync_with_mac_output[0]*32 + n)*ACTIVATIONS_WIDTH +: ACTIVATIONS_WIDTH] <= 
                                             mac_out_relu[n * ACTIVATIONS_WIDTH +: ACTIVATIONS_WIDTH];
            end

            if (!read_hazard) begin
                // @TODO: weight_addr derived directly from pc
                if (weight_addr < WEIGHTS - 32)
                    weight_addr <= weight_addr + 32;
                pc <= pc + 1;
            end
        end
    end

    assign progress = pc;
    assign command = {clear_accumulators, activations_in_addr, bias_addr, activations_out_addr};
    assign accumulators = mac_out;
    assign accumulators_afrer_nonlinearity = mac_out_relu;
endmodule

