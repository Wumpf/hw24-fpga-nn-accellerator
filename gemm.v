
module multiply_add_accumulator_with_relu 
#(
    parameter PARAMS_WIDTH = 16
) (
    input clk,
    input reset,
    input clear_accumulator,
    input [PARAMS_WIDTH-1:0] a, // weight
    input [PARAMS_WIDTH-1:0] b, // activations
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

module gemm_processor
#(
    parameter MAC_COUNT = 32,
    parameter PARAMS_WIDTH = 8
) (
    input clk,
    input reset,
    input load,
    input [7:0] activation_addr,
    input [ACTIVATIONS_WIDTH-1:0] activation_in,
    output reg [ACTIVATIONS_WIDTH-1:0] activation_out,
    output wire [15:0] progress,
    output wire [7:0] command
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
    reg [COMMAND_WIDTH-1:0] command_stream [0:511];

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
        $readmemh("dense_2_weights.txt", weights, 2048*4, 2048*5-1);
        $readmemh("dense_2_biases.txt",  biases,    32*4,   32*5-1);
        $readmemh("dense_3_weights.txt", weights, 2048*5, 2048*5+64*4-1);
        $readmemh("dense_3_biases.txt",  biases,    32*5,   32*5+   4-1);

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
        end else if (load) begin
            activations[activation_addr] <= activation_in;
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
        end

        activation_out <= activations[activation_addr];
    end

    assign progress = pc;
    assign command = {clear_accumulators, activations_in_addr, bias_addr, activations_out_addr};
    // assign activations_out = activations;
    // assign accumulator_out = mac_out_relu;
endmodule





////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

`define VGA

module top (
    input CLK,
    input BTN1,

    output LED1,
    output LED2,
    output LED3,
    output LED4,
    output LED5,
    
`ifdef VGA
    output           vga_hsync,
    output           vga_vsync,
    output wire[3:0] vga_r,
    output wire[3:0] vga_g,
    output wire[3:0] vga_b
`elsif DVI
    output           dvi_clk,
    output           dvi_hsync,
    output           dvi_vsync,
    output           dvi_de,
    output wire[3:0] dvi_r,
    output wire[3:0] dvi_g,
    output wire[3:0] dvi_b
`else
    output wire[7:0] pmod_1a,
    output wire[7:0] pmod_1b
`endif
);
    wire reset = BTN1;
    reg [11:0] image_xy;
    wire [5:0] image_x = image_xy[5:0];
    wire [5:0] image_y = image_xy[11:6];
    wire [6*12-1:0] image_hash = {image_xy, image_xy, image_xy, image_xy, image_xy, image_xy};
    reg [15:0] counter;
    wire [15:0] progress;

    reg [6:0] pixel_addr;
    wire [7:0] pixel_read;
    reg [7:0] pixel_write;

    reg [12:0] rgb_output;

    gemm_processor processor(
        .clk(clk),
        .reset(reset),
        .load(counter < 64),
        .activation_in (pixel_write),
        .activation_addr(pixel_addr),
        .activation_out (pixel_read),
        .progress(progress),
    );

    always @(posedge clk) begin
        if (reset)
            counter <= 0;
        else begin
            counter <= counter + 1;
            if (counter < 64) begin
                // load inputs
                pixel_addr <= counter;
                pixel_write <= image_hash[counter] ? 8'h7f : 8'hff;

                // rgb_output <= {4'hf, 4'h0, 4'h0};
            end else if (counter < 128+128) begin // 384, 128+128+64) begin
                pixel_addr <= 0;
                // rgb_output <= pixel_read;
                rgb_output = 12'b0;
                // execute network
                // rgb_output <= {4'h0, 4'hf, 4'h0};
64) begin // 384, 128+128+128+64) begin
                // get results
                pixel_addr <= counter[6:0];
                rgb_output <= pixel_read;//{rgb_output[7:0], pixel_read[7:4]};

                // rgb_output <= {4'h0, 4'h0, 4'hf};
            end else begin
                counter <= 0;

                image_xy <= image_xy + 1'b1;
            end
        end
    end

`ifdef VGA
    wire clk = clk_pixel;

    reg clk_pixel;
    vga_pll pll(
        .clk_in(CLK),
        .clk_out(clk_pixel),
        .locked()
    );

    reg h_sync, v_sync, is_display_area;
    reg [9:0] counter_h;
    reg [9:0] counter_v;
    vga_sync_generator vga_sync(
        .clk(clk_pixel),
        .h_sync(h_sync),
        .v_sync(v_sync),
        .is_display_area(is_display_area),
        .counter_h(counter_h),
        .counter_v(counter_v)
    );

    assign {vga_r, vga_g, vga_b,
            vga_hsync, vga_vsync} = {rgb_output * is_display_area,
                                     rgb_output * is_display_area,
                                     rgb_output * is_display_area,
                                     h_sync, v_sync};
`else
    wire clk = CLK;
    assign pmod_1a = pixel_value;
    assign pmod_1b = pixel_value;
`endif
endmodule


module vga_pll(
    input  clk_in,
    output clk_out,
    output locked
);
    SB_PLL40_PAD #(
        .FEEDBACK_PATH("SIMPLE"),
        .DIVR(4'b0000),         // DIVR =  0
        .DIVF(7'b1000010),      // DIVF = 66
        .DIVQ(3'b101),          // DIVQ =  5
        .FILTER_RANGE(3'b001)   // FILTER_RANGE = 1
    ) pll (
        .LOCK(locked),
        .RESETB(1'b1),
        .BYPASS(1'b0),
        .PACKAGEPIN(clk_in),
        .PLLOUTCORE(clk_out)
    );
endmodule

module vga_sync_generator(
    input clk,
    output h_sync,
    output v_sync,
    output is_display_area,
    output reg[9:0] counter_h,
    output reg[9:0] counter_v
);
    always @(posedge clk) begin
        h_sync <= (counter_h >= 639+16 && counter_h < 639+16+96);     // invert: negative polarity
        v_sync <= (counter_v >= 479+10 && counter_v < 479+10+2);      // invert: negative polarity
        is_display_area <= (counter_h <= 639 && counter_v <= 479);
    end

    always @(posedge clk)
        if (counter_h == 799) begin
            counter_h <= 0;

            if (counter_v == 525)
                counter_v <= 0;
            else
                counter_v <= counter_v + 1;
        end
        else
            counter_h <= counter_h + 1;
endmodule


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

// from https://github.com/smunaut/ice40-playground/blob/master/cores/spi_flash/rtl/spi_flash_reader.v
// https://github.com/bastibl/ice40/blob/master/spi/spi_flash_reader.v

module spi_flash_reader (
    // SPI interface
    output wire spi_mosi,
    input  wire spi_miso,
    output wire spi_cs_n,
    output wire spi_clk,

    // Command interface
    input  wire [23:0] addr,
    input  wire [15:0] len,
    input  wire go,
    output wire rdy,

    // Data interface
    output wire [7:0] data,
    output wire valid,

    // Clock / Reset
    input  wire clk,
    input  wire rst
);

    // Signals
    // -------

    // FSM
    localparam
        ST_IDLE     = 0,
        ST_CMD      = 1,
        ST_DUMMY    = 2,
        ST_READ     = 3;

    reg [1:0] fsm_state;
    reg [1:0] fsm_state_next;

    // Counters
    reg [2:0] cnt_bit;
    reg cnt_bit_last;

    reg [ 1:0] cnt_cmd;
    wire cnt_cmd_last;

    reg [16:0] cnt_len;
    wire cnt_len_last;

    // Shift register
    reg [31:0] shift_reg;

    // Misc
    reg rdy_i;
    reg valid_i;

    // IOB control
    wire io_mosi;
    wire io_miso;
    wire io_cs_n;
    wire io_clk;


    // FSM
    // ---

    // State register
    always @(posedge clk or posedge rst)
        if (rst)
            fsm_state <= ST_IDLE;
        else
            fsm_state <= fsm_state_next;

    // Next-State logic
    always @(*)
    begin
        // Default is not to move
        fsm_state_next = fsm_state;

        // Transitions ?
        case (fsm_state)
            ST_IDLE:
                if (go)
                    fsm_state_next = ST_CMD;

            ST_CMD:
                if (cnt_cmd_last & cnt_bit_last)
                    fsm_state_next = ST_DUMMY;

            ST_DUMMY:
                if (cnt_bit_last)
                    fsm_state_next = ST_READ;

            ST_READ:
                if (cnt_len_last & cnt_bit_last)
                    fsm_state_next = ST_IDLE;
        endcase
    end


    // Shift Register
    // --------------

    always @(posedge clk or posedge rst)
        if (rst)
            shift_reg <= 32'hAB000000;
        else begin
            if (go)
                shift_reg <= { 8'h0B, addr };
            else
                shift_reg <= { shift_reg[30:0], io_miso };
        end


    // Counters
    // --------

    always @(posedge clk)
        if (go) begin
            cnt_bit <= 3'b000;
            cnt_bit_last <= 1'b0;
        end else if (fsm_state != ST_IDLE) begin
            cnt_bit <= cnt_bit + 1;
            cnt_bit_last <= (cnt_bit == 3'b110);
        end

    always @(posedge clk)
        if (go)
            cnt_cmd <= 2'b00;
        else if (fsm_state == ST_CMD)
            cnt_cmd <= cnt_cmd + cnt_bit_last;

    assign cnt_cmd_last = (cnt_cmd == 2'b11);

    always @(posedge clk)
        if (go)
            cnt_len <= { 1'b0, len } - 1;
        else if (fsm_state == ST_READ)
            cnt_len <= cnt_len - cnt_bit_last;

    assign cnt_len_last = cnt_len[16];


    // User IF
    // -------

    // Ready
    always @(posedge clk or posedge rst)
        if (rst)
            rdy_i <= 1'b0;
        else
            // This only raises rdy one cycle after we're back to IDLE to
            // leave time for the shift reg to push out the last read byte
            rdy_i <= (rdy_i | (fsm_state == ST_IDLE)) & ~go;

    assign rdy = rdy_i;

    // Data readout
    assign data = { shift_reg[6:0], io_miso };
    assign valid = valid_i;

    always @(posedge clk)
        valid_i <= (fsm_state == ST_READ) & cnt_bit_last;


    // IO control
    // ----------

    assign io_mosi = (fsm_state == ST_CMD) ? shift_reg[31] : 1'b0;
    assign io_cs_n = (fsm_state == ST_IDLE);
    assign io_clk  = (fsm_state != ST_IDLE);


    // IOBs
    // ----

    // MOSI output
        // Use DDR output to be half a cycle in advance
    SB_IO #(
        .PIN_TYPE(6'b010001),
        .PULLUP(1'b0),
        .NEG_TRIGGER(1'b0),
        .IO_STANDARD("SB_LVCMOS")
    ) iob_mosi_I (
        .PACKAGE_PIN(spi_mosi),
        .CLOCK_ENABLE(1'b1),
        .OUTPUT_CLK(clk),
        .D_OUT_0(io_mosi),
        .D_OUT_1(io_mosi)
    );

    // MISO capture
        // Because SPI_CLK is aligned with out clock we can
        // use a simple register here to sample on rising SPI_CLK
    SB_IO #(
        .PIN_TYPE(6'b000000),
        .PULLUP(1'b0),
        .NEG_TRIGGER(1'b0),
        .IO_STANDARD("SB_LVCMOS")
    ) iob_miso_I (
        .PACKAGE_PIN(spi_miso),
        .CLOCK_ENABLE(1'b1),
        .INPUT_CLK(clk),
        .D_IN_0(io_miso)
    );

    // Chip Select
        // Use DDR output to be half a cycle in advance
    SB_IO #(
        .PIN_TYPE(6'b010001),
        .PULLUP(1'b0),
        .NEG_TRIGGER(1'b0),
        .IO_STANDARD("SB_LVCMOS")
    ) iob_cs_n_I (
        .PACKAGE_PIN(spi_cs_n),
        .CLOCK_ENABLE(1'b1),
        .OUTPUT_CLK(clk),
        .D_OUT_0(io_cs_n),
        .D_OUT_1(io_cs_n)
    );

    // Clock
        // Use DDR output to have rising edge of SPI_CLK with
        // the rising edge of our internal clock
    SB_IO #(
        .PIN_TYPE(6'b010001),
        .PULLUP(1'b0),
        .NEG_TRIGGER(1'b0),
        .IO_STANDARD("SB_LVCMOS")
    ) iob_clk_I (
        .PACKAGE_PIN(spi_clk),
        .CLOCK_ENABLE(1'b1),
        .OUTPUT_CLK(clk),
        .D_OUT_0(io_clk),
        .D_OUT_1(1'b0)
    );

endmodule // spi_flash_reader
