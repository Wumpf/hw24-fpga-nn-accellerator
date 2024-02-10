

module multiply_add (
    input wire[15:0] a,
    input wire[15:0] b,
    input wire[30:0] c,

    output wire[31:0] out
);

    assign out = a * b + c;

endmodule


// `define FPGA_TEST

module mac_not_yet_a_grid(
    input clk,
    input reset,
    output [31:0] out,
    output [15:0] progress
);

    reg [15:0] weights_memory [0:255];
    reg [15:0] activations_memory [0:255];
    initial begin
        $readmemh("weights.memh", weights_memory);
        $readmemh("activations.memh", activations_memory);
    end

    reg [63:0] time_counter = 0;
    reg [15:0] counter = 0;
    // wire [15:0] w;
    // wire [15:0] a;
    reg [31:0] accumulator = 0;
    always @(posedge clk) begin
        if (reset) begin
            counter <= 0;
            accumulator <= 0;
        end else if (counter < 256) begin
`ifdef FPGA_TEST
            if (time_counter == 12_000_000/16) begin
                time_counter <= 0;
                counter <= counter + 1;

                accumulator <= accumulator +
                            weights_memory[counter] *
                            activations_memory[counter];

                // accumulator <= {weights_memory[counter],
                //                 activations_memory[counter]};

                // accumulator <= {weights_memory[counter] *
                //                 activations_memory[counter]};
            end else
                time_counter <= time_counter + 1;
`else
                counter <= counter + 1;
                accumulator <= accumulator +
                            weights_memory[counter] *
                            activations_memory[counter];
`endif
        end
    end

    assign out = accumulator;
    assign progress = counter;

endmodule


// `define VGA
// `define DVI

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

`ifdef VGA
    wire [31:0] accumulator;
    mac_not_yet_a_grid mac_not_yet_a_grid(
        // .clk(CLK),
        .clk(clk_pixel),
        .reset(BTN1),
        .out(accumulator)
    );

    // assign pmod_1a = accumulator[7:0];
    //assign pmod_1b = accumulator[15:8];
    assign {LED1, LED2, LED3, LED4, LED5} = accumulator[20:16];
    // assign {LED1, LED2, LED3, LED4, LED5} = accumulator[4:0];


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

    wire [5:0] bit_index;
    wire output_bit;
    reg [4:0] output_color;
    always @(posedge clk_pixel) begin
        bit_index <= 5'd31-counter_h/16;
        output_bit <= accumulator[bit_index];
        output_color <= (counter_h < 32*16) ?
            {{3{output_bit}}, bit_index[0]}:
            4'b1000;

        // output_color <= bit_index[0] ? {12{output_bit}} : {{8{output_bit}}, 4'b0};
    end

    assign {vga_r, vga_g, vga_b,
            vga_hsync, vga_vsync} = {output_color * is_display_area,
                                     output_color * is_display_area,
                                     output_color * is_display_area,
                                     h_sync, v_sync};
`else
    wire [31:0] accumulator;
    mac_not_yet_a_grid mac_not_yet_a_grid(
        .clk(CLK),
        .reset(BTN1),
        .out(accumulator)
    );

    assign pmod_1a = accumulator[7:0];
    assign pmod_1b = accumulator[15:8];
    assign {LED1, LED2, LED3, LED4, LED5} = accumulator[20:16];

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
