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
    wire [11:0] rgb_output;
    reg [7:0] activation;
    reg [7:0] pixel_r;
    reg [7:0] pixel_g;
    reg [7:0] pixel_b;
    reg [7:0] activations      [0:64*64*3-1];

    reg [1:0] counter;

    integer cmd, ii;
    initial begin
        $readmemh("activation_3_12bits.txt", activations);
    end

    always @(posedge clk) begin
        activation = activations[counter_v[9:3] * 64 * 3 + counter_h[9:3] * 3 + counter];

        if (counter == 0)
            pixel_r <= activation;
        else if (counter == 1)
            pixel_g <= activation;
        else if (counter == 2)
            pixel_b <= activation;

        counter <= counter + 1;
    end

    localparam OFF=1;
    assign rgb_output = {pixel_r[3+OFF:0+OFF], pixel_g[3+OFF:0+OFF], pixel_b[3+OFF:0+OFF]};

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
