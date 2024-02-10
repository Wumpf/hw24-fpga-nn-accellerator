
module multiply_add_accumulator (
    input clk,
    input reset,
    input enable,
    input wire[15:0] a,
    input wire[15:0] b,

    output reg[31:0] out
);

    reg[31:0] accumulator;
    always @(posedge clk) begin
        if (reset)
            accumulator <= 0;
        else if (enable)
            accumulator <= accumulator + a * b;
    end

    assign out = accumulator;

endmodule


// info how to setup SB_MAC16: https://trmm.net/Multiplier/
module multiply_add_accumulator_ (
    input clk,
    input reset,
    input enable,
    input wire[15:0] a,
    input wire[15:0] b,

    output reg[31:0] out
);
    wire [15:0] dsp_c = 16'b0;
    wire [15:0] dsp_d = 16'b0;

    reg dsp_irsttop;
    reg dsp_irstbot;
    reg dsp_orsttop;
    reg dsp_orstbot;
    reg dsp_ahold;
    reg dsp_bhold;
    reg dsp_chold;
    reg dsp_dhold;
    reg dsp_oholdtop;
    reg dsp_oholdbot;
    reg dsp_addsubtop;
    reg dsp_addsubbot;
    reg dsp_oloadtop;
    reg dsp_oloadbot;
    reg dsp_ci;

    always @(posedge clk) begin
        dsp_irsttop <= 0;
        dsp_irstbot <= 0;
        dsp_orsttop <= 0;
        dsp_orstbot <= 0;
        dsp_ahold <= 0;
        dsp_bhold <= 0;
        dsp_chold <= 0;
        dsp_dhold <= 0;
        dsp_oholdtop <= 0;
        dsp_oholdbot <= 0;
        dsp_addsubtop <= 0;
        dsp_addsubbot <= 0;
        dsp_oloadtop <= reset;
        dsp_oloadbot <= reset;
        dsp_ci <= 0;        
    end

    //setup the dsp, parameters TOPADDSUB_LOWERINPUT and BOTADDSUB_LOWERINPUT at 2 means we can use MAC operations
    SB_MAC16 #(
        .TOPOUTPUT_SELECT(2'b00), // adder, unregistered
        .TOPADDSUB_LOWERINPUT(2'b10), // multiplier hi bits
        .TOPADDSUB_UPPERINPUT(1'b0), // input C
        .TOPADDSUB_CARRYSELECT(2'b11), // top carry in is bottom carry out
        .BOTOUTPUT_SELECT(2'b00), // adder, unregistered
        .BOTADDSUB_LOWERINPUT(2'b10), // multiplier lo bits
        .BOTADDSUB_UPPERINPUT(1'b0), // input D
        .BOTADDSUB_CARRYSELECT(2'b00) // bottom carry in constant 0
    ) SB_MAC16_inst(
      .CLK(clk), .CE(enable), .C(dsp_c), .A(a), .B(b), .D(dsp_d),
      
      .IRSTTOP(dsp_irsttop), .IRSTBOT(dsp_irstbot), .ORSTTOP(dsp_orsttop), .ORSTBOT(dsp_orstbot),
      .AHOLD(dsp_ahold), .BHOLD(dsp_bhold), .CHOLD(dsp_chold), .DHOLD(dsp_dhold), .OHOLDTOP(dsp_oholdtop), .OHOLDBOT(dsp_oholdbot),
      .ADDSUBTOP(dsp_addsubtop), .ADDSUBBOT(dsp_addsubbot),
      .CI(dsp_ci), .CO(dsp_co),

      .OLOADTOP(dsp_oloadtop), .OLOADBOT(dsp_oloadbot),
      .O(out)
    );
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

    reg [15:0] mac_a;
    reg [15:0] mac_b;
    multiply_add_accumulator mac(
        .clk(clk),
        .reset(reset),
        .enable(1),
        .a(mac_a),
        .b(mac_b),
        .out(out)
    );

    reg [63:0] time_counter = 0;
    reg [15:0] counter = 0;
    always @(posedge clk) begin
        if (reset) begin
            counter <= 0;
            mac_a <= 0;
            mac_b <= 0;
        end else if (counter < 256) begin // && time_counter == 12_000_000/30) begin
            time_counter <= 0;
            counter <= counter + 1;

            mac_a <= weights_memory[counter];
            mac_b <= activations_memory[counter];
        end else begin
            time_counter <= time_counter + 1;
            mac_a <= 0;
            mac_b <= 0;
        end
    end

    // assign out = accumulator;
    assign progress = counter;

endmodule

module mac_grid(
    input clk,
    input reset,
    output reg [31:0] out,
    output [15:0] progress
);

    reg [15:0] weights_memory [0:255];
    reg [15:0] activations_memory [0:255];
    initial begin
        $readmemh("weights.memh", weights_memory);
        $readmemh("activations.memh", activations_memory);
    end

    localparam MAC_COUNT = 32;

    reg [15:0] mac_a[MAC_COUNT-1:0];
    reg [15:0] mac_b[MAC_COUNT-1:0];
    reg [31:0] mac_out[MAC_COUNT-1:0];

    genvar i;
    generate
        for (i = 0; i < MAC_COUNT; i = i + 1) begin : element
            multiply_add_accumulator mac(
                .clk(clk),
                .reset(reset),
                .enable(1'b1),
                .a(mac_a[i]),
                .b(mac_b[i]),
                .out(mac_out[i])
            );
        end
    endgenerate

    if (MAC_COUNT == 4)
        assign out = mac_out[0] + mac_out[1] + mac_out[2] + mac_out[3];
    else if (MAC_COUNT == 8)
        assign out = mac_out[0] + mac_out[1] + mac_out[2] + mac_out[3] + mac_out[4] + mac_out[5] + mac_out[6] + mac_out[7];
    else begin
        integer nn;
        always @ (*)
        begin
            out = 0;
            for(nn = 0; nn < MAC_COUNT; nn = nn + 1)
                out = out + mac_out[nn];
        end
    end

    reg [7:0] n;
    reg [63:0] time_counter = 0;
    reg [15:0] counter = 0;
    always @(posedge clk) begin
        if (reset) begin
            counter <= 0;
            for (n = 0; n < MAC_COUNT; n = n + 1) begin
                mac_a[n] <= 0;
                mac_b[n] <= 0;
            end
        end else if (counter < 256) begin //  && time_counter == 12_000_000/30) begin
            time_counter <= 0;
            counter <= counter + MAC_COUNT;

            for (n = 0; n < MAC_COUNT; n = n + 1) begin
                mac_a[n] <= weights_memory[counter + n];
                mac_b[n] <= activations_memory[counter + n];
            end
        end else begin
            for (n = 0; n < MAC_COUNT; n = n + 1) begin
                mac_a[n] <= 0;
                mac_b[n] <= 0;
            end        
        end
    end

    // assign out = accumulator;
    assign progress = counter;

endmodule


`define VGA
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
    // mac_not_yet_a_grid mac_not_yet_a_grid(
    mac_grid mac_grid(
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
