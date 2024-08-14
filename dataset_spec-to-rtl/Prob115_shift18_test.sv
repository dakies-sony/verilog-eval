`timescale 1 ps/1 ps
`define OK 12
`define INCORRECT 13


module stimulus_gen (
	input clk,
	output reg load,
	output reg ena,
	output reg[1:0] amount,
	output reg[63:0] data,
	output reg[511:0] wavedrom_title,
	output reg wavedrom_enable
);


// Add two ports to module stimulus_gen:
//    output [511:0] wavedrom_title
//    output reg wavedrom_enable

	task wavedrom_start(input[511:0] title = "");
	endtask
	
	task wavedrom_stop;
		#1;
	endtask	



	initial begin
		load <= 1;
		ena <= 0;
		data <= 'x;
		amount <= 0;
		@(posedge clk) data <= 64'h000100;
		wavedrom_start("Shifting");
			@(posedge clk) load <= 0; ena <= 1;
							amount <= 2;
			@(posedge clk) amount <= 2;
			@(posedge clk) amount <= 2;
			@(posedge clk) amount <= 1;
			@(posedge clk) amount <= 1;
			@(posedge clk) amount <= 0;
			@(posedge clk) amount <= 0;
			@(posedge clk) amount <= 3;
			@(posedge clk) amount <= 3;
			@(posedge clk) amount <= 2;
			@(posedge clk) amount <= 2;
			@(negedge clk);
		wavedrom_stop();
		
		@(posedge clk); load <= 1; data <= 64'hx;
		@(posedge clk); load <= 1; data <= 64'h80000000_00000000;
		wavedrom_start("Arithmetic right shift");
			@(posedge clk) load <= 0; ena <= 1;
							amount <= 2;
			@(posedge clk) amount <= 2;
			@(posedge clk) amount <= 2;
			@(posedge clk) amount <= 2;
			@(posedge clk) amount <= 2;
			@(negedge clk);
		wavedrom_stop();

		@(posedge clk);
		@(posedge clk);
		
		
		
		repeat(4000) @(posedge clk, negedge clk) begin
			load <= !($random & 31);
			ena <= |($random & 15);
			amount <= $random;
			data <= {$random,$random};
		end
		#1 $finish;
	end
	
endmodule

module tb();

	typedef struct packed {
		int errors;
		int errortime;
		int errors_q;
		int errortime_q;

		int clocks;
	} stats;
	
	stats stats1;
	
	
	wire[511:0] wavedrom_title;
	wire wavedrom_enable;
	int wavedrom_hide_after_time;
	
	reg clk=0;
	initial forever
		#5 clk = ~clk;

	logic load;
	logic ena;
	logic [1:0] amount;
	logic [63:0] data;
	logic [63:0] q_ref;
	logic [63:0] q_dut;

	initial begin 
		$dumpfile("wave.vcd");
		$dumpvars(1, stim1.clk, tb_mismatch ,clk,load,ena,amount,data,q_ref,q_dut );
	end


	wire tb_match;		// Verification
	wire tb_mismatch = ~tb_match;
	
	stimulus_gen stim1 (
		.clk,
		.* ,
		.load,
		.ena,
		.amount,
		.data );
	RefModule good1 (
		.clk,
		.load,
		.ena,
		.amount,
		.data,
		.q(q_ref) );
		
	TopModule top_module1 (
		.clk,
		.load,
		.ena,
		.amount,
		.data,
		.q(q_dut) );

	
	bit strobe = 0;
	task wait_for_end_of_timestep;
		repeat(5) begin
			strobe <= !strobe;  // Try to delay until the very end of the time step.
			@(strobe);
		end
	endtask	

	
	final begin
		if (stats1.errors_q) $display("Hint: Output '%s' has %0d mismatches. First mismatch occurred at time %0d.", "q", stats1.errors_q, stats1.errortime_q);
		else $display("Hint: Output '%s' has no mismatches.", "q");

		$display("Hint: Total mismatched samples is %1d out of %1d samples\n", stats1.errors, stats1.clocks);
		$display("Simulation finished at %0d ps", $time);
		$display("Mismatches: %1d in %1d samples", stats1.errors, stats1.clocks);
	end
	
	// Verification: XORs on the right makes any X in good_vector match anything, but X in dut_vector will only match X.
	assign tb_match = ( { q_ref } === ( { q_ref } ^ { q_dut } ^ { q_ref } ) );
	// Use explicit sensitivity list here. @(*) causes NetProc::nex_input() to be called when trying to compute
	// the sensitivity list of the @(strobe) process, which isn't implemented.
	always @(posedge clk, negedge clk) begin

		stats1.clocks++;
		if (!tb_match) begin
			if (stats1.errors == 0) stats1.errortime = $time;
			stats1.errors++;
		end
		if (q_ref !== ( q_ref ^ q_dut ^ q_ref ))
		begin if (stats1.errors_q == 0) stats1.errortime_q = $time;
			stats1.errors_q = stats1.errors_q+1'b1; end

	end

   // add timeout after 100K cycles
   initial begin
     #1000000
     $display("TIMEOUT");
     $finish();
   end

endmodule

