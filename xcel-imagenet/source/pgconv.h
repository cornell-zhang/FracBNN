#ifndef PGCONV_H
#define PGCONV_H

#include "bnn.h"
#include "typedefs.h"
#include "dimension_def.h"
#include <iostream>

using namespace std;
const static uint64 m1 = 6148914691236517205;
const static uint64 m2 = 3689348814741910323;
const static uint64 m4 = 1085102592571150095;
const static uint4 lut16[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
const static int4 lut16bip[16] = {-4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, -2, 0, 2, 2, 4};

inline uint8 compute_engine_64(uint64 b, uint64 w)
{
#pragma HLS latency max=1
	uint64 x = ~(b^w);

	x -= (x >> 1) & m1;
	x = (x & m2) + ((x >> 2) & m2);
	x = (x + (x >> 4)) & m4;
	x += x >>  8;
	x += x >> 16;
	x += x >> 32;
	return (x & 0x7f);
}

inline uint6 compute_engine_32_0(uint32 b, uint32 w)
{
#pragma HLS latency max=1
	uint32 x = ~(b^w);
	x -= (x >> 1) & m1;
	x = (x & m2) + ((x >> 2) & m2);
	x = (x + (x >> 4)) & m4;
	x += x >>  8;
	x += x >> 16;
	return x;
}

inline uint6 compute_engine_32_1(uint32 b, uint32 w)
{
	uint32 x = ~(b^w);
	uint6 s = 0;
	for (int i = 0; i < 32; i ++){
		s += x[i];
	}
	return s;
}

inline uint6 compute_engine_32_2(uint32 b, uint32 w)
{
	uint32 x = ~(b^w);
	uint6 s = 0;
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

inline uint6 compute_engine_32_3(uint32 b, uint32 w)
{
	//#pragma HLS latency max=1
	uint32 x = ~(b^w);
	uint6 s = 0;
	for (int i = 0; i < 8; i ++){
		uint4 x0 = 0;
		x0.range(3, 0) = x.range(i*4 + 3, i*4);
		s += lut16[x0];
	}
	return s;
}

/*
 * Attention: 
 * when lsb_outputs is not used, for example, in the first binary conv layer,
 * its values are still modified.
 * This makes the next accumulation in the lsb_outputs buffer incorrect because
 * the registers directly copy the values from the buffer.
 */

void pg_conv3x3_tile( 
		uint32 msb_inputs[BUF_HEIGHT_1*BUF_WIDTH_1],
		uint32 weights[OUT_CHANNEL_PARALLELISM][3][3],
		int10 msb_outputs[OUT_CHANNEL_PARALLELISM][(ROW_TILE_SIZE+2)*(BUF_WIDTH_1+1)],

		int c_in,
		int H_fmap_out,
		int row_offset,
		int out_buf_start
)
{
#pragma HLS DEPENDENCE variable=msb_outputs array inter false
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=msb_outputs complete dim=1

	uint32 msb_line_buffer_0[BUF_WIDTH_1];
	uint32 msb_line_buffer_1[BUF_WIDTH_1];
	uint32 msb_window_buffer[3][3];

#pragma HLS ARRAY_PARTITION variable=msb_window_buffer complete dim=0

	int10 msb_partial_out_feature[OUT_CHANNEL_PARALLELISM];
#pragma HLS ARRAY_PARTITION variable=msb_partial_out_feature complete dim=1

	Loop_Tile:
	for (int row=0; row<ROW_TILE_SIZE+2; row++) {
		for (int col=0; col<H_fmap_out+1; col++) {
#pragma HLS PIPELINE 

			// update window buffer and line buffer
			for (int i=0; i<3; i++) {
				msb_window_buffer[i][0] = msb_window_buffer[i][1];
				msb_window_buffer[i][1] = msb_window_buffer[i][2];
			}

			int read_index = c_in*(H_fmap_out+1)*(H_fmap_out+1) + (row+row_offset)*(H_fmap_out+1) + col;
			if (H_fmap_out == 224) read_index = (row+row_offset)*(H_fmap_out+1) + col;
			msb_window_buffer[0][2] = (msb_line_buffer_0[col]);
			msb_window_buffer[1][2] = (msb_line_buffer_0[col] = msb_line_buffer_1[col]);
			msb_window_buffer[2][2] = (msb_line_buffer_1[col] = msb_inputs[read_index]);

			int msb_output_index = row*(H_fmap_out+1) + col + out_buf_start;
			// copy output features into registers
			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				if (c_in > 0){
					msb_partial_out_feature[channel_pt] = msb_outputs[channel_pt][msb_output_index];
				}
				else{
					msb_partial_out_feature[channel_pt] = 0;
				}
			}

			// Compute each feature in an output channel
			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				int10 msb_accumulation = 0;
				// Compute each output channel
				for (int k_row=0; k_row<3; k_row++) {
					for (int k_col=0; k_col<3; k_col++) {
						int row_idx_pad = row + row_offset - 2 + k_row;
						int col_idx_pad = col - 2 + k_col;
						if(row_idx_pad>=1 && row_idx_pad<H_fmap_out+1 && col_idx_pad>=0 && col_idx_pad<H_fmap_out){
							uint32 msb_a = msb_window_buffer[k_row][k_col];
							uint32 w = weights[channel_pt][k_row][k_col];
							uint6 popcnt = compute_engine_32_1(msb_a, w);
							msb_accumulation += 2*popcnt - 32;
						}
					}
				}
				msb_partial_out_feature[channel_pt] += msb_accumulation;
			}

			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				msb_outputs[channel_pt][msb_output_index] = msb_partial_out_feature[channel_pt];
			}

		}
	}
	return;
}

void pg_conv1x1_tile(
		uint32 inputs[BUF_HEIGHT_1*BUF_WIDTH_1],
		uint32 weights[OUT_CHANNEL_PARALLELISM],
		int10 outputs[OUT_CHANNEL_PARALLELISM][(ROW_TILE_SIZE+2)*(BUF_WIDTH_1+1)],

		int c_in,
		int H_fmap_out,
		int row_offset,
		int out_buf_start
)
{
#pragma HLS DEPENDENCE variable=outputs array inter false
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1

	int10 partial_out_feature[OUT_CHANNEL_PARALLELISM];
#pragma HLS ARRAY_PARTITION variable=partial_out_feature complete dim=1

	Loop_Tile:
	for (int row=0; row<ROW_TILE_SIZE; row++) {
		for (int col=0; col<H_fmap_out; col++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_MERGE

			int output_index = (row+2)*(H_fmap_out+1) + (col+1) + out_buf_start;
			// copy output features into registers
			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				if (c_in > 0){
					partial_out_feature[channel_pt] = outputs[channel_pt][output_index];
				}
				else{
					partial_out_feature[channel_pt] = 0;
				}
			}

			int read_index = c_in*(H_fmap_out)*(H_fmap_out) + (row + row_offset)*(H_fmap_out) + col + FEAT_BUF_OFFSET;
			uint32 act = inputs[read_index];
			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				// Compute each feature in an output channel
				uint32 w = weights[channel_pt];
				int6 accumulation = 2*compute_engine_32_1(act, w) - 32;
				partial_out_feature[channel_pt] += accumulation;
			}
			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				outputs[channel_pt][output_index] = partial_out_feature[channel_pt];
			}

		}
	}
	return;
}

#endif

