#ifndef LAYER_H
#define LAYER_H

#include "typedefs.h"
#include "dimension_def.h"
#include "bnn.h"
#include <iostream>

using namespace std;

static uint128 ddr_tmp[ROW_TILE_SIZE+1][BUF_WIDTH_0];
static uint512 ddr_stage[(ROW_TILE_SIZE+1)*BUF_WIDTH_0/4];

inline uint2 to2bit(FIX_FM_acc x)
{
#pragma HLS INLINE
	const CONST_4 scale = 1.5;
	ap_ufixed<2, 2, AP_RND_MIN_INF, AP_SAT> temp = (ap_ufixed<2, 2, AP_RND_MIN_INF, AP_SAT>)((x+1)*scale);
	return (uint2)temp;
}

inline uint4 to4bit(FIX_FM_acc x)
{
#pragma HLS INLINE
	const CONST_4 scale = 7.5;
	ap_ufixed<4, 4, AP_RND_MIN_INF, AP_SAT> temp = (ap_ufixed<4, 4, AP_RND_MIN_INF, AP_SAT>)((x+1)*scale);
	return (uint4)temp;
}

inline int ceil_div_4(int x)
{
#pragma HLS INLINE
	uint16 ret = (uint16)x;
	uint16 ret_int = ret >> 2;
	uint16 ret_mask = ret & 0x0003;
	if (ret_mask == 0){
		return ret_int;
	} else {
		return ret_int + 1;
	}
}

void load_1D_weights(
		uint512 weights_all[WEIGHTS_ALL_DIM],
		int weights_all_ptr,
		FIX_WT weight_buffer[OUT_CHANNEL_PARALLELISM]
)
{
	//#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=2

	uint512 data_pack[NUM_BUS_READS_OTHER];

	for (int i = 0; i < NUM_BUS_READS_OTHER; i ++) {
#pragma HLS PIPELINE
		data_pack[i] = weights_all[weights_all_ptr+i];
	}
	for (int i = 0; i < NUM_BUS_READS_OTHER; i ++) {
		for (int j = 0; j < NUM_WT_PACKS_OTHER; j ++) {
#pragma HLS UNROLL
					FIX_1D_PACK tmp = 0;
					tmp.range(WEIGHT_BITS_OTHER-1, 0) = data_pack[i].range(WEIGHT_BITS_OTHER-1+j*WEIGHT_BITS_OTHER, j*WEIGHT_BITS_OTHER);
					weight_buffer[i*NUM_WT_PACKS_OTHER+j] = (FIX_WT)tmp;
		}
	}
}

void load_layer_1D_weights_conv(
		FIX_WT conv_0_threshold[OUT_CHANNEL_PARALLELISM],
		FIX_WT conv_1_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT conv_1_bias[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_shift_x[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_shift_y[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT bn_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT bn_bias[OUT_CHANNEL_PARALLELISM],

		uint512 weights_all[WEIGHTS_ALL_DIM],
		int weights_all_ptr_start,
		int c_out,
		int conv_out_channels,
		int pw_out_channels
)
{
#pragma HLS INLINE
	int conv_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;
	int pw_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	int weights_all_ptr = weights_all_ptr_start + c_out*NUM_BUS_READS_OTHER;

	load_1D_weights(weights_all, weights_all_ptr, conv_0_threshold);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, conv_1_weight);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, conv_1_bias);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;

	//	load_1D_weights(weights_all, weights_all_ptr, conv_0_threshold);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	//	load_1D_weights(weights_all, weights_all_ptr, conv_1_weight);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	//	load_1D_weights(weights_all, weights_all_ptr, conv_1_bias);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;

	load_1D_weights(weights_all, weights_all_ptr, relu_shift_x);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, relu_shift_y);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, relu_weight);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;

	//	load_1D_weights(weights_all, weights_all_ptr, relu_shift_x);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	//	load_1D_weights(weights_all, weights_all_ptr, relu_shift_y);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	//	load_1D_weights(weights_all, weights_all_ptr, relu_weight);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;

	load_1D_weights(weights_all, weights_all_ptr, bn_weight);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, bn_bias);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;

	//	load_1D_weights(weights_all, weights_all_ptr, bn_weight);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	//	load_1D_weights(weights_all, weights_all_ptr, bn_bias);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
}

void load_layer_1D_weights_pw(
		FIX_WT conv_0_threshold[OUT_CHANNEL_PARALLELISM],
		FIX_WT conv_1_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT conv_1_bias[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_shift_x[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_shift_y[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT bn_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT bn_bias[OUT_CHANNEL_PARALLELISM],

		uint512 weights_all[WEIGHTS_ALL_DIM],
		int weights_all_ptr_start,
		int c_out,
		int conv_out_channels,
		int pw_out_channels
)
{
#pragma HLS INLINE
	int conv_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;
	int pw_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	int weights_all_ptr = weights_all_ptr_start + c_out*NUM_BUS_READS_OTHER;

	//		load_1D_weights(weights_all, weights_all_ptr, conv_0_threshold);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	//		load_1D_weights(weights_all, weights_all_ptr, conv_1_weight);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	//		load_1D_weights(weights_all, weights_all_ptr, conv_1_bias);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;

	load_1D_weights(weights_all, weights_all_ptr, conv_0_threshold);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, conv_1_weight);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, conv_1_bias);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;

	//		load_1D_weights(weights_all, weights_all_ptr, relu_shift_x);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	//		load_1D_weights(weights_all, weights_all_ptr, relu_shift_y);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	//		load_1D_weights(weights_all, weights_all_ptr, relu_weight);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;

	load_1D_weights(weights_all, weights_all_ptr, relu_shift_x);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, relu_shift_y);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, relu_weight);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;

	//		load_1D_weights(weights_all, weights_all_ptr, bn_weight);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;
	//		load_1D_weights(weights_all, weights_all_ptr, bn_bias);
	weights_all_ptr += conv_channels_after_tile*NUM_BUS_READS_OTHER;

	load_1D_weights(weights_all, weights_all_ptr, bn_weight);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
	load_1D_weights(weights_all, weights_all_ptr, bn_bias);
	weights_all_ptr += pw_channels_after_tile*NUM_BUS_READS_OTHER;
}

void load_conv3x3_weights(
		uint32 weight3x3_tile_buffer[OUT_CHANNEL_PARALLELISM][3][3],
		uint512 conv_weight_3x3_all[49104],
		int conv3x3_weight_ptr,
		int c_out,
		int c_in,
		int in_channels_after_pack
)
{

#pragma HLS ARRAY_PARTITION variable=weight3x3_tile_buffer complete dim=0

	int ptr_start = conv3x3_weight_ptr + (c_out*in_channels_after_pack*NUM_BUS_READS + c_in*NUM_BUS_READS)*9;

	for (int i = 0; i < NUM_BUS_READS*3*3; i ++){
#pragma HLS PIPELINE
		ddr_stage[i]= conv_weight_3x3_all[ptr_start+i];
	}
	for (int i = 0; i < NUM_BUS_READS; i ++){
		for (int row = 0; row < 3; row ++) {
			for (int col = 0; col < 3; col ++) {
#pragma HLS PIPELINE
				int index = i*9 + row*3 +col;
				uint512 ddr_ss = ddr_stage[index];
				for (int j = 0; j < NUM_WT_PACKS; j ++) {
					weight3x3_tile_buffer[i*NUM_WT_PACKS + j][row][col].range(WEIGHT_BITS-1, 0) = ddr_ss.range(WEIGHT_BITS-1+j*WEIGHT_BITS, j*WEIGHT_BITS);
				}
			}
		}
	}
}

void load_conv1x1_weights(
		uint32 weight1x1_tile_buffer[OUT_CHANNEL_PARALLELISM],
		uint512 conv_weight_1x1_all[6132],
		int conv1x1_weight_ptr,
		int c_out,
		int c_in,
		int in_channels_after_pack
)
{
	//#pragma HLS ARRAY_PARTITION variable=weight1x1_tile_buffer complete dim=1

	int ptr_start = conv1x1_weight_ptr + c_out*in_channels_after_pack*NUM_BUS_READS + c_in*NUM_BUS_READS;
	for (int i = 0; i < NUM_BUS_READS; i ++) {
#pragma HLS PIPELINE
		uint512 data_pack = conv_weight_1x1_all[ptr_start+i];
		for (int j = 0; j < NUM_WT_PACKS; j ++) {
			weight1x1_tile_buffer[i*NUM_WT_PACKS + j].range(WEIGHT_BITS-1, 0) = data_pack.range(WEIGHT_BITS-1+j*WEIGHT_BITS, j*WEIGHT_BITS);
		}
	}
}


void load_shortcut(
		uint4 out_buf_sc[OUT_CHANNEL_PARALLELISM][ROW_TILE_SIZE + 1][BUF_WIDTH_0],
		uint512 DDR_buf[DDR_BUF_DIM],

		int H_fmap_out,
		int in_channels,
		int out_channel_start,
		int row_tile_start,
		int switch_bank
)
{
#pragma HLS ARRAY_PARTITION variable=ddr_tmp cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf_sc complete dim=1

	int ddr_channel_ptr = out_channel_start/OUT_CHANNEL_PARALLELISM;
	if (switch_bank) {
		if (ddr_channel_ptr >= in_channels/OUT_CHANNEL_PARALLELISM) ddr_channel_ptr -= in_channels/OUT_CHANNEL_PARALLELISM;
		ddr_channel_ptr += DDR_CH_OFFSET;
	}

	int col_lim = ceil_div_4(H_fmap_out);
	Load_from_DDR:
	for (int i = 0; i < (ROW_TILE_SIZE + 1)*col_lim; i ++){
#pragma HLS PIPELINE
		ddr_stage[i] = DDR_buf[ddr_channel_ptr*H_fmap_out*col_lim + row_tile_start*col_lim + i];
	}

	Stage_for_shortcut:
	for (int row = 0; row < (ROW_TILE_SIZE + 1); row ++) {
		for (int col = 0; col < col_lim; col ++) {
#pragma HLS PIPELINE
			uint512 ddr_ss = ddr_stage[row*col_lim + col];
			for (int col_i = 0; col_i < 4; col_i ++) {
				ddr_tmp[row][col*4 + col_i].range(127, 0) = ddr_ss.range(col_i*128+127, col_i*128);
			}
		}
	}

	Write_to_shortcut:
	for (int row = 0; row < ROW_TILE_SIZE + 1; row ++){
		for (int col = 0; col < H_fmap_out; col ++){
#pragma HLS PIPELINE
			uint128 ddr_tt = ddr_tmp[row][col];
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++){
				out_buf_sc[ch][row][col].range(3, 0) = ddr_tt.range(ch*4+3, ch*4);
			}
		}
	}

}

void bn_relu_small(
		FIX_WT bn_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT bn_bias[OUT_CHANNEL_PARALLELISM],

		uint512 DDR_buf[DDR_BUF_DIM],
		int10 out_buf_all[OUT_CHANNEL_PARALLELISM][(ROW_TILE_SIZE+2)*(BUF_WIDTH_1+1)],
		uint32 feat_buf_all_1[FEAT_MAP_DIM],
		uint32 feat_buf_all_0[FEAT_MAP_DIM],

		int H_fmap_in,
		int H_fmap_out,
		int row_tile_start,
		int stride
)
{
	int odd = row_tile_start%2;
	int row_tile_offset = row_tile_start/stride + odd;
	for (int row = 0; row < (ROW_TILE_SIZE + 1 - odd)/stride; row ++) { // Stride 2
		for (int col = 0; col < H_fmap_out; col ++) { // Stride 2
#pragma HLS PIPELINE
			uint128 pack_tmp = 0;
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				FIX_FM_acc out_feature = out_buf_all[ch][(row*stride+odd+2)*(H_fmap_in+1)+col*stride+1];
				out_feature = bn_weight[ch]*out_feature + bn_bias[ch];

				uint4 tmp = to4bit(out_feature);
				pack_tmp.range(ch*4+3, ch*4) = tmp.range(3, 0);

				uint2 result_t = to2bit(out_feature);
				feat_buf_all_1[(row_tile_offset+row+1)*(H_fmap_out+1) + col][ch] = result_t[1]; // MSB
				feat_buf_all_0[(row_tile_offset+row+1)*(H_fmap_out+1) + col][ch] = result_t[0]; // LSB
			}
			ddr_tmp[row][col] = pack_tmp;
		}
	}

	int col_lim = ceil_div_4(H_fmap_out);
	for (int row = 0; row < (ROW_TILE_SIZE + 1 - odd)/stride; row ++) {
		for (int col = 0; col < col_lim; col ++){
#pragma HLS PIPELINE II=2
			uint512 ddr_ss = 0;
			for (int col_i = 0; col_i < 4; col_i ++) {
				ddr_ss.range(col_i*128+127, col_i*128) = ddr_tmp[row][col*4 + col_i].range(127, 0);
			}
			ddr_stage[row*col_lim + col] = ddr_ss;
		}
	}
	for (int i = 0; i < (ROW_TILE_SIZE + 1 - odd)/stride*col_lim; i ++) {
#pragma HLS PIPELINE
		DDR_buf[row_tile_offset*col_lim + i] = ddr_stage[i];
	}
}

void bn_relu_sc_relu(
		FIX_WT conv_threshold[OUT_CHANNEL_PARALLELISM],
		FIX_WT conv_bn_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT conv_bn_bias[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_shift_x[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_shift_y[OUT_CHANNEL_PARALLELISM],
		FIX_WT relu_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT bn_weight[OUT_CHANNEL_PARALLELISM],
		FIX_WT bn_bias[OUT_CHANNEL_PARALLELISM],

		uint512 DDR_buf[DDR_BUF_DIM],
		int10 out_buf_all[OUT_CHANNEL_PARALLELISM][(ROW_TILE_SIZE+2)*(BUF_WIDTH_1+1)],
		uint4 out_buf_sc[OUT_CHANNEL_PARALLELISM][ROW_TILE_SIZE+1][BUF_WIDTH_0],
		uint32 feat_buf_all_1[FEAT_MAP_DIM],
		uint32 feat_buf_all_0[FEAT_MAP_DIM],

		int H_fmap_in,
		int H_fmap_out,
		int c_out,
		int row_tile_start,
		int stride,
		int switch_bank
)
{
	FIX_FM_acc out_feature_alt0[OUT_CHANNEL_PARALLELISM];
	FIX_FM_acc out_feature_alt1[OUT_CHANNEL_PARALLELISM];
#pragma HLS ARRAY_PARTITION variable=out_feature_t0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_feature_t1 complete dim=1

	int10 out_buf_all_tmp_t0[OUT_CHANNEL_PARALLELISM];
	int10 out_buf_all_tmp_t1[OUT_CHANNEL_PARALLELISM];
#pragma HLS ARRAY_PARTITION variable=out_buf_all_tmp_t0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf_all_tmp_t1 complete dim=1

	const CONST_2 msb_scale = 2.0/3.0;
	const CONST_2 lsb_scale = 1.0/3.0;
	const CONST_2 ddr_scale = 2.0/15.0;
	const int out_buf_offset = (ROW_TILE_SIZE+2)*(H_fmap_in+1);
	const int fmap_ptr_offset_0 = c_out*(H_fmap_out)*(H_fmap_out);
	const int fmap_ptr_offset_1 = c_out*(H_fmap_out+1)*(H_fmap_out+1);
	int ddr_ptr = c_out + DDR_CH_OFFSET;
	if (switch_bank == 1) ddr_ptr = c_out;
	int odd = row_tile_start%2*(stride-1);
	int row_tile_offset = row_tile_start/stride + odd;


	for (int row = 0; row < (ROW_TILE_SIZE + (1 - odd)*(stride-1))/stride; row ++) {
		for (int col = 0; col < H_fmap_out; col ++) {
#pragma HLS PIPELINE
			int fmap_ptr = fmap_ptr_offset_0 + (row_tile_offset+row)*(H_fmap_out) + col + FEAT_BUF_OFFSET;
			if (switch_bank == 1) {
				// switch_bank = 0: prepare for 1x1 conv
				// switch_bank = 1: prepare for 3x3 conv
				fmap_ptr = fmap_ptr_offset_1 + (row_tile_offset+row+1)*(H_fmap_out+1) + col;
			}
			int out_buf_index = (row*stride+odd+2)*(H_fmap_in+1)+col*stride+1;

			// merge_tile
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				out_buf_all_tmp_t1[ch] = out_buf_all[ch][out_buf_index];
			}
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				out_feature_alt1[ch] = out_buf_all_tmp_t1[ch];
			}
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				out_buf_all_tmp_t0[ch] = out_buf_all[ch][out_buf_index+out_buf_offset];
			}
			// threshold
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				out_feature_alt0[ch] = out_feature_alt1[ch]*msb_scale;
				if (out_feature_alt0[ch] > conv_threshold[ch]) {
					out_feature_alt0[ch] += out_buf_all_tmp_t0[ch]*lsb_scale;
				}
			}

			// Batch-norm
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				out_feature_alt1[ch] = conv_bn_weight[ch]*out_feature_alt0[ch] + conv_bn_bias[ch];
			}

			// ReLU
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				out_feature_alt0[ch] = out_feature_alt1[ch] + relu_shift_x[ch];
				if (out_feature_alt0[ch] < 0) out_feature_alt0[ch]*= relu_weight[ch];
				out_feature_alt0[ch] += relu_shift_y[ch];
			}

			// Shortcut
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				out_feature_alt1[ch] = out_feature_alt0[ch] + out_buf_sc[ch][row][col]*ddr_scale - 1;
			}

			// Batch-norm
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				out_feature_alt0[ch] = bn_weight[ch]*out_feature_alt1[ch] + bn_bias[ch];
			}

			// Write-back
			uint128 pack_tmp = 0;
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				uint4 tmp = to4bit(out_feature_alt0[ch]);
				pack_tmp.range(ch*4+3, ch*4) = tmp.range(3, 0);
			}
			ddr_tmp[row][col] = pack_tmp;

			// quant and pack
			for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++) {
				uint2 result_t = to2bit(out_feature_alt0[ch]);
				feat_buf_all_1[fmap_ptr][ch] = result_t[1]; // MSB
				feat_buf_all_0[fmap_ptr][ch] = result_t[0]; // LSB
			}
		}
	}
	int col_lim = ceil_div_4(H_fmap_out);
	for (int row = 0; row < (ROW_TILE_SIZE + (1 - odd)*(stride-1))/stride; row ++) {
		for (int col = 0; col < col_lim; col ++){
#pragma HLS PIPELINE II=2
			uint512 ddr_ss = 0;
			for (int col_i = 0; col_i < 4; col_i ++) {
				ddr_ss.range(col_i*128+127, col_i*128) = ddr_tmp[row][col*4 + col_i].range(127, 0);
			}
			ddr_stage[row*col_lim + col] = ddr_ss;
		}
	}
	for (int i = 0; i < (ROW_TILE_SIZE + (1 - odd)*(stride-1))/stride*col_lim; i ++) {
#pragma HLS PIPELINE
		DDR_buf[ddr_ptr*H_fmap_out*col_lim + row_tile_offset*col_lim + i] = ddr_stage[i];
	}
}

void avgpool(
		uint4 out_buf_sc[OUT_CHANNEL_PARALLELISM][ROW_TILE_SIZE+1][BUF_WIDTH_0],
		int H_fmap,
		int row_tile_start
)
{
#pragma HLS DEPENDENCE variable=out_buf_sc array inter false
	int odd = row_tile_start %2;

	const CONST_2 ddr_scale = 2.0/15.0;

	// even 0, 2, 4, 6,
	// even_row 0, 1, 2, 3
	// odd: 1, 3, 5,
	// odd_row 0, 1, 2

	FIX_FM_acc tmp[OUT_CHANNEL_PARALLELISM];
	for (int row = 0; row < ROW_TILE_SIZE/2 + (1 - odd) ; row ++){
		for (int col = 0; col < H_fmap/2; col ++){
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
			int row_start = row*2 + odd;
			int col_start = col*2;
			for (int ii = 0; ii < 2; ii ++) {
				for (int jj = 0; jj < 2; jj ++) {
#pragma HLS PIPELINE
					for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++){
						FIX_FM_acc t = out_buf_sc[ch][row_start+ii][col_start+jj]*ddr_scale - 1;
						if (ii + jj == 0) tmp[ch] = t;
						else tmp[ch] += t;
					}
					if (ii + jj == 2){
						for (int ch = 0; ch < OUT_CHANNEL_PARALLELISM; ch ++){
							out_buf_sc[ch][row][col] = to4bit(tmp[ch]/(FIX_FM_acc)4.0);
						}
					}
				}
			}
		}
	}
}

#endif
