#include "bnn.h"
#include "layer.h"
#include "pgconv.h"
#include "dimension_def.h"

using namespace std;

static uint32 feat_buf_all_1[FEAT_MAP_DIM]; // 3*226*225 = 4881600 = 272 18K
static uint32 feat_buf_all_0[FEAT_MAP_DIM]; // 3*226*225 = 4881600 = 272 18K

static uint4 out_buf_sc[OUT_CHANNEL_PARALLELISM][ROW_TILE_SIZE+1][BUF_WIDTH_0]; // shortcut 32*8*114 = 917504
static int10 out_buf_all[OUT_CHANNEL_PARALLELISM][(ROW_TILE_SIZE+2)*(BUF_WIDTH_1+1)]; // all 32*10*226 = 2315240

static uint32 weight3x3_tile_buffer[OUT_CHANNEL_PARALLELISM][3][3];
static uint32 weight1x1_tile_buffer[OUT_CHANNEL_PARALLELISM];

static FIX_WT conv3x3_0_threshold[OUT_CHANNEL_PARALLELISM];
static FIX_WT conv3x3_1_weight[OUT_CHANNEL_PARALLELISM];
static FIX_WT conv3x3_1_bias[OUT_CHANNEL_PARALLELISM];
static FIX_WT pw_0_threshold[OUT_CHANNEL_PARALLELISM];
static FIX_WT pw_1_weight[OUT_CHANNEL_PARALLELISM];
static FIX_WT pw_1_bias[OUT_CHANNEL_PARALLELISM];
static FIX_WT relu1_shift_x[OUT_CHANNEL_PARALLELISM];
static FIX_WT relu1_shift_y[OUT_CHANNEL_PARALLELISM];
static FIX_WT relu1_weight[OUT_CHANNEL_PARALLELISM];
static FIX_WT relu2_shift_x[OUT_CHANNEL_PARALLELISM];
static FIX_WT relu2_shift_y[OUT_CHANNEL_PARALLELISM];
static FIX_WT relu2_weight[OUT_CHANNEL_PARALLELISM];
static FIX_WT bn1_weight[OUT_CHANNEL_PARALLELISM];
static FIX_WT bn1_bias[OUT_CHANNEL_PARALLELISM];
static FIX_WT bn2_weight[OUT_CHANNEL_PARALLELISM];
static FIX_WT bn2_bias[OUT_CHANNEL_PARALLELISM];


void bundle_conv_s1(
		uint512 conv_weight_3x3_all_new[49104],
		uint512 weights_all[WEIGHTS_ALL_DIM],
		uint512 DDR_buf_pack[DDR_BUF_DIM],

		int in_channels_after_pack,
		int out_channels_after_tile,
		int conv3x3_weight_ptr,
		int weights_all_ptr,
		int conv_in_channels,
		int conv_out_channels,
		int pw_out_channels,
		int H_fmap_in,
		int H_fmap_out,
		int switch_bank,
		int stride
)
{
#pragma HLS INLINE

	int in_channel_start;
	int out_channel_start;
	int out_buf_col_start;
	int row_tile_start;
	int row_count_after_tile = H_fmap_in/ROW_TILE_SIZE;
	if (H_fmap_in < ROW_TILE_SIZE) row_count_after_tile = 1;

	for (int c_out = 0; c_out < out_channels_after_tile; c_out ++) {
		load_layer_1D_weights_conv(
				conv3x3_0_threshold,
				conv3x3_1_weight, conv3x3_1_bias,
				relu1_shift_x, relu1_shift_y, relu1_weight,
				bn1_weight, bn1_bias,

				weights_all, weights_all_ptr,
				c_out, conv_out_channels, pw_out_channels
		);
		for (int row_t = 0; row_t < row_count_after_tile; row_t ++) {
			for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
				out_channel_start = c_out*OUT_CHANNEL_PARALLELISM;
				row_tile_start = row_t*ROW_TILE_SIZE;
				in_channel_start = c_in*IN_CHANNEL_BITPACK;

				load_conv3x3_weights(
						weight3x3_tile_buffer,
						conv_weight_3x3_all_new,
						conv3x3_weight_ptr,
						c_out, c_in,
						in_channels_after_pack
				);
				out_buf_col_start = 0;
				pg_conv3x3_tile(
						feat_buf_all_1, weight3x3_tile_buffer, out_buf_all,
						c_in, H_fmap_in, row_tile_start, out_buf_col_start
				);
				out_buf_col_start = (ROW_TILE_SIZE+2)*(H_fmap_in+1);
				pg_conv3x3_tile(
						feat_buf_all_0, weight3x3_tile_buffer, out_buf_all,
						c_in, H_fmap_out, row_tile_start, out_buf_col_start
				);
			}
			load_shortcut(
					out_buf_sc, DDR_buf_pack,
					H_fmap_in, conv_in_channels,
					out_channel_start, row_tile_start, switch_bank
			);
			bn_relu_sc_relu(
					conv3x3_0_threshold, conv3x3_1_weight, conv3x3_1_bias,
					relu1_shift_x, relu1_shift_y, relu1_weight,
					bn1_weight, bn1_bias,

					DDR_buf_pack,
					out_buf_all, out_buf_sc, feat_buf_all_1, feat_buf_all_0,

					H_fmap_in, H_fmap_out, c_out,
					row_tile_start,
					stride, switch_bank
			);
		}
	}
}

void bundle_conv_s2(
		uint512 conv_weight_3x3_all_new[49104],
		uint512 weights_all[WEIGHTS_ALL_DIM],
		uint512 DDR_buf_pack[DDR_BUF_DIM],

		int in_channels_after_pack,
		int out_channels_after_tile,
		int conv3x3_weight_ptr,
		int weights_all_ptr,
		int conv_in_channels,
		int conv_out_channels,
		int pw_out_channels,
		int H_fmap_in,
		int H_fmap_out,
		int switch_bank,
		int stride
)
{
#pragma HLS INLINE

	int in_channel_start;
	int out_channel_start;
	int out_buf_col_start;
	int row_tile_start;

	for (int c_out = 0; c_out < out_channels_after_tile; c_out ++) {
		load_layer_1D_weights_conv(
				conv3x3_0_threshold,
				conv3x3_1_weight, conv3x3_1_bias,
				relu1_shift_x, relu1_shift_y, relu1_weight,
				bn1_weight, bn1_bias,

				weights_all, weights_all_ptr,
				c_out, conv_out_channels, pw_out_channels
		);
		for (int row_t = 0; row_t < H_fmap_in/ROW_TILE_SIZE; row_t ++) {
			for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
				out_channel_start = c_out*OUT_CHANNEL_PARALLELISM;
				row_tile_start = row_t*ROW_TILE_SIZE;
				in_channel_start = c_in*IN_CHANNEL_BITPACK;

				load_conv3x3_weights(
						weight3x3_tile_buffer,
						conv_weight_3x3_all_new,
						conv3x3_weight_ptr,
						c_out, c_in,
						in_channels_after_pack
				);
				out_buf_col_start = 0;
				pg_conv3x3_tile(
						feat_buf_all_1, weight3x3_tile_buffer, out_buf_all,
						c_in, H_fmap_in, row_tile_start, out_buf_col_start
				);
				out_buf_col_start = (ROW_TILE_SIZE+2)*(H_fmap_in+1);
				pg_conv3x3_tile(
						feat_buf_all_0, weight3x3_tile_buffer, out_buf_all,
						c_in, H_fmap_in, row_tile_start, out_buf_col_start
				);
			}
			load_shortcut(
					out_buf_sc, DDR_buf_pack,
					H_fmap_in, conv_in_channels,
					out_channel_start, row_tile_start, switch_bank
			);
			avgpool(out_buf_sc, H_fmap_in, row_tile_start);
			bn_relu_sc_relu(
					conv3x3_0_threshold, conv3x3_1_weight, conv3x3_1_bias,
					relu1_shift_x, relu1_shift_y, relu1_weight,
					bn1_weight, bn1_bias,

					DDR_buf_pack,
					out_buf_all, out_buf_sc, feat_buf_all_1, feat_buf_all_0,

					H_fmap_in, H_fmap_out, c_out,
					row_tile_start,
					stride, switch_bank
			);
		}
	}
}

void bundle_pw(
		uint512 conv_weight_1x1_all_new[49104],
		uint512 weights_all[WEIGHTS_ALL_DIM],
		uint512 DDR_buf_pack[DDR_BUF_DIM],

		int in_channels_after_pack,
		int out_channels_after_tile,
		int conv1x1_weight_ptr,
		int weights_all_ptr,
		int conv_out_channels,
		int pw_in_channels,
		int pw_out_channels,
		int H_fmap_in,
		int H_fmap_out,
		int switch_bank,
		int stride
)
{
#pragma HLS INLINE

	int in_channel_start;
	int out_channel_start;
	int out_buf_col_start;
	int row_tile_start;
	int row_count_after_tile = H_fmap_in/ROW_TILE_SIZE;
	if (H_fmap_in < ROW_TILE_SIZE) row_count_after_tile = 1;

	for (int c_out = 0; c_out < out_channels_after_tile; c_out ++) {
		load_layer_1D_weights_pw(
				pw_0_threshold,
				pw_1_weight, pw_1_bias,
				relu2_shift_x, relu2_shift_y, relu2_weight,
				bn2_weight, bn2_bias,

				weights_all, weights_all_ptr,
				c_out, conv_out_channels, pw_out_channels
		);
		for (int row_t = 0; row_t < row_count_after_tile; row_t ++) {
			for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
				out_channel_start = c_out*OUT_CHANNEL_PARALLELISM;
				row_tile_start = row_t*ROW_TILE_SIZE;
				in_channel_start = c_in*IN_CHANNEL_BITPACK;

				load_conv1x1_weights(
						weight1x1_tile_buffer,
						conv_weight_1x1_all_new,
						conv1x1_weight_ptr,
						c_out, c_in,
						in_channels_after_pack
				);
				out_buf_col_start = 0;
				pg_conv1x1_tile(
						feat_buf_all_1, weight1x1_tile_buffer, out_buf_all,
						c_in, H_fmap_in, row_tile_start, out_buf_col_start
				);
				out_buf_col_start = (ROW_TILE_SIZE+2)*(H_fmap_in+1);
				pg_conv1x1_tile(
						feat_buf_all_0, weight1x1_tile_buffer, out_buf_all,
						c_in, H_fmap_in, row_tile_start, out_buf_col_start
				);
			}
			load_shortcut(
					out_buf_sc, DDR_buf_pack,
					H_fmap_in, pw_in_channels,
					out_channel_start, row_tile_start, switch_bank
			);
			bn_relu_sc_relu(
					pw_0_threshold, pw_1_weight, pw_1_bias,
					relu2_shift_x, relu2_shift_y, relu2_weight,
					bn2_weight, bn2_bias,

					DDR_buf_pack,
					out_buf_all, out_buf_sc, feat_buf_all_1, feat_buf_all_0,

					H_fmap_in, H_fmap_out, c_out,
					row_tile_start,
					stride, switch_bank
			);
		}
	}
}

void FracNet(
		uint32 image_thermo[3*224*224],
		uint512 conv_weight_3x3_all_new[49104],
		uint512 conv_weight_1x1_all_new[6132],
		uint512 weights_all[WEIGHTS_ALL_DIM],
		uint512 DDR_buf_pack[DDR_BUF_DIM]
)
{
#pragma HLS ALLOCATION instances=bundle_conv_s1 limit=1 function
#pragma HLS ALLOCATION instances=bundle_conv_s2 limit=1 function
#pragma HLS ALLOCATION instances=bundle_pw limit=1 function
#pragma HLS ALLOCATION instances=pg_conv3x3_tile limit=1 function
#pragma HLS ALLOCATION instances=pg_conv1x1_tile limit=1 function
#pragma HLS ALLOCATION instances=bn_relu_sc_relu limit=1 function
#pragma HLS ALLOCATION instances=matmul32 limit=1 function


#pragma HLS INTERFACE m_axi depth=151528 port=image_thermo offset=slave bundle=BUS32


#pragma HLS INTERFACE m_axi depth=89998 port=conv_weight_3x3_all_new offset=slave bundle=BUS512
#pragma HLS INTERFACE m_axi depth=89998 port=conv_weight_1x1_all_new offset=slave bundle=BUS512

#pragma HLS INTERFACE m_axi depth=89998 port=weights_all offset=slave bundle=BUS512

#pragma HLS INTERFACE m_axi depth=203400 port=DDR_buf_pack offset=slave bundle=DDR512

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL


#pragma HLS ARRAY_PARTITION variable=out_buf_sc complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf_all complete dim=1

#pragma HLS ARRAY_PARTITION variable=weight3x3_tile_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight3x3_tile_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight3x3_tile_buffer complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight1x1_tile_buffer complete dim=1

#pragma HLS ARRAY_PARTITION variable=conv3x3_0_threshold complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv3x3_1_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv3x3_1_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=pw_0_threshold complete dim=1
#pragma HLS ARRAY_PARTITION variable=pw_1_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=pw_1_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu1_shift_x complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu1_shift_y complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu1_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu2_shift_x complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu2_shift_y complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu2_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=bn1_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=bn1_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=bn2_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=bn2_bias complete dim=1


#pragma HLS RESET variable=feat_buf_all_1 off
#pragma HLS RESET variable=feat_buf_all_0 off

#pragma HLS RESET variable=out_buf_sc off
#pragma HLS RESET variable=out_buf_all off

#pragma HLS RESET variable=weight3x3_tile_buffer off
#pragma HLS RESET variable=weight1x1_tile_buffer off

#pragma HLS RESET variable=conv3x3_0_threshold off
#pragma HLS RESET variable=conv3x3_1_weight off
#pragma HLS RESET variable=conv3x3_1_bias off
#pragma HLS RESET variable=pw_0_threshold off
#pragma HLS RESET variable=pw_1_weight off
#pragma HLS RESET variable=pw_1_bias off
#pragma HLS RESET variable=relu1_shift_x off
#pragma HLS RESET variable=relu1_shift_y off
#pragma HLS RESET variable=relu1_weight off
#pragma HLS RESET variable=relu2_shift_x off
#pragma HLS RESET variable=relu2_shift_y off
#pragma HLS RESET variable=relu2_weight off
#pragma HLS RESET variable=bn1_weight off
#pragma HLS RESET variable=bn1_bias off
#pragma HLS RESET variable=bn2_weight off
#pragma HLS RESET variable=bn2_bias off



	int H_fmap_in, H_fmap_out, stride;
	int conv_in_channels, conv_out_channels;
	int pw_in_channels, pw_out_channels;
	int in_channels_after_pack, out_channels_after_tile;
	int in_channel_start, out_channel_start, row_tile_start, out_buf_col_start;
	int conv3x3_weight_ptr, conv1x1_weight_ptr, weights_all_ptr;
	int conv3x3_weight_ptr_inc, conv1x1_weight_ptr_inc, weights_all_inc;
	int switch_bank;


	////////////////////////////////////////////////
	//////////// GET IMAGE + CONV1 /////////////////
	////////////////////////////////////////////////
	conv3x3_weight_ptr = 0;
	weights_all_ptr = 0;
	in_channels_after_pack = 3;
	conv_in_channels = 96;
	conv_out_channels = 32;
	H_fmap_in = 224;
	H_fmap_out = 112;
	stride=2;

	load_1D_weights(weights_all, weights_all_ptr, bn1_weight);
	weights_all_ptr += 2;
	load_1D_weights(weights_all, weights_all_ptr, bn1_bias);
	weights_all_ptr += 2;

	LOOP_Conv1:
	for (int row_t = 0; row_t < H_fmap_in/ROW_TILE_SIZE; row_t++){
		row_tile_start = row_t*ROW_TILE_SIZE;
		for (int c_in = 0; c_in < in_channels_after_pack; c_in ++) {
			LOOP_GetImg:
			for (int row = 0; row < ROW_TILE_SIZE+2; row ++){
				for (int col = 0; col < 225; col ++){
#pragma HLS PIPELINE
					int index_thermo = c_in*224*224 + (row_tile_start+row-1)*224 + col;
					int index_feature = (row_tile_start+row)*225 + col;
					if (row_tile_start+row >= 1 && row_tile_start+row <= 224 && col <= 223){
						feat_buf_all_0[index_feature] = image_thermo[index_thermo];
					} else {
						feat_buf_all_0[index_feature] = 0;
					}
				}
			}
			load_conv3x3_weights(
					weight3x3_tile_buffer,
					conv_weight_3x3_all_new,
					conv3x3_weight_ptr,
					0, c_in,
					in_channels_after_pack
			);
			out_buf_col_start = 0;
			pg_conv3x3_tile(
					feat_buf_all_0, weight3x3_tile_buffer, out_buf_all,
					c_in, H_fmap_in, row_tile_start, out_buf_col_start
			);
		}
		bn_relu_small(
				bn1_weight, bn1_bias,

				DDR_buf_pack,
				out_buf_all, feat_buf_all_1, feat_buf_all_0,

				H_fmap_in, H_fmap_out,
				row_tile_start, stride
		);
	}
#ifdef LAYER_TEST
	cout << "Conv1 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 32;
	conv_out_channels = 32;
	pw_in_channels = 32;
	pw_out_channels = 64;

	////////////////////////////////////////////////
	//////////// layer1_conv ///////////////////////

	H_fmap_in = 112;
	H_fmap_out = 112;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr = 6*9;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	weights_all_ptr = 4;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;

	LOOP_layer1_0:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer1_pw /////////////////////////

	H_fmap_in = 112;
	H_fmap_out = 112;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr = 0;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer1_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L1 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 64;
	conv_out_channels = 64;
	pw_in_channels = 64;
	pw_out_channels = 128;

	////////////////////////////////////////////////
	//////////// layer2_conv /////////////////////

	H_fmap_in = 112;
	H_fmap_out = 56;
	stride = 2;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;

	LOOP_layer2_conv:
	bundle_conv_s2(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer2_pw /////////////////////////

	H_fmap_in = 56;
	H_fmap_out = 56;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer2_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L2 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 128;
	conv_out_channels = 128;
	pw_in_channels = 128;
	pw_out_channels = 128;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;

	////////////////////////////////////////////////
	//////////// layer3_conv /////////////////////

	H_fmap_in = 56;
	H_fmap_out = 56;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer3_conv:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer3_pw /////////////////////////

	H_fmap_in = 56;
	H_fmap_out = 56;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer3_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L3 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 4 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 128;
	conv_out_channels = 128;
	pw_in_channels = 128;
	pw_out_channels = 256;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;

	////////////////////////////////////////////////
	//////////// layer4_conv /////////////////////

	H_fmap_in = 56;
	H_fmap_out = 28;
	stride = 2;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer4_conv:
	bundle_conv_s2(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer4_pw /////////////////////////

	H_fmap_in = 28;
	H_fmap_out = 28;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer4_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L4 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 5 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 256;
	conv_out_channels = 256;
	pw_in_channels = 256;
	pw_out_channels = 256;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;

	////////////////////////////////////////////////
	//////////// layer5_conv /////////////////////

	H_fmap_in = 28;
	H_fmap_out = 28;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer5_conv:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer5_pw /////////////////////////

	H_fmap_in = 28;
	H_fmap_out = 28;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer5_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L5 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 6 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 256;
	conv_out_channels = 256;
	pw_in_channels = 256;
	pw_out_channels = 512;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;

	////////////////////////////////////////////////
	//////////// layer6_conv /////////////////////

	H_fmap_in = 28;
	H_fmap_out = 14;
	stride = 2;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer6_conv:
	bundle_conv_s2(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer6_pw /////////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer6_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L6 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 7 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 512;
	conv_out_channels = 512;
	pw_in_channels = 512;
	pw_out_channels = 512;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;

	////////////////////////////////////////////////
	//////////// layer7_conv /////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer7_conv:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer7_pw /////////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer7_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L7 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 8 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 512;
	conv_out_channels = 512;
	pw_in_channels = 512;
	pw_out_channels = 512;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;


	////////////////////////////////////////////////
	//////////// layer8_conv /////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer8_conv:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer8_pw /////////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer8_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L8 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 9 ///////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 512;
	conv_out_channels = 512;
	pw_in_channels = 512;
	pw_out_channels = 512;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;


	////////////////////////////////////////////////
	//////////// layer9_conv /////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer9_conv:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer9_pw /////////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer9_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L9 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 10 //////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 512;
	conv_out_channels = 512;
	pw_in_channels = 512;
	pw_out_channels = 512;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;


	////////////////////////////////////////////////
	//////////// layer10_conv /////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer10_conv:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);


	////////////////////////////////////////////////
	//////////// layer10_pw ////////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer10_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L10 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 11 //////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 512;
	conv_out_channels = 512;
	pw_in_channels = 512;
	pw_out_channels = 512;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;


	////////////////////////////////////////////////
	//////////// layer11_conv /////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer11_conv:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer11_pw ////////////////////////

	H_fmap_in = 14;
	H_fmap_out = 14;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer11_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L11 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 12 //////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 512;
	conv_out_channels = 512;
	pw_in_channels = 512;
	pw_out_channels = 1024;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;


	////////////////////////////////////////////////
	//////////// layer12_conv /////////////////////

	H_fmap_in = 14;
	H_fmap_out = 7;
	stride = 2;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer12_conv:
	bundle_conv_s2(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer12_pw /////////////////////////

	H_fmap_in = 7;
	H_fmap_out = 7;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer12_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L12 Done" << endl;
#endif

	////////////////////////////////////////////////
	//////////// LAYER 13 //////////////////////////
	////////////////////////////////////////////////

	conv_in_channels = 1024;
	conv_out_channels = 1024;
	pw_in_channels = 1024;
	pw_out_channels = 1024;

	weights_all_ptr += weights_all_inc;
	weights_all_inc = (8*conv_out_channels/OUT_CHANNEL_PARALLELISM + 8*pw_out_channels/OUT_CHANNEL_PARALLELISM)*NUM_BUS_READS_OTHER;


	////////////////////////////////////////////////
	//////////// layer13_conv /////////////////////

	H_fmap_in = 7;
	H_fmap_out = 7;
	stride = 1;
	switch_bank = 0;

	in_channels_after_pack = conv_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = conv_out_channels/OUT_CHANNEL_PARALLELISM;

	conv3x3_weight_ptr += conv3x3_weight_ptr_inc;
	conv3x3_weight_ptr_inc = conv_in_channels*conv_out_channels/BUS_WIDTH*9;

	LOOP_layer13_conv:
	bundle_conv_s1(
			conv_weight_3x3_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv3x3_weight_ptr,
			weights_all_ptr,
			conv_in_channels,
			conv_out_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);

	////////////////////////////////////////////////
	//////////// layer13_pw ////////////////////////

	H_fmap_in = 7;
	H_fmap_out = 7;
	stride = 1;
	switch_bank = 1;

	in_channels_after_pack = pw_in_channels/IN_CHANNEL_BITPACK;
	out_channels_after_tile = pw_out_channels/OUT_CHANNEL_PARALLELISM;

	conv1x1_weight_ptr += conv1x1_weight_ptr_inc;
	conv1x1_weight_ptr_inc = pw_in_channels*pw_out_channels/BUS_WIDTH;

	LOOP_layer13_pw:
	bundle_pw(
			conv_weight_1x1_all_new, weights_all, DDR_buf_pack,

			in_channels_after_pack,
			out_channels_after_tile,
			conv1x1_weight_ptr,
			weights_all_ptr,
			conv_out_channels,
			pw_in_channels,
			pw_out_channels,
			H_fmap_in,
			H_fmap_out,
			switch_bank,
			stride
	);
#ifdef LAYER_TEST
	cout << "L13 Done" << endl;
#endif
}
