#ifndef ARRAY_DEF_TB
#define ARRAY_DEF_TB

#define NUM_TESTS 1


using namespace std;

////////////////////////////
// Individual weight buffer
////////////////////////////
float layer_0_0_weight[32][96][3][3];
float layer_0_1_weight[32];
float layer_0_1_bias[32];

float layer_1_conv3x3_0_weight[32][32][3][3];
float layer_1_conv3x3_0_threshold[32];
float layer_1_conv3x3_1_weight[32];
float layer_1_conv3x3_1_bias[32];
float layer_1_pointwise_0_weight[64][32];
float layer_1_pointwise_0_threshold[64];
float layer_1_pointwise_1_weight[64];
float layer_1_pointwise_1_bias[64];
float layer_1_rprelu1_shift_x_bias[32];
float layer_1_rprelu1_shift_y_bias[32];
float layer_1_rprelu1_prelu_weight[32];
float layer_1_rprelu2_shift_x_bias[64];
float layer_1_rprelu2_shift_y_bias[64];
float layer_1_rprelu2_prelu_weight[64];
float layer_1_shiftbn1_weight[32];
float layer_1_shiftbn1_bias[32];
float layer_1_shiftbn2_weight[64];
float layer_1_shiftbn2_bias[64];

float layer_2_conv3x3_0_weight[64][64][3][3];
float layer_2_conv3x3_0_threshold[64];
float layer_2_conv3x3_1_weight[64];
float layer_2_conv3x3_1_bias[64];
float layer_2_pointwise_0_weight[128][64];
float layer_2_pointwise_0_threshold[128];
float layer_2_pointwise_1_weight[128];
float layer_2_pointwise_1_bias[128];
float layer_2_rprelu1_shift_x_bias[64];
float layer_2_rprelu1_shift_y_bias[64];
float layer_2_rprelu1_prelu_weight[64];
float layer_2_rprelu2_shift_x_bias[128];
float layer_2_rprelu2_shift_y_bias[128];
float layer_2_rprelu2_prelu_weight[128];
float layer_2_shiftbn1_weight[64];
float layer_2_shiftbn1_bias[64];
float layer_2_shiftbn2_weight[128];
float layer_2_shiftbn2_bias[128];

float layer_3_conv3x3_0_weight[128][128][3][3];
float layer_3_conv3x3_0_threshold[128];
float layer_3_conv3x3_1_weight[128];
float layer_3_conv3x3_1_bias[128];
float layer_3_pointwise_0_weight[128][128];
float layer_3_pointwise_0_threshold[128];
float layer_3_pointwise_1_weight[128];
float layer_3_pointwise_1_bias[128];
float layer_3_rprelu1_shift_x_bias[128];
float layer_3_rprelu1_shift_y_bias[128];
float layer_3_rprelu1_prelu_weight[128];
float layer_3_rprelu2_shift_x_bias[128];
float layer_3_rprelu2_shift_y_bias[128];
float layer_3_rprelu2_prelu_weight[128];
float layer_3_shiftbn1_weight[128];
float layer_3_shiftbn1_bias[128];
float layer_3_shiftbn2_weight[128];
float layer_3_shiftbn2_bias[128];

float layer_4_conv3x3_0_weight[128][128][3][3];
float layer_4_conv3x3_0_threshold[128];
float layer_4_conv3x3_1_weight[128];
float layer_4_conv3x3_1_bias[128];
float layer_4_pointwise_0_weight[256][128];
float layer_4_pointwise_0_threshold[256];
float layer_4_pointwise_1_weight[256];
float layer_4_pointwise_1_bias[256];
float layer_4_rprelu1_shift_x_bias[128];
float layer_4_rprelu1_shift_y_bias[128];
float layer_4_rprelu1_prelu_weight[128];
float layer_4_rprelu2_shift_x_bias[256];
float layer_4_rprelu2_shift_y_bias[256];
float layer_4_rprelu2_prelu_weight[256];
float layer_4_shiftbn1_weight[128];
float layer_4_shiftbn1_bias[128];
float layer_4_shiftbn2_weight[256];
float layer_4_shiftbn2_bias[256];

float layer_5_conv3x3_0_weight[256][256][3][3];
float layer_5_conv3x3_0_threshold[256];
float layer_5_conv3x3_1_weight[256];
float layer_5_conv3x3_1_bias[256];
float layer_5_pointwise_0_weight[256][256];
float layer_5_pointwise_0_threshold[256];
float layer_5_pointwise_1_weight[256];
float layer_5_pointwise_1_bias[256];
float layer_5_rprelu1_shift_x_bias[256];
float layer_5_rprelu1_shift_y_bias[256];
float layer_5_rprelu1_prelu_weight[256];
float layer_5_rprelu2_shift_x_bias[256];
float layer_5_rprelu2_shift_y_bias[256];
float layer_5_rprelu2_prelu_weight[256];
float layer_5_shiftbn1_weight[256];
float layer_5_shiftbn1_bias[256];
float layer_5_shiftbn2_weight[256];
float layer_5_shiftbn2_bias[256];

float layer_6_conv3x3_0_weight[256][256][3][3];
float layer_6_conv3x3_0_threshold[256];
float layer_6_conv3x3_1_weight[256];
float layer_6_conv3x3_1_bias[256];
float layer_6_pointwise_0_weight[512][256];
float layer_6_pointwise_0_threshold[512];
float layer_6_pointwise_1_weight[512];
float layer_6_pointwise_1_bias[512];
float layer_6_rprelu1_shift_x_bias[256];
float layer_6_rprelu1_shift_y_bias[256];
float layer_6_rprelu1_prelu_weight[256];
float layer_6_rprelu2_shift_x_bias[512];
float layer_6_rprelu2_shift_y_bias[512];
float layer_6_rprelu2_prelu_weight[512];
float layer_6_shiftbn1_weight[256];
float layer_6_shiftbn1_bias[256];
float layer_6_shiftbn2_weight[512];
float layer_6_shiftbn2_bias[512];

float layer_7_conv3x3_0_weight[512][512][3][3];
float layer_7_conv3x3_0_threshold[512];
float layer_7_conv3x3_1_weight[512];
float layer_7_conv3x3_1_bias[512];
float layer_7_pointwise_0_weight[512][512];
float layer_7_pointwise_0_threshold[512];
float layer_7_pointwise_1_weight[512];
float layer_7_pointwise_1_bias[512];
float layer_7_rprelu1_shift_x_bias[512];
float layer_7_rprelu1_shift_y_bias[512];
float layer_7_rprelu1_prelu_weight[512];
float layer_7_rprelu2_shift_x_bias[512];
float layer_7_rprelu2_shift_y_bias[512];
float layer_7_rprelu2_prelu_weight[512];
float layer_7_shiftbn1_weight[512];
float layer_7_shiftbn1_bias[512];
float layer_7_shiftbn2_weight[512];
float layer_7_shiftbn2_bias[512];

float layer_8_conv3x3_0_weight[512][512][3][3];
float layer_8_conv3x3_0_threshold[512];
float layer_8_conv3x3_1_weight[512];
float layer_8_conv3x3_1_bias[512];
float layer_8_pointwise_0_weight[512][512];
float layer_8_pointwise_0_threshold[512];
float layer_8_pointwise_1_weight[512];
float layer_8_pointwise_1_bias[512];
float layer_8_rprelu1_shift_x_bias[512];
float layer_8_rprelu1_shift_y_bias[512];
float layer_8_rprelu1_prelu_weight[512];
float layer_8_rprelu2_shift_x_bias[512];
float layer_8_rprelu2_shift_y_bias[512];
float layer_8_rprelu2_prelu_weight[512];
float layer_8_shiftbn1_weight[512];
float layer_8_shiftbn1_bias[512];
float layer_8_shiftbn2_weight[512];
float layer_8_shiftbn2_bias[512];

float layer_9_conv3x3_0_weight[512][512][3][3];
float layer_9_conv3x3_0_threshold[512];
float layer_9_conv3x3_1_weight[512];
float layer_9_conv3x3_1_bias[512];
float layer_9_pointwise_0_weight[512][512];
float layer_9_pointwise_0_threshold[512];
float layer_9_pointwise_1_weight[512];
float layer_9_pointwise_1_bias[512];
float layer_9_rprelu1_shift_x_bias[512];
float layer_9_rprelu1_shift_y_bias[512];
float layer_9_rprelu1_prelu_weight[512];
float layer_9_rprelu2_shift_x_bias[512];
float layer_9_rprelu2_shift_y_bias[512];
float layer_9_rprelu2_prelu_weight[512];
float layer_9_shiftbn1_weight[512];
float layer_9_shiftbn1_bias[512];
float layer_9_shiftbn2_weight[512];
float layer_9_shiftbn2_bias[512];

float layer_10_conv3x3_0_weight[512][512][3][3];
float layer_10_conv3x3_0_threshold[512];
float layer_10_conv3x3_1_weight[512];
float layer_10_conv3x3_1_bias[512];
float layer_10_pointwise_0_weight[512][512];
float layer_10_pointwise_0_threshold[512];
float layer_10_pointwise_1_weight[512];
float layer_10_pointwise_1_bias[512];
float layer_10_rprelu1_shift_x_bias[512];
float layer_10_rprelu1_shift_y_bias[512];
float layer_10_rprelu1_prelu_weight[512];
float layer_10_rprelu2_shift_x_bias[512];
float layer_10_rprelu2_shift_y_bias[512];
float layer_10_rprelu2_prelu_weight[512];
float layer_10_shiftbn1_weight[512];
float layer_10_shiftbn1_bias[512];
float layer_10_shiftbn2_weight[512];
float layer_10_shiftbn2_bias[512];

float layer_11_conv3x3_0_weight[512][512][3][3];
float layer_11_conv3x3_0_threshold[512];
float layer_11_conv3x3_1_weight[512];
float layer_11_conv3x3_1_bias[512];
float layer_11_pointwise_0_weight[512][512];
float layer_11_pointwise_0_threshold[512];
float layer_11_pointwise_1_weight[512];
float layer_11_pointwise_1_bias[512];
float layer_11_rprelu1_shift_x_bias[512];
float layer_11_rprelu1_shift_y_bias[512];
float layer_11_rprelu1_prelu_weight[512];
float layer_11_rprelu2_shift_x_bias[512];
float layer_11_rprelu2_shift_y_bias[512];
float layer_11_rprelu2_prelu_weight[512];
float layer_11_shiftbn1_weight[512];
float layer_11_shiftbn1_bias[512];
float layer_11_shiftbn2_weight[512];
float layer_11_shiftbn2_bias[512];

float layer_12_conv3x3_0_weight[512][512][3][3];
float layer_12_conv3x3_0_threshold[512];
float layer_12_conv3x3_1_weight[512];
float layer_12_conv3x3_1_bias[512];
float layer_12_pointwise_0_weight[1024][512];
float layer_12_pointwise_0_threshold[1024];
float layer_12_pointwise_1_weight[1024];
float layer_12_pointwise_1_bias[1024];
float layer_12_rprelu1_shift_x_bias[512];
float layer_12_rprelu1_shift_y_bias[512];
float layer_12_rprelu1_prelu_weight[512];
float layer_12_rprelu2_shift_x_bias[1024];
float layer_12_rprelu2_shift_y_bias[1024];
float layer_12_rprelu2_prelu_weight[1024];
float layer_12_shiftbn1_weight[512];
float layer_12_shiftbn1_bias[512];
float layer_12_shiftbn2_weight[1024];
float layer_12_shiftbn2_bias[1024];

float layer_13_conv3x3_0_weight[1024][1024][3][3];
float layer_13_conv3x3_0_threshold[1024];
float layer_13_conv3x3_1_weight[1024];
float layer_13_conv3x3_1_bias[1024];
float layer_13_pointwise_0_weight[1024][1024];
float layer_13_pointwise_0_threshold[1024];
float layer_13_pointwise_1_weight[1024];
float layer_13_pointwise_1_bias[1024];
float layer_13_rprelu1_shift_x_bias[1024];
float layer_13_rprelu1_shift_y_bias[1024];
float layer_13_rprelu1_prelu_weight[1024];
float layer_13_rprelu2_shift_x_bias[1024];
float layer_13_rprelu2_shift_y_bias[1024];
float layer_13_rprelu2_prelu_weight[1024];
float layer_13_shiftbn1_weight[1024];
float layer_13_shiftbn1_bias[1024];
float layer_13_shiftbn2_weight[1024];
float layer_13_shiftbn2_bias[1024];

float fc_weight[1000][1024];
float fc_bias[1000];

///////////////////
// Output buffers
///////////////////
float layer_0_0_output[32][112][112];
float layer_0_0_output_quant[32][112][112];
float layer_1_conv3x3_0_output[32][112][112];
float layer_1_conv3x3_0_output_quant[32][112][112];
float layer_1_pointwise_0_output[64][112][112];
float layer_1_conv3x3_0_output_concat[64][112][112];

float layer_1_pointwise_0_output_quant[64][112][112];
float layer_2_conv3x3_0_output[64][56][56];
float layer_1_pointwise_0_output_avgpool[64][56][56];
float layer_2_conv3x3_0_output_quant[64][56][56];
float layer_2_pointwise_0_output[128][56][56];
float layer_2_conv3x3_0_output_concat[128][56][56];

float layer_2_pointwise_0_output_quant[128][56][56];
float layer_3_conv3x3_0_output[128][56][56];
float layer_3_conv3x3_0_output_quant[128][56][56];
float layer_3_pointwise_0_output[128][56][56];

float layer_3_pointwise_0_output_quant[128][56][56];
float layer_4_conv3x3_0_output[128][28][28];
float layer_3_pointwise_0_output_avgpool[128][28][28];
float layer_4_conv3x3_0_output_quant[128][28][28];
float layer_4_pointwise_0_output[256][28][28];
float layer_4_conv3x3_0_output_concat[256][28][28];

float layer_4_pointwise_0_output_quant[256][28][28];
float layer_5_conv3x3_0_output[256][28][28];
float layer_5_conv3x3_0_output_quant[256][28][28];
float layer_5_pointwise_0_output[256][28][28];

float layer_5_pointwise_0_output_quant[256][28][28];
float layer_6_conv3x3_0_output[256][14][14];
float layer_5_pointwise_0_output_avgpool[256][14][14];
float layer_6_conv3x3_0_output_quant[256][14][14];
float layer_6_pointwise_0_output[512][14][14];
float layer_6_conv3x3_0_output_concat[512][14][14];

float layer_6_pointwise_0_output_quant[512][14][14];
float layer_7_conv3x3_0_output[512][14][14];
float layer_7_conv3x3_0_output_quant[512][14][14];
float layer_7_pointwise_0_output[512][14][14];

float layer_7_pointwise_0_output_quant[512][14][14];
float layer_8_conv3x3_0_output[512][14][14];
float layer_8_conv3x3_0_output_quant[512][14][14];
float layer_8_pointwise_0_output[512][14][14];

float layer_8_pointwise_0_output_quant[512][14][14];
float layer_9_conv3x3_0_output[512][14][14];
float layer_9_conv3x3_0_output_quant[512][14][14];
float layer_9_pointwise_0_output[512][14][14];

float layer_9_pointwise_0_output_quant[512][14][14];
float layer_10_conv3x3_0_output[512][14][14];
float layer_10_conv3x3_0_output_quant[512][14][14];
float layer_10_pointwise_0_output[512][14][14];

float layer_10_pointwise_0_output_quant[512][14][14];
float layer_11_conv3x3_0_output[512][14][14];
float layer_11_conv3x3_0_output_quant[512][14][14];
float layer_11_pointwise_0_output[512][14][14];

float layer_11_pointwise_0_output_quant[512][14][14];
float layer_12_conv3x3_0_output[512][7][7];
float layer_11_pointwise_0_output_avgpool[512][7][7];
float layer_12_conv3x3_0_output_quant[512][7][7];
float layer_12_pointwise_0_output[1024][7][7];
float layer_12_conv3x3_0_output_concat[1024][7][7];

float layer_12_pointwise_0_output_quant[1024][7][7];
float layer_13_conv3x3_0_output[1024][7][7];
float layer_13_conv3x3_0_output_quant[1024][7][7];
float layer_13_pointwise_0_output[1024][7][7];

float avgpool_out_sw[1024];

float linear_out_sw[1000];



/////////////////////
// lump sum buffers
/////////////////////
unsigned char images[NUM_TESTS*96*224*224];
float conv3x3_all[25141248];
float conv1x1_all[3139584];
float other_weights[1140014];
int ptr[3] = {0, 0, 6}; // {weight_3x3_ptr, weight_1x1_ptr, weight_other_ptr (first six are mean and std)}

uint512 conv3x3_all_hw[5456][3][3];
uint512 conv1x1_all_hw[6132];
uint512 weights_all_hw[2730];
uint512 conv3x3_all_hw_new_3d[5456][3][3];
uint512 conv3x3_all_hw_new[49104];
uint512 conv1x1_all_hw_new[6132];

void read_all_images()
{
	std::ifstream ifs_param0("conv1_input.bin", std::ios::in | std::ios::binary);
	ifs_param0.read((char*)(images), NUM_TESTS*96*224*224*sizeof(unsigned char));
	ifs_param0.close();
}

void get_image(unsigned char *images, unsigned int idx, unsigned char image[96][224][224])
{
	unsigned int offset = idx*96*224*224;
	for (int c = 0; c < 96; c ++) {
		for (int row = 0; row < 224; row ++) {
			for (int col = 0; col < 224; col ++) {
				image[c][row][col] = images[offset + c*224*224 + row*224 + col];
			}
		}
	}
}

void load_weights()
{
	std::ifstream ifs_param0("conv3x3_weights.bin", std::ios::in | std::ios::binary);
	ifs_param0.read((char*)(conv3x3_all), 25141248*sizeof(float));
	ifs_param0.close();
	std::ifstream ifs_param1("conv1x1_weights.bin", std::ios::in | std::ios::binary);
	ifs_param1.read((char*)(conv1x1_all), 3139584*sizeof(float));
	ifs_param1.close();
	std::ifstream ifs_param2("other_weights.bin", std::ios::in | std::ios::binary);
	ifs_param2.read((char*)(other_weights), 1112366*sizeof(float));
	ifs_param2.close();
}

template <int C_OUT, int C_IN>
void get_weight_3x3(float weight_3x3[C_OUT][C_IN][3][3])
{
	for (int co = 0; co < C_OUT; co ++) {
		for (int ci = 0; ci < C_IN; ci ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					int index = co*C_IN*3*3 + ci*3*3 + row*3 + col;
					weight_3x3[co][ci][row][col] = conv3x3_all[ptr[0] + index];
				}
			}
		}
	}
	ptr[0] += C_OUT * C_IN * 3 * 3;
}

template <int C_OUT, int C_IN>
void get_weight_1x1(float weight_1x1[C_OUT][C_IN])
{
	for (int co = 0; co < C_OUT; co ++) {
		for (int ci = 0; ci < C_IN; ci ++) {
			int index = co*C_IN + ci;
			weight_1x1[co][ci] = conv1x1_all[ptr[1] + index];
		}
	}
	ptr[1] += C_OUT * C_IN;
}

template <int CH>
void get_weight_other(float weight_other[CH])
{
	for (int ch = 0; ch < CH; ch ++) {

		FIX_1D_PACK weight = other_weights[ptr[2] + ch];
		weight_other[ch] = (float)weight;
	}
	ptr[2] += CH;
}

template <int C_OUT, int C_IN>
int reorder_weights3x3(int conv3x3_ptr)
{
	float weight3x3[C_OUT][C_IN][3][3] = {};
	float weight3x3_flip[C_OUT*C_IN][3][3] = {};

	for (int c_out = 0; c_out < C_OUT; c_out ++) {
		for (int c_in = 0; c_in < C_IN; c_in ++) {
			for (int row = 0; row < 3; row ++) {
				for (int col = 0; col < 3; col ++) {
					int index = c_out*C_IN*3*3 + c_in*3*3 + row*3 + col;
					weight3x3[c_out][c_in][row][col] = conv3x3_all[conv3x3_ptr + index];
				}
			}
		}
	}
	for (int c_out = 0; c_out < C_OUT/32; c_out ++){
		for (int out_t = 0; out_t < 32; out_t ++){
			for (int c_in = 0; c_in < C_IN/32; c_in ++){
				for (int in_t = 0; in_t < 32; in_t ++){
					for (int row = 0; row < 3; row ++){
						for (int col = 0; col < 3; col ++){
							int flip_index = c_out*32*C_IN + c_in*32*32 + out_t*32 + in_t;
							weight3x3_flip[flip_index][row][col] = weight3x3[c_out*32+out_t][c_in*32+in_t][row][col];
						}
					}
				}
			}
		}
	}
	for (int i = 0; i < C_OUT*C_IN/512; i ++){
		for (int ch = 0; ch < 512; ch ++) {
			for (int row = 0; row < 3; row ++){
				for (int col = 0; col < 3; col ++){
					int index = i*512 + ch;
					float w = weight3x3_flip[index][row][col];
					if (w > 0) {
						conv3x3_all_hw_new_3d[conv3x3_ptr/9/512+i][row][col][ch] = 1;
					} else {
						conv3x3_all_hw_new_3d[conv3x3_ptr/9/512+i][row][col][ch] = 0;
					}
				}
			}
		}
	}

	return conv3x3_ptr + C_OUT*C_IN*3*3;
}

template <int C_OUT, int C_IN>
int reorder_weights1x1(int conv1x1_ptr)
{
	float weight1x1[C_OUT][C_IN] = {};
	float weight1x1_flip[C_OUT*C_IN] = {};

	for (int c_out = 0; c_out < C_OUT; c_out ++) {
		for (int c_in = 0; c_in < C_IN; c_in ++) {
			int index = c_out*C_IN + c_in;
			weight1x1[c_out][c_in] = conv1x1_all[conv1x1_ptr + index];
		}
	}
	for (int c_out = 0; c_out < C_OUT/32; c_out ++){
		for (int out_t = 0; out_t < 32; out_t ++){
			for (int c_in = 0; c_in < C_IN/32; c_in ++){
				for (int in_t = 0; in_t < 32; in_t ++){
					int flip_index = c_out*32*C_IN + c_in*32*32 + out_t*32 + in_t;
					weight1x1_flip[flip_index] = weight1x1[c_out*32+out_t][c_in*32+in_t];

				}
			}
		}
	}
	for (int i = 0; i < C_OUT*C_IN/512; i ++){
		for (int ch = 0; ch < 512; ch ++) {
			int index = i*512 + ch;
			float w = weight1x1_flip[index];
			if (w > 0) {
				conv1x1_all_hw_new[conv1x1_ptr/512+i][ch] = 1;
			} else {
				conv1x1_all_hw_new[conv1x1_ptr/512+i][ch] = 0;
			}
		}
	}
	return conv1x1_ptr + C_OUT*C_IN;
}

void load_individual_weight()
{
	// conv1 + bn1
	get_weight_3x3<32, 96>(layer_0_0_weight);
	get_weight_other<32>(layer_0_1_weight);
	get_weight_other<32>(layer_0_1_bias);

	// layer 1
	get_weight_3x3<32, 32>(layer_1_conv3x3_0_weight);
	get_weight_other<32>(layer_1_conv3x3_0_threshold);
	get_weight_other<32>(layer_1_conv3x3_1_weight);
	get_weight_other<32>(layer_1_conv3x3_1_bias);
	get_weight_1x1<64, 32>(layer_1_pointwise_0_weight);
	get_weight_other<64>(layer_1_pointwise_0_threshold);
	get_weight_other<64>(layer_1_pointwise_1_weight);
	get_weight_other<64>(layer_1_pointwise_1_bias);
	get_weight_other<32>(layer_1_rprelu1_shift_x_bias);
	get_weight_other<32>(layer_1_rprelu1_shift_y_bias);
	get_weight_other<32>(layer_1_rprelu1_prelu_weight);
	get_weight_other<64>(layer_1_rprelu2_shift_x_bias);
	get_weight_other<64>(layer_1_rprelu2_shift_y_bias);
	get_weight_other<64>(layer_1_rprelu2_prelu_weight);
	get_weight_other<32>(layer_1_shiftbn1_weight);
	get_weight_other<32>(layer_1_shiftbn1_bias);
	get_weight_other<64>(layer_1_shiftbn2_weight);
	get_weight_other<64>(layer_1_shiftbn2_bias);
	cout << (ptr[2]-6)/32 << endl;

	// layer 2
	get_weight_3x3<64, 64>(layer_2_conv3x3_0_weight);
	get_weight_other<64>(layer_2_conv3x3_0_threshold);
	get_weight_other<64>(layer_2_conv3x3_1_weight);
	get_weight_other<64>(layer_2_conv3x3_1_bias);
	get_weight_1x1<128, 64>(layer_2_pointwise_0_weight);
	get_weight_other<128>(layer_2_pointwise_0_threshold);
	get_weight_other<128>(layer_2_pointwise_1_weight);
	get_weight_other<128>(layer_2_pointwise_1_bias);
	get_weight_other<64>(layer_2_rprelu1_shift_x_bias);
	get_weight_other<64>(layer_2_rprelu1_shift_y_bias);
	get_weight_other<64>(layer_2_rprelu1_prelu_weight);
	get_weight_other<128>(layer_2_rprelu2_shift_x_bias);
	get_weight_other<128>(layer_2_rprelu2_shift_y_bias);
	get_weight_other<128>(layer_2_rprelu2_prelu_weight);
	get_weight_other<64>(layer_2_shiftbn1_weight);
	get_weight_other<64>(layer_2_shiftbn1_bias);
	get_weight_other<128>(layer_2_shiftbn2_weight);
	get_weight_other<128>(layer_2_shiftbn2_bias);
	cout << (ptr[2]-6)/32 << endl;

	// layer 3
	get_weight_3x3<128, 128>(layer_3_conv3x3_0_weight);
	get_weight_other<128>(layer_3_conv3x3_0_threshold);
	get_weight_other<128>(layer_3_conv3x3_1_weight);
	get_weight_other<128>(layer_3_conv3x3_1_bias);
	get_weight_1x1<128, 128>(layer_3_pointwise_0_weight);
	get_weight_other<128>(layer_3_pointwise_0_threshold);
	get_weight_other<128>(layer_3_pointwise_1_weight);
	get_weight_other<128>(layer_3_pointwise_1_bias);
	get_weight_other<128>(layer_3_rprelu1_shift_x_bias);
	get_weight_other<128>(layer_3_rprelu1_shift_y_bias);
	get_weight_other<128>(layer_3_rprelu1_prelu_weight);
	get_weight_other<128>(layer_3_rprelu2_shift_x_bias);
	get_weight_other<128>(layer_3_rprelu2_shift_y_bias);
	get_weight_other<128>(layer_3_rprelu2_prelu_weight);
	get_weight_other<128>(layer_3_shiftbn1_weight);
	get_weight_other<128>(layer_3_shiftbn1_bias);
	get_weight_other<128>(layer_3_shiftbn2_weight);
	get_weight_other<128>(layer_3_shiftbn2_bias);

	// layer 4
	get_weight_3x3<128, 128>(layer_4_conv3x3_0_weight);
	get_weight_other<128>(layer_4_conv3x3_0_threshold);
	get_weight_other<128>(layer_4_conv3x3_1_weight);
	get_weight_other<128>(layer_4_conv3x3_1_bias);
	get_weight_1x1<256, 128>(layer_4_pointwise_0_weight);
	get_weight_other<256>(layer_4_pointwise_0_threshold);
	get_weight_other<256>(layer_4_pointwise_1_weight);
	get_weight_other<256>(layer_4_pointwise_1_bias);
	get_weight_other<128>(layer_4_rprelu1_shift_x_bias);
	get_weight_other<128>(layer_4_rprelu1_shift_y_bias);
	get_weight_other<128>(layer_4_rprelu1_prelu_weight);
	get_weight_other<256>(layer_4_rprelu2_shift_x_bias);
	get_weight_other<256>(layer_4_rprelu2_shift_y_bias);
	get_weight_other<256>(layer_4_rprelu2_prelu_weight);
	get_weight_other<128>(layer_4_shiftbn1_weight);
	get_weight_other<128>(layer_4_shiftbn1_bias);
	get_weight_other<256>(layer_4_shiftbn2_weight);
	get_weight_other<256>(layer_4_shiftbn2_bias);

	// layer 5
	get_weight_3x3<256, 256>(layer_5_conv3x3_0_weight);
	get_weight_other<256>(layer_5_conv3x3_0_threshold);
	get_weight_other<256>(layer_5_conv3x3_1_weight);
	get_weight_other<256>(layer_5_conv3x3_1_bias);
	get_weight_1x1<256, 256>(layer_5_pointwise_0_weight);
	get_weight_other<256>(layer_5_pointwise_0_threshold);
	get_weight_other<256>(layer_5_pointwise_1_weight);
	get_weight_other<256>(layer_5_pointwise_1_bias);
	get_weight_other<256>(layer_5_rprelu1_shift_x_bias);
	get_weight_other<256>(layer_5_rprelu1_shift_y_bias);
	get_weight_other<256>(layer_5_rprelu1_prelu_weight);
	get_weight_other<256>(layer_5_rprelu2_shift_x_bias);
	get_weight_other<256>(layer_5_rprelu2_shift_y_bias);
	get_weight_other<256>(layer_5_rprelu2_prelu_weight);
	get_weight_other<256>(layer_5_shiftbn1_weight);
	get_weight_other<256>(layer_5_shiftbn1_bias);
	get_weight_other<256>(layer_5_shiftbn2_weight);
	get_weight_other<256>(layer_5_shiftbn2_bias);

	// layer 6
	get_weight_3x3<256, 256>(layer_6_conv3x3_0_weight);
	get_weight_other<256>(layer_6_conv3x3_0_threshold);
	get_weight_other<256>(layer_6_conv3x3_1_weight);
	get_weight_other<256>(layer_6_conv3x3_1_bias);
	get_weight_1x1<512, 256>(layer_6_pointwise_0_weight);
	get_weight_other<512>(layer_6_pointwise_0_threshold);
	get_weight_other<512>(layer_6_pointwise_1_weight);
	get_weight_other<512>(layer_6_pointwise_1_bias);
	get_weight_other<256>(layer_6_rprelu1_shift_x_bias);
	get_weight_other<256>(layer_6_rprelu1_shift_y_bias);
	get_weight_other<256>(layer_6_rprelu1_prelu_weight);
	get_weight_other<512>(layer_6_rprelu2_shift_x_bias);
	get_weight_other<512>(layer_6_rprelu2_shift_y_bias);
	get_weight_other<512>(layer_6_rprelu2_prelu_weight);
	get_weight_other<256>(layer_6_shiftbn1_weight);
	get_weight_other<256>(layer_6_shiftbn1_bias);
	get_weight_other<512>(layer_6_shiftbn2_weight);
	get_weight_other<512>(layer_6_shiftbn2_bias);

	// layer 7
	get_weight_3x3<512, 512>(layer_7_conv3x3_0_weight);
	get_weight_other<512>(layer_7_conv3x3_0_threshold);
	get_weight_other<512>(layer_7_conv3x3_1_weight);
	get_weight_other<512>(layer_7_conv3x3_1_bias);
	get_weight_1x1<512, 512>(layer_7_pointwise_0_weight);
	get_weight_other<512>(layer_7_pointwise_0_threshold);
	get_weight_other<512>(layer_7_pointwise_1_weight);
	get_weight_other<512>(layer_7_pointwise_1_bias);
	get_weight_other<512>(layer_7_rprelu1_shift_x_bias);
	get_weight_other<512>(layer_7_rprelu1_shift_y_bias);
	get_weight_other<512>(layer_7_rprelu1_prelu_weight);
	get_weight_other<512>(layer_7_rprelu2_shift_x_bias);
	get_weight_other<512>(layer_7_rprelu2_shift_y_bias);
	get_weight_other<512>(layer_7_rprelu2_prelu_weight);
	get_weight_other<512>(layer_7_shiftbn1_weight);
	get_weight_other<512>(layer_7_shiftbn1_bias);
	get_weight_other<512>(layer_7_shiftbn2_weight);
	get_weight_other<512>(layer_7_shiftbn2_bias);

	// layer 8
	get_weight_3x3<512, 512>(layer_8_conv3x3_0_weight);
	get_weight_other<512>(layer_8_conv3x3_0_threshold);
	get_weight_other<512>(layer_8_conv3x3_1_weight);
	get_weight_other<512>(layer_8_conv3x3_1_bias);
	get_weight_1x1<512, 512>(layer_8_pointwise_0_weight);
	get_weight_other<512>(layer_8_pointwise_0_threshold);
	get_weight_other<512>(layer_8_pointwise_1_weight);
	get_weight_other<512>(layer_8_pointwise_1_bias);
	get_weight_other<512>(layer_8_rprelu1_shift_x_bias);
	get_weight_other<512>(layer_8_rprelu1_shift_y_bias);
	get_weight_other<512>(layer_8_rprelu1_prelu_weight);
	get_weight_other<512>(layer_8_rprelu2_shift_x_bias);
	get_weight_other<512>(layer_8_rprelu2_shift_y_bias);
	get_weight_other<512>(layer_8_rprelu2_prelu_weight);
	get_weight_other<512>(layer_8_shiftbn1_weight);
	get_weight_other<512>(layer_8_shiftbn1_bias);
	get_weight_other<512>(layer_8_shiftbn2_weight);
	get_weight_other<512>(layer_8_shiftbn2_bias);

	// layer 9
	get_weight_3x3<512, 512>(layer_9_conv3x3_0_weight);
	get_weight_other<512>(layer_9_conv3x3_0_threshold);
	get_weight_other<512>(layer_9_conv3x3_1_weight);
	get_weight_other<512>(layer_9_conv3x3_1_bias);
	get_weight_1x1<512, 512>(layer_9_pointwise_0_weight);
	get_weight_other<512>(layer_9_pointwise_0_threshold);
	get_weight_other<512>(layer_9_pointwise_1_weight);
	get_weight_other<512>(layer_9_pointwise_1_bias);
	get_weight_other<512>(layer_9_rprelu1_shift_x_bias);
	get_weight_other<512>(layer_9_rprelu1_shift_y_bias);
	get_weight_other<512>(layer_9_rprelu1_prelu_weight);
	get_weight_other<512>(layer_9_rprelu2_shift_x_bias);
	get_weight_other<512>(layer_9_rprelu2_shift_y_bias);
	get_weight_other<512>(layer_9_rprelu2_prelu_weight);
	get_weight_other<512>(layer_9_shiftbn1_weight);
	get_weight_other<512>(layer_9_shiftbn1_bias);
	get_weight_other<512>(layer_9_shiftbn2_weight);
	get_weight_other<512>(layer_9_shiftbn2_bias);

	// layer 10
	get_weight_3x3<512, 512>(layer_10_conv3x3_0_weight);
	get_weight_other<512>(layer_10_conv3x3_0_threshold);
	get_weight_other<512>(layer_10_conv3x3_1_weight);
	get_weight_other<512>(layer_10_conv3x3_1_bias);
	get_weight_1x1<512, 512>(layer_10_pointwise_0_weight);
	get_weight_other<512>(layer_10_pointwise_0_threshold);
	get_weight_other<512>(layer_10_pointwise_1_weight);
	get_weight_other<512>(layer_10_pointwise_1_bias);
	get_weight_other<512>(layer_10_rprelu1_shift_x_bias);
	get_weight_other<512>(layer_10_rprelu1_shift_y_bias);
	get_weight_other<512>(layer_10_rprelu1_prelu_weight);
	get_weight_other<512>(layer_10_rprelu2_shift_x_bias);
	get_weight_other<512>(layer_10_rprelu2_shift_y_bias);
	get_weight_other<512>(layer_10_rprelu2_prelu_weight);
	get_weight_other<512>(layer_10_shiftbn1_weight);
	get_weight_other<512>(layer_10_shiftbn1_bias);
	get_weight_other<512>(layer_10_shiftbn2_weight);
	get_weight_other<512>(layer_10_shiftbn2_bias);

	// layer 11
	get_weight_3x3<512, 512>(layer_11_conv3x3_0_weight);
	get_weight_other<512>(layer_11_conv3x3_0_threshold);
	get_weight_other<512>(layer_11_conv3x3_1_weight);
	get_weight_other<512>(layer_11_conv3x3_1_bias);
	get_weight_1x1<512, 512>(layer_11_pointwise_0_weight);
	get_weight_other<512>(layer_11_pointwise_0_threshold);
	get_weight_other<512>(layer_11_pointwise_1_weight);
	get_weight_other<512>(layer_11_pointwise_1_bias);
	get_weight_other<512>(layer_11_rprelu1_shift_x_bias);
	get_weight_other<512>(layer_11_rprelu1_shift_y_bias);
	get_weight_other<512>(layer_11_rprelu1_prelu_weight);
	get_weight_other<512>(layer_11_rprelu2_shift_x_bias);
	get_weight_other<512>(layer_11_rprelu2_shift_y_bias);
	get_weight_other<512>(layer_11_rprelu2_prelu_weight);
	get_weight_other<512>(layer_11_shiftbn1_weight);
	get_weight_other<512>(layer_11_shiftbn1_bias);
	get_weight_other<512>(layer_11_shiftbn2_weight);
	get_weight_other<512>(layer_11_shiftbn2_bias);

	// layer 12
	get_weight_3x3<512, 512>(layer_12_conv3x3_0_weight);
	get_weight_other<512>(layer_12_conv3x3_0_threshold);
	get_weight_other<512>(layer_12_conv3x3_1_weight);
	get_weight_other<512>(layer_12_conv3x3_1_bias);
	get_weight_1x1<1024, 512>(layer_12_pointwise_0_weight);
	get_weight_other<1024>(layer_12_pointwise_0_threshold);
	get_weight_other<1024>(layer_12_pointwise_1_weight);
	get_weight_other<1024>(layer_12_pointwise_1_bias);
	get_weight_other<512>(layer_12_rprelu1_shift_x_bias);
	get_weight_other<512>(layer_12_rprelu1_shift_y_bias);
	get_weight_other<512>(layer_12_rprelu1_prelu_weight);
	get_weight_other<1024>(layer_12_rprelu2_shift_x_bias);
	get_weight_other<1024>(layer_12_rprelu2_shift_y_bias);
	get_weight_other<1024>(layer_12_rprelu2_prelu_weight);
	get_weight_other<512>(layer_12_shiftbn1_weight);
	get_weight_other<512>(layer_12_shiftbn1_bias);
	get_weight_other<1024>(layer_12_shiftbn2_weight);
	get_weight_other<1024>(layer_12_shiftbn2_bias);

	// layer 13
	get_weight_3x3<1024, 1024>(layer_13_conv3x3_0_weight);
	get_weight_other<1024>(layer_13_conv3x3_0_threshold);
	get_weight_other<1024>(layer_13_conv3x3_1_weight);
	get_weight_other<1024>(layer_13_conv3x3_1_bias);
	get_weight_1x1<1024, 1024>(layer_13_pointwise_0_weight);
	get_weight_other<1024>(layer_13_pointwise_0_threshold);
	get_weight_other<1024>(layer_13_pointwise_1_weight);
	get_weight_other<1024>(layer_13_pointwise_1_bias);
	get_weight_other<1024>(layer_13_rprelu1_shift_x_bias);
	get_weight_other<1024>(layer_13_rprelu1_shift_y_bias);
	get_weight_other<1024>(layer_13_rprelu1_prelu_weight);
	get_weight_other<1024>(layer_13_rprelu2_shift_x_bias);
	get_weight_other<1024>(layer_13_rprelu2_shift_y_bias);
	get_weight_other<1024>(layer_13_rprelu2_prelu_weight);
	get_weight_other<1024>(layer_13_shiftbn1_weight);
	get_weight_other<1024>(layer_13_shiftbn1_bias);
	get_weight_other<1024>(layer_13_shiftbn2_weight);
	get_weight_other<1024>(layer_13_shiftbn2_bias);
	cout << (ptr[2]-6)/32 << endl;

	for (int i = 0; i < 1000; i ++){
		for (int j = 0; j < 1024; j ++){
			fc_weight[i][j] = other_weights[ptr[2]];
			ptr[2] += 1;
		}
	}
	cout << (ptr[2]-6)/32 << endl;
	for (int i = 0; i < 1000; i ++){
		fc_bias[i] = other_weights[ptr[2]];
		ptr[2] += 1;
	}
	cout << (ptr[2]-6)/32 << endl;



	// Re-order weights
	int weight3x3_ptr = 0;
	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<32, 96>(weight3x3_ptr); //conv1
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<32, 32>(weight3x3_ptr); //layer1
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<64, 64>(weight3x3_ptr); //layer2
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<128, 128>(weight3x3_ptr); //layer3
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<128, 128>(weight3x3_ptr); //layer4
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<256, 256>(weight3x3_ptr); //layer5
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<256, 256>(weight3x3_ptr); //layer6
    //	cout << "LAYER 7 weight3x3_ptr " << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<512, 512>(weight3x3_ptr); //layer7
    //	cout << "LAYER 8 weight3x3_ptr " << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<512, 512>(weight3x3_ptr); //layer8
    //	cout << "LAYER 9 weight3x3_ptr " << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<512, 512>(weight3x3_ptr); //layer9
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<512, 512>(weight3x3_ptr); //layer10
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<512, 512>(weight3x3_ptr); //layer11
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<512, 512>(weight3x3_ptr); //layer12
	//	cout << weight3x3_ptr << endl;
	weight3x3_ptr = reorder_weights3x3<1024, 1024>(weight3x3_ptr); //layer13
	cout << weight3x3_ptr << endl;
	cout << "WEIGHT3x3 LOADED!!!" << endl;


	for (int ch = 0; ch < 5456; ch ++) {
		for (int row = 0; row < 3; row ++) {
			for (int col = 0; col < 3; col ++){
				int index = ch*3*3 + row*3 + col;
				conv3x3_all_hw_new[index] = conv3x3_all_hw_new_3d[ch][row][col];
			}
		}
	}


	int weight1x1_ptr = 0;
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<64, 32>(weight1x1_ptr); //layer1
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<128, 64>(weight1x1_ptr); //layer2
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<128, 128>(weight1x1_ptr); //layer3
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<256, 128>(weight1x1_ptr); //layer4
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<256, 256>(weight1x1_ptr); //layer5
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<512, 256>(weight1x1_ptr); //layer6
    //	cout << "LAYER 7 weight1x1_ptr " << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<512, 512>(weight1x1_ptr); //layer7
    //	cout << "LAYER 8 weight1x1_ptr " << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<512, 512>(weight1x1_ptr); //layer8
    //	cout << "LAYER 9 weight1x1_ptr " << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<512, 512>(weight1x1_ptr); //layer9
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<512, 512>(weight1x1_ptr); //layer10
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<512, 512>(weight1x1_ptr); //layer11
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<1024, 512>(weight1x1_ptr); //layer12
	//	cout << weight1x1_ptr << endl;
	weight1x1_ptr = reorder_weights1x1<1024, 1024>(weight1x1_ptr); //layer13
	cout << weight1x1_ptr << endl;
	cout << "WEIGHT1x1 LOADED!!!" << endl;


	/////////////////////////////////////////////////////////////////////////////
	////////////// THIS IS FOR 32 BIT WEIGHTS !!! DONT DELETE //////////////////
	///////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < 5460; i ++){
		for (int j = 0; j < 16; j ++){
			// CHANGE TYPEDEF USE 16 bits
			FIX_1D_PACK weight = other_weights[i*16 + j + 6];
			weights_all_hw[i].range(j*32 + 31, j*32) = weight.range(31, 0);
		}
	}

	FILE* pFile0 = fopen("conv3x3_weights_host.bin", "wb");
	fwrite(conv3x3_all_hw_new, sizeof(uint512), 49104, pFile0);
	fclose(pFile0);
	FILE* pFile1 = fopen("conv1x1_weights_host.bin", "wb");
	fwrite(conv1x1_all_hw_new, sizeof(uint512), 6132, pFile1);
	fclose(pFile1);
	FILE* pFile2 = fopen("other_weights_host.bin", "wb");
	fwrite(weights_all_hw, sizeof(uint512), 2730, pFile2);
	fclose(pFile2);

	cout << "WRITE ALL WEIGHT OK" << endl;

	uint32 image_hw_all[NUM_TESTS*3*224*224];
	for (int k = 0; k < NUM_TESTS; k ++){
		unsigned char image[96][224][224] = {0};
		get_image(images, k, image);

		for (int j = 0; j < 3; j ++){
			for (int row = 0; row < 224; row ++){
				for (int col = 0; col < 224; col ++){
					int index = k*3*224*224 + j*224*224 + row*224 + col;
					for (int b = 0; b < 32; b ++){
						if (image[j*32 + b][row][col] > 0) {
							image_hw_all[index][b] = 1;
						} else {
							image_hw_all[index][b] = 0;
						}
					}
				}
			}
		}
		FILE* pFile0 = fopen("image_hw_all_host.bin", "wb");
		fwrite(image_hw_all, sizeof(uint32), NUM_TESTS*3*224*224, pFile0);
		fclose(pFile0);

		cout << "WRITE IMAGE OK" << endl;
	}


}

void load_layer_output(std::string file_name, float* arr)
{
	std::ifstream in( file_name.c_str() );
	std::string line;

	int i = 0;

	while (std::getline(in, line))
	{
		float value;
		std::stringstream ss(line);

		while (ss >> value)
		{
			arr[i] = value;
			i++;
		}
	}
}



#endif
