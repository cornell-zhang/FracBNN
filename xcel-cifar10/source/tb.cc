#include "typedefs.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "bnn.h"
#include "weights_tb.h"
#include <fstream>
#include <hls_math.h>

using namespace std;

#define NUM_TESTS 3

unsigned char images[NUM_TESTS*96*32*32];
unsigned char labels[NUM_TESTS];

float conv1_out[16][32][32];
float bn1_out[16][32][32];
float layer1_0_binarize1_out[16][32][32];
float layer1_0_binarize2_out[16][32][32];
float layer1_0_pgconv1_out[16][32][32];
float layer1_0_pgconv2_out[16][32][32];
float layer1_0_bn1_out[16][32][32];
float layer1_0_bn2_out[16][32][32];
float layer1_0_bn3_out[16][32][32];
float layer1_0_bn4_out[16][32][32];
float layer1_0_rprelu1_out[16][32][32];
float layer1_0_rprelu2_out[16][32][32];
float layer1_0_shortcut1_out[16][32][32];
float layer1_0_shortcut2_out[16][32][32];

float layer1_1_binarize1_out[16][32][32];
float layer1_1_binarize2_out[16][32][32];
float layer1_1_pgconv1_out[16][32][32];
float layer1_1_pgconv2_out[16][32][32];
float layer1_1_bn1_out[16][32][32];
float layer1_1_bn2_out[16][32][32];
float layer1_1_bn3_out[16][32][32];
float layer1_1_bn4_out[16][32][32];
float layer1_1_rprelu1_out[16][32][32];
float layer1_1_rprelu2_out[16][32][32];
float layer1_1_shortcut1_out[16][32][32];
float layer1_1_shortcut2_out[16][32][32];

float layer1_2_binarize1_out[16][32][32];
float layer1_2_binarize2_out[16][32][32];
float layer1_2_pgconv1_out[16][32][32];
float layer1_2_pgconv2_out[16][32][32];
float layer1_2_bn1_out[16][32][32];
float layer1_2_bn2_out[16][32][32];
float layer1_2_bn3_out[16][32][32];
float layer1_2_bn4_out[16][32][32];
float layer1_2_rprelu1_out[16][32][32];
float layer1_2_rprelu2_out[16][32][32];
float layer1_2_shortcut1_out[16][32][32];
float layer1_2_shortcut2_out[16][32][32];

float layer2_0_binarize1_out[16][32][32];
float layer2_0_binarize2_out[32][16][16];
float layer2_0_pgconv1_out[32][16][16];
float layer2_0_pgconv2_out[32][16][16];
float layer2_0_bn1_out[32][16][16];
float layer2_0_bn2_out[32][16][16];
float layer2_0_bn3_out[32][16][16];
float layer2_0_bn4_out[32][16][16];
float layer2_0_rprelu1_out[32][16][16];
float layer2_0_rprelu2_out[32][16][16];
float layer2_0_shortcut1_out[32][16][16];
float layer2_0_shortcut2_out[32][16][16];
float layer2_0_concat_out[32][16][16];

float layer2_1_binarize1_out[32][16][16];
float layer2_1_binarize2_out[32][16][16];
float layer2_1_pgconv1_out[32][16][16];
float layer2_1_pgconv2_out[32][16][16];
float layer2_1_bn1_out[32][16][16];
float layer2_1_bn2_out[32][16][16];
float layer2_1_bn3_out[32][16][16];
float layer2_1_bn4_out[32][16][16];
float layer2_1_rprelu1_out[32][16][16];
float layer2_1_rprelu2_out[32][16][16];
float layer2_1_shortcut1_out[32][16][16];
float layer2_1_shortcut2_out[32][16][16];

float layer2_2_binarize1_out[32][16][16];
float layer2_2_binarize2_out[32][16][16];
float layer2_2_pgconv1_out[32][16][16];
float layer2_2_pgconv2_out[32][16][16];
float layer2_2_bn1_out[32][16][16];
float layer2_2_bn2_out[32][16][16];
float layer2_2_bn3_out[32][16][16];
float layer2_2_bn4_out[32][16][16];
float layer2_2_rprelu1_out[32][16][16];
float layer2_2_rprelu2_out[32][16][16];
float layer2_2_shortcut1_out[32][16][16];
float layer2_2_shortcut2_out[32][16][16];

float layer3_0_binarize1_out[32][16][16];
float layer3_0_binarize2_out[64][8][8];
float layer3_0_pgconv1_out[64][8][8];
float layer3_0_pgconv2_out[64][8][8];
float layer3_0_bn1_out[64][8][8];
float layer3_0_bn2_out[64][8][8];
float layer3_0_bn3_out[64][8][8];
float layer3_0_bn4_out[64][8][8];
float layer3_0_rprelu1_out[64][8][8];
float layer3_0_rprelu2_out[64][8][8];
float layer3_0_shortcut1_out[64][8][8];
float layer3_0_shortcut2_out[64][8][8];
float layer3_0_concat_out[64][8][8];

float layer3_1_binarize1_out[64][8][8];
float layer3_1_binarize2_out[64][8][8];
float layer3_1_pgconv1_out[64][8][8];
float layer3_1_pgconv2_out[64][8][8];
float layer3_1_bn1_out[64][8][8];
float layer3_1_bn2_out[64][8][8];
float layer3_1_bn3_out[64][8][8];
float layer3_1_bn4_out[64][8][8];
float layer3_1_rprelu1_out[64][8][8];
float layer3_1_rprelu2_out[64][8][8];
float layer3_1_shortcut1_out[64][8][8];
float layer3_1_shortcut2_out[64][8][8];

float layer3_2_binarize1_out[64][8][8];
float layer3_2_binarize2_out[64][8][8];
float layer3_2_pgconv1_out[64][8][8];
float layer3_2_pgconv2_out[64][8][8];
float layer3_2_bn1_out[64][8][8];
float layer3_2_bn2_out[64][8][8];
float layer3_2_bn3_out[64][8][8];
float layer3_2_bn4_out[64][8][8];
float layer3_2_rprelu1_out[64][8][8];
float layer3_2_rprelu2_out[64][8][8];
float layer3_2_shortcut1_out[64][8][8];
float layer3_2_shortcut2_out[64][8][8];

float avg_pool_out[64];
float classifier_out[10];



// Convolution Layer 1 (binary)
void conv1(float input[96][32][32], const float weights[16][96][3][3], float output[16][32][32])
{
	for (int c_out = 0; c_out < 16; c_out ++) {
		for (int row = 0; row < 32; row ++) {
			for (int col = 0; col < 32; col ++) {
				int accum = 0;
				for (int c_in = 0; c_in < 96; c_in ++) {
					for (int k_row = 0; k_row < 3; k_row ++) {
						for (int k_col = 0; k_col < 3; k_col ++) {
							int row_in = row + k_row - 1;
							int col_in = col + k_col - 1;
							if (row_in >= 0 && row_in < 32 && col_in >= 0 && col_in < 32) {
								int inp = input[c_in][row_in][col_in];
								float wei = weights[c_out][c_in][k_row][k_col];
								if ((inp == 1 && wei > 0) || (inp == 0 && wei <= 0)) {
									accum += 1;
								} else {
									accum -= 1;
								}
							}
						}
					}
				}
				output[c_out][row][col] = accum;
			}
		}
	}
}

template <int CHANNEL, int HEIGHT, int WIDTH>
void bn(float input[CHANNEL][HEIGHT][WIDTH], const float weight[CHANNEL], const float bias[CHANNEL], float output[CHANNEL][HEIGHT][WIDTH]){
	for (int c = 0; c < CHANNEL; c ++) {
		for (int row = 0; row < HEIGHT; row ++) {
			for (int col = 0; col < WIDTH; col ++) {
				output[c][row][col] = input[c][row][col]*weight[c] + bias[c];
			}
		}
	}
}

template <int CHANNEL, int HEIGHT, int WIDTH>
void quant_sign(float input[CHANNEL][HEIGHT][WIDTH], float output[CHANNEL][HEIGHT][WIDTH]){
	for (int c = 0; c < CHANNEL; c ++) {
		for (int row = 0; row < HEIGHT; row ++) {
			for (int col = 0; col < WIDTH; col ++) {
				float x = input[c][row][col];
				float o = -1.0;
				if (x > 2.0/3) {
					o = 1.0;
				} else if (x <= 2.0/3 && x >= 0) {
					o = 1.0/3;
				} else if (x < 0 && x >= -2.0/3) {
					o = -1.0/3;
				} else {
					o = -1.0;
				}
				output[c][row][col] = o;
			}
		}
	}
}

template <int CHANNEL, int HEIGHT, int WIDTH>
void truncate(float input[CHANNEL][HEIGHT][WIDTH], float output[CHANNEL][HEIGHT][WIDTH]){
	for (int c = 0; c < CHANNEL; c ++) {
		for (int row = 0; row < HEIGHT; row ++) {
			for (int col = 0; col < WIDTH; col ++) {
				float x = input[c][row][col];
				float o = -1.0;
				if (x >= 0) {
					o = 1.0; //o = 1.0/3;
				} else {
					o = -1.0;
				}
				output[c][row][col] = o;
			}
		}
	}
}

template <int CHANNEL, int HEIGHT, int WIDTH>
void rprelu(float input[CHANNEL][HEIGHT][WIDTH],
		const float x_bias[CHANNEL],
		const float y_bias[CHANNEL],
		const float weight[CHANNEL],
		float output[CHANNEL][HEIGHT][WIDTH])
{
	for (int c = 0; c < CHANNEL; c ++) {
		for (int row = 0; row < HEIGHT; row ++) {
			for (int col = 0; col < WIDTH; col ++) {
				float t = input[c][row][col] + x_bias[c];
				if (t < 0) {
					t = weight[c]*t;
				}
				output[c][row][col] = t + y_bias[c];
			}
		}
	}
}

template <int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void avgpool_concat(float input[CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
		float output[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT])
{
	for (int c = 0; c < CHANNEL_IN; c ++) {
		for (int row = 0; row < HEIGHT_OUT; row ++) {
			for (int col = 0; col < WIDTH_OUT; col ++) {
				float m = 0;
				m += input[c][row*2 + 0][col*2 + 0];
				m += input[c][row*2 + 0][col*2 + 1];
				m += input[c][row*2 + 1][col*2 + 0];
				m += input[c][row*2 + 1][col*2 + 1];
				m = m/4.0;
				output[c][row][col] = m;
				output[c+CHANNEL_IN][row][col] = m;
			}
		}
	}
}

template <int CHANNEL, int HEIGHT, int WIDTH>
void shortcut(float input_a[CHANNEL][HEIGHT][WIDTH],
		float input_b[CHANNEL][HEIGHT][WIDTH],
		float output[CHANNEL][HEIGHT][WIDTH])
{
	for (int c = 0; c < CHANNEL; c ++) {
		for (int row = 0; row < HEIGHT; row ++) {
			for (int col = 0; col < WIDTH; col ++) {
				float x = input_a[c][row][col];
				float y = input_b[c][row][col];
				output[c][row][col] = x + y;
			}
		}
	}
}

template <int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void generic_conv(float input[CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
		const float weight[CHANNEL_OUT][CHANNEL_IN][3][3],
		float output[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT],
		int stride)
{
	for (int co = 0; co < CHANNEL_OUT; co ++) {
		for (int row = 0; row < HEIGHT_OUT; row ++) {
			for (int col = 0; col < WIDTH_OUT; col ++) {
				float accum = 0;
				for (int ci = 0; ci < CHANNEL_IN; ci ++) {
					for (int krow = 0; krow < 3; krow ++) {
						for (int kcol = 0; kcol < 3; kcol ++) {
							int row_in = row*stride + krow - 1;
							int col_in = col*stride + kcol - 1;
							if (row_in >= 0 && row_in < HEIGHT_IN && col_in >= 0 && col_in < WIDTH_IN) {
								accum += input[ci][row_in][col_in]*weight[co][ci][krow][kcol];
							}
						}
					}
				}
				output[co][row][col] = accum;
			}
		}
	}
}

float sigmoid(float x)
{
	return 1/(1 + exp(-x));
}

template <int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void layer1_pgconv(float input[CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
		const float weight[CHANNEL_OUT][CHANNEL_IN][3][3],
		const float threshold[CHANNEL_OUT],
		float output[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT])
{
	float input_truncate[CHANNEL_IN][HEIGHT_IN][WIDTH_IN];
	float out_msb[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT];
	float full_msb[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT];
	truncate<CHANNEL_IN, HEIGHT_IN, WIDTH_IN>(input, input_truncate);
	generic_conv<CHANNEL_IN, CHANNEL_OUT, HEIGHT_IN, WIDTH_IN, HEIGHT_OUT, WIDTH_OUT>(input_truncate, weight, out_msb, 1);
	generic_conv<CHANNEL_IN, CHANNEL_OUT, HEIGHT_IN, WIDTH_IN, HEIGHT_OUT, WIDTH_OUT>(input, weight, full_msb, 1);

	for (int c = 0; c < CHANNEL_OUT; c ++) {
		for (int row = 0; row < HEIGHT_OUT; row ++) {
			for (int col = 0; col < WIDTH_OUT; col ++) {
				float t = out_msb[c][row][col]*2.0/3.0 - threshold[c];
				float s = sigmoid(5*t);
				if (s > 0.5) {
					output[c][row][col] = full_msb[c][row][col];
				} else {
					output[c][row][col] = out_msb[c][row][col] * 2.0 / 3.0;
				}
			}
		}
	}
}

template <int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void layer2_pgconv(float input[CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
		const float weight[CHANNEL_OUT][CHANNEL_IN][3][3],
		const float threshold[CHANNEL_OUT],
		float output[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT])
{
	float input_truncate[CHANNEL_IN][HEIGHT_IN][HEIGHT_IN];
	float out_msb[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT];
	float full_msb[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT];
	int stride = 1;
	if (CHANNEL_OUT != CHANNEL_IN) {
		stride = 2;
	}
	truncate<CHANNEL_IN, HEIGHT_IN, WIDTH_IN>(input, input_truncate);
	generic_conv<CHANNEL_IN, CHANNEL_OUT, HEIGHT_IN, WIDTH_IN, HEIGHT_OUT, WIDTH_OUT>(input_truncate, weight, out_msb, stride);
	generic_conv<CHANNEL_IN, CHANNEL_OUT, HEIGHT_IN, WIDTH_IN, HEIGHT_OUT, WIDTH_OUT>(input, weight, full_msb, stride);
	for (int c = 0; c < CHANNEL_OUT; c ++) {
		for (int row = 0; row < HEIGHT_OUT; row ++) {
			for (int col = 0; col < WIDTH_OUT; col ++) {
				float t = out_msb[c][row][col]*2.0/3.0 - threshold[c];
				float s = sigmoid(5*t);
				if (s > 0.5) {
					output[c][row][col] = full_msb[c][row][col];
				} else {
					output[c][row][col] = out_msb[c][row][col]*2.0/3.0;
				}
			}
		}
	}
}

template <int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT_IN, int WIDTH_IN, int HEIGHT_OUT, int WIDTH_OUT>
void layer3_pgconv(float input[CHANNEL_IN][HEIGHT_IN][WIDTH_IN],
		const float weight[CHANNEL_OUT][CHANNEL_IN][3][3],
		const float threshold[CHANNEL_OUT],
		float output[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT])
{
	float input_truncate[CHANNEL_IN][HEIGHT_IN][HEIGHT_IN];
	float out_msb[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT];
	float full_msb[CHANNEL_OUT][HEIGHT_OUT][WIDTH_OUT];
	int stride = 1;
	if (CHANNEL_OUT != CHANNEL_IN) {
		stride = 2;
	}
	truncate<CHANNEL_IN, HEIGHT_IN, WIDTH_IN>(input, input_truncate);
	generic_conv<CHANNEL_IN, CHANNEL_OUT, HEIGHT_IN, WIDTH_IN, HEIGHT_OUT, WIDTH_OUT>(input_truncate, weight, out_msb, stride);
	generic_conv<CHANNEL_IN, CHANNEL_OUT, HEIGHT_IN, WIDTH_IN, HEIGHT_OUT, WIDTH_OUT>(input, weight, full_msb, stride);
	for (int c = 0; c < CHANNEL_OUT; c ++) {
		for (int row = 0; row < HEIGHT_OUT; row ++) {
			for (int col = 0; col < WIDTH_OUT; col ++) {
				float t = out_msb[c][row][col]*2.0/3.0 - threshold[c];
				float s = sigmoid(5*t);
				if (s > 0.5) {
					output[c][row][col] = full_msb[c][row][col];
				} else {
					output[c][row][col] = out_msb[c][row][col]*2.0/3.0;
				}
			}
		}
	}
}

void load_image()
{    
	std::ifstream ifs_param("conv1_input.bin", std::ios::in | std::ios::binary);
	ifs_param.read((char*)(images), sizeof(unsigned char)*96*NUM_TESTS*32*32);
	ifs_param.close();
}

void load_label()
{    
	std::ifstream ifs_param("labels.bin", std::ios::in | std::ios::binary);
	ifs_param.read((char*)(labels), sizeof(unsigned char)*NUM_TESTS);
	ifs_param.close();
}

void get_image(unsigned char *images, unsigned int idx, float image[96][32][32])
{
	unsigned int offset = idx*96*32*32;
	for (int c = 0; c < 96; c ++) {
		for (int row = 0; row < 32; row ++) {
			for (int col = 0; col < 32; col ++) {
				image[c][row][col] = images[offset + c*32*32 + row*32 + col];
			}
		}
	}
}

void FracNet_sw(float image[96][32][32]) {
	conv1(image, conv1_weight, conv1_out);
	bn<16, 32, 32>(conv1_out, bn1_weight, bn1_bias, bn1_out);

	////////////////////////////////////
	//////////// LAYER 1 ///////////////
	////////////////////////////////////

	quant_sign<16, 32, 32>(bn1_out, layer1_0_binarize1_out);
	layer1_pgconv<16, 16, 32, 32, 32, 32>(layer1_0_binarize1_out, layer1_0_conv1_weight, layer1_0_conv1_threshold, layer1_0_pgconv1_out);
	bn<16, 32, 32>(layer1_0_pgconv1_out, layer1_0_bn1_weight, layer1_0_bn1_bias, layer1_0_bn1_out);
	rprelu<16, 32, 32>(layer1_0_bn1_out, layer1_0_rprelu1_shift_x_bias, layer1_0_rprelu1_shift_y_bias, layer1_0_rprelu1_prelu_weight, layer1_0_rprelu1_out);
	shortcut<16, 32, 32>(layer1_0_rprelu1_out, bn1_out, layer1_0_shortcut1_out);
	bn<16, 32, 32>(layer1_0_shortcut1_out, layer1_0_bn3_weight, layer1_0_bn3_bias, layer1_0_bn3_out);

	quant_sign<16, 32, 32>(layer1_0_bn3_out, layer1_0_binarize2_out);
	layer1_pgconv<16, 16, 32, 32, 32, 32>(layer1_0_binarize2_out, layer1_0_conv2_weight, layer1_0_conv2_threshold, layer1_0_pgconv2_out);
	bn<16, 32, 32>(layer1_0_pgconv2_out, layer1_0_bn2_weight, layer1_0_bn2_bias, layer1_0_bn2_out);
	rprelu<16, 32, 32>(layer1_0_bn2_out, layer1_0_rprelu2_shift_x_bias, layer1_0_rprelu2_shift_y_bias, layer1_0_rprelu2_prelu_weight, layer1_0_rprelu2_out);
	shortcut<16, 32, 32>(layer1_0_rprelu2_out, layer1_0_bn3_out, layer1_0_shortcut2_out);
	bn<16, 32, 32>(layer1_0_shortcut2_out, layer1_0_bn4_weight, layer1_0_bn4_bias, layer1_0_bn4_out);

	quant_sign<16, 32, 32>(layer1_0_bn4_out, layer1_1_binarize1_out);
	layer1_pgconv<16, 16, 32, 32, 32, 32>(layer1_1_binarize1_out, layer1_1_conv1_weight, layer1_1_conv1_threshold, layer1_1_pgconv1_out);
	bn<16, 32, 32>(layer1_1_pgconv1_out, layer1_1_bn1_weight, layer1_1_bn1_bias, layer1_1_bn1_out);
	rprelu<16, 32, 32>(layer1_1_bn1_out, layer1_1_rprelu1_shift_x_bias, layer1_1_rprelu1_shift_y_bias, layer1_1_rprelu1_prelu_weight, layer1_1_rprelu1_out);
	shortcut<16, 32, 32>(layer1_1_rprelu1_out, layer1_0_bn4_out, layer1_1_shortcut1_out);
	bn<16, 32, 32>(layer1_1_shortcut1_out, layer1_1_bn3_weight, layer1_1_bn3_bias, layer1_1_bn3_out);

	quant_sign<16, 32, 32>(layer1_1_bn3_out, layer1_1_binarize2_out);
	layer1_pgconv<16, 16, 32, 32, 32, 32>(layer1_1_binarize2_out, layer1_1_conv2_weight, layer1_1_conv2_threshold, layer1_1_pgconv2_out);
	bn<16, 32, 32>(layer1_1_pgconv2_out, layer1_1_bn2_weight, layer1_1_bn2_bias, layer1_1_bn2_out);
	rprelu<16, 32, 32>(layer1_1_bn2_out, layer1_1_rprelu2_shift_x_bias, layer1_1_rprelu2_shift_y_bias, layer1_1_rprelu2_prelu_weight, layer1_1_rprelu2_out);
	shortcut<16, 32, 32>(layer1_1_rprelu2_out, layer1_1_bn3_out, layer1_1_shortcut2_out);
	bn<16, 32, 32>(layer1_1_shortcut2_out, layer1_1_bn4_weight, layer1_1_bn4_bias, layer1_1_bn4_out);

	quant_sign<16, 32, 32>(layer1_1_bn4_out, layer1_2_binarize1_out);
	layer1_pgconv<16, 16, 32, 32, 32, 32>(layer1_2_binarize1_out, layer1_2_conv1_weight, layer1_2_conv1_threshold, layer1_2_pgconv1_out);
	bn<16, 32, 32>(layer1_2_pgconv1_out, layer1_2_bn1_weight, layer1_2_bn1_bias, layer1_2_bn1_out);
	rprelu<16, 32, 32>(layer1_2_bn1_out, layer1_2_rprelu1_shift_x_bias, layer1_2_rprelu1_shift_y_bias, layer1_2_rprelu1_prelu_weight, layer1_2_rprelu1_out);
	shortcut<16, 32, 32>(layer1_2_rprelu1_out, layer1_1_bn4_out, layer1_2_shortcut1_out);
	bn<16, 32, 32>(layer1_2_shortcut1_out, layer1_2_bn3_weight, layer1_2_bn3_bias, layer1_2_bn3_out);

	quant_sign<16, 32, 32>(layer1_2_bn3_out, layer1_2_binarize2_out);
	layer1_pgconv<16, 16, 32, 32, 32, 32>(layer1_2_binarize2_out, layer1_2_conv2_weight, layer1_2_conv2_threshold, layer1_2_pgconv2_out);
	bn<16, 32, 32>(layer1_2_pgconv2_out, layer1_2_bn2_weight, layer1_2_bn2_bias, layer1_2_bn2_out);
	rprelu<16, 32, 32>(layer1_2_bn2_out, layer1_2_rprelu2_shift_x_bias, layer1_2_rprelu2_shift_y_bias, layer1_2_rprelu2_prelu_weight, layer1_2_rprelu2_out);
	shortcut<16, 32, 32>(layer1_2_rprelu2_out, layer1_2_bn3_out, layer1_2_shortcut2_out);
	bn<16, 32, 32>(layer1_2_shortcut2_out, layer1_2_bn4_weight, layer1_2_bn4_bias, layer1_2_bn4_out);


	////////////////////////////////////
	//////////// LAYER 2 ///////////////
	////////////////////////////////////

	quant_sign<16, 32, 32>(layer1_2_bn4_out, layer2_0_binarize1_out);
	layer2_pgconv<16, 32, 32, 32, 16, 16>(layer2_0_binarize1_out, layer2_0_conv1_weight, layer2_0_conv1_threshold, layer2_0_pgconv1_out);
	bn<32, 16, 16>(layer2_0_pgconv1_out, layer2_0_bn1_weight, layer2_0_bn1_bias, layer2_0_bn1_out);
	rprelu<32, 16, 16>(layer2_0_bn1_out, layer2_0_rprelu1_shift_x_bias, layer2_0_rprelu1_shift_y_bias, layer2_0_rprelu1_prelu_weight, layer2_0_rprelu1_out);
	avgpool_concat<16, 32, 32, 32, 16, 16>(layer1_2_bn4_out, layer2_0_concat_out);
	shortcut<32, 16, 16>(layer2_0_rprelu1_out, layer2_0_concat_out, layer2_0_shortcut1_out);
	bn<32, 16, 16>(layer2_0_shortcut1_out, layer2_0_bn3_weight, layer2_0_bn3_bias, layer2_0_bn3_out);

	quant_sign<32, 16, 16>(layer2_0_bn3_out, layer2_0_binarize2_out);
	layer2_pgconv<32, 32, 16, 16, 16, 16>(layer2_0_binarize2_out, layer2_0_conv2_weight, layer2_0_conv2_threshold, layer2_0_pgconv2_out);
	bn<32, 16, 16>(layer2_0_pgconv2_out, layer2_0_bn2_weight, layer2_0_bn2_bias, layer2_0_bn2_out);
	rprelu<32, 16, 16>(layer2_0_bn2_out, layer2_0_rprelu2_shift_x_bias, layer2_0_rprelu2_shift_y_bias, layer2_0_rprelu2_prelu_weight, layer2_0_rprelu2_out);
	shortcut<32, 16, 16>(layer2_0_rprelu2_out, layer2_0_bn3_out, layer2_0_shortcut2_out);
	bn<32, 16, 16>(layer2_0_shortcut2_out, layer2_0_bn4_weight, layer2_0_bn4_bias, layer2_0_bn4_out);

	quant_sign<32, 16, 16>(layer2_0_bn4_out, layer2_1_binarize1_out);
	layer2_pgconv<32, 32, 16, 16, 16, 16>(layer2_1_binarize1_out, layer2_1_conv1_weight, layer2_1_conv1_threshold, layer2_1_pgconv1_out);
	bn<32, 16, 16>(layer2_1_pgconv1_out, layer2_1_bn1_weight, layer2_1_bn1_bias, layer2_1_bn1_out);
	rprelu<32, 16, 16>(layer2_1_bn1_out, layer2_1_rprelu1_shift_x_bias, layer2_1_rprelu1_shift_y_bias, layer2_1_rprelu1_prelu_weight, layer2_1_rprelu1_out);
	shortcut<32, 16, 16>(layer2_1_rprelu1_out, layer2_0_bn4_out, layer2_1_shortcut1_out);
	bn<32, 16, 16>(layer2_1_shortcut1_out, layer2_1_bn3_weight, layer2_1_bn3_bias, layer2_1_bn3_out);

	quant_sign<32, 16, 16>(layer2_1_bn3_out, layer2_1_binarize2_out);
	layer2_pgconv<32, 32, 16, 16, 16, 16>(layer2_1_binarize2_out, layer2_1_conv2_weight, layer2_1_conv2_threshold, layer2_1_pgconv2_out);
	bn<32, 16, 16>(layer2_1_pgconv2_out, layer2_1_bn2_weight, layer2_1_bn2_bias, layer2_1_bn2_out);
	rprelu<32, 16, 16>(layer2_1_bn2_out, layer2_1_rprelu2_shift_x_bias, layer2_1_rprelu2_shift_y_bias, layer2_1_rprelu2_prelu_weight, layer2_1_rprelu2_out);
	shortcut<32, 16, 16>(layer2_1_rprelu2_out, layer2_1_bn3_out, layer2_1_shortcut2_out);
	bn<32, 16, 16>(layer2_1_shortcut2_out, layer2_1_bn4_weight, layer2_1_bn4_bias, layer2_1_bn4_out);

	quant_sign<32, 16, 16>(layer2_1_bn4_out, layer2_2_binarize1_out);
	layer2_pgconv<32, 32, 16, 16, 16, 16>(layer2_2_binarize1_out, layer2_2_conv1_weight, layer2_2_conv1_threshold, layer2_2_pgconv1_out);
	bn<32, 16, 16>(layer2_2_pgconv1_out, layer2_2_bn1_weight, layer2_2_bn1_bias, layer2_2_bn1_out);
	rprelu<32, 16, 16>(layer2_2_bn1_out, layer2_2_rprelu1_shift_x_bias, layer2_2_rprelu1_shift_y_bias, layer2_2_rprelu1_prelu_weight, layer2_2_rprelu1_out);
	shortcut<32, 16, 16>(layer2_2_rprelu1_out, layer2_1_bn4_out, layer2_2_shortcut1_out);
	bn<32, 16, 16>(layer2_2_shortcut1_out, layer2_2_bn3_weight, layer2_2_bn3_bias, layer2_2_bn3_out);

	quant_sign<32, 16, 16>(layer2_2_bn3_out, layer2_2_binarize2_out);
	layer2_pgconv<32, 32, 16, 16, 16, 16>(layer2_2_binarize2_out, layer2_2_conv2_weight, layer2_2_conv2_threshold, layer2_2_pgconv2_out);
	bn<32, 16, 16>(layer2_2_pgconv2_out, layer2_2_bn2_weight, layer2_2_bn2_bias, layer2_2_bn2_out);
	rprelu<32, 16, 16>(layer2_2_bn2_out, layer2_2_rprelu2_shift_x_bias, layer2_2_rprelu2_shift_y_bias, layer2_2_rprelu2_prelu_weight, layer2_2_rprelu2_out);
	shortcut<32, 16, 16>(layer2_2_rprelu2_out, layer2_2_bn3_out, layer2_2_shortcut2_out);
	bn<32, 16, 16>(layer2_2_shortcut2_out, layer2_2_bn4_weight, layer2_2_bn4_bias, layer2_2_bn4_out);

	////////////////////////////////////
	//////////// LAYER 3 ///////////////
	////////////////////////////////////

	quant_sign<32, 16, 16>(layer2_2_bn4_out, layer3_0_binarize1_out);
	layer3_pgconv<32, 64, 16, 16, 8, 8>(layer3_0_binarize1_out, layer3_0_conv1_weight, layer3_0_conv1_threshold, layer3_0_pgconv1_out);
	bn<64, 8, 8>(layer3_0_pgconv1_out, layer3_0_bn1_weight, layer3_0_bn1_bias, layer3_0_bn1_out);
	rprelu<64, 8, 8>(layer3_0_bn1_out, layer3_0_rprelu1_shift_x_bias, layer3_0_rprelu1_shift_y_bias, layer3_0_rprelu1_prelu_weight, layer3_0_rprelu1_out);
	avgpool_concat<32, 64, 16, 16, 8, 8>(layer2_2_bn4_out, layer3_0_concat_out);
	shortcut<64, 8, 8>(layer3_0_rprelu1_out, layer3_0_concat_out, layer3_0_shortcut1_out);
	bn<64, 8, 8>(layer3_0_shortcut1_out, layer3_0_bn3_weight, layer3_0_bn3_bias, layer3_0_bn3_out);

	quant_sign<64, 8, 8>(layer3_0_bn3_out, layer3_0_binarize2_out);
	layer3_pgconv<64, 64, 8, 8, 8, 8>(layer3_0_binarize2_out, layer3_0_conv2_weight, layer3_0_conv2_threshold, layer3_0_pgconv2_out);
	bn<64, 8, 8>(layer3_0_pgconv2_out, layer3_0_bn2_weight, layer3_0_bn2_bias, layer3_0_bn2_out);
	rprelu<64, 8, 8>(layer3_0_bn2_out, layer3_0_rprelu2_shift_x_bias, layer3_0_rprelu2_shift_y_bias, layer3_0_rprelu2_prelu_weight, layer3_0_rprelu2_out);
	shortcut<64, 8, 8>(layer3_0_rprelu2_out, layer3_0_bn3_out, layer3_0_shortcut2_out);
	bn<64, 8, 8>(layer3_0_shortcut2_out, layer3_0_bn4_weight, layer3_0_bn4_bias, layer3_0_bn4_out);

	quant_sign<64, 8, 8>(layer3_0_bn4_out, layer3_1_binarize1_out);
	layer3_pgconv<64, 64, 8, 8, 8, 8>(layer3_1_binarize1_out, layer3_1_conv1_weight, layer3_1_conv1_threshold, layer3_1_pgconv1_out);
	bn<64, 8, 8>(layer3_1_pgconv1_out, layer3_1_bn1_weight, layer3_1_bn1_bias, layer3_1_bn1_out);
	rprelu<64, 8, 8>(layer3_1_bn1_out, layer3_1_rprelu1_shift_x_bias, layer3_1_rprelu1_shift_y_bias, layer3_1_rprelu1_prelu_weight, layer3_1_rprelu1_out);
	shortcut<64, 8, 8>(layer3_1_rprelu1_out, layer3_0_bn4_out, layer3_1_shortcut1_out);
	bn<64, 8, 8>(layer3_1_shortcut1_out, layer3_1_bn3_weight, layer3_1_bn3_bias, layer3_1_bn3_out);

	quant_sign<64, 8, 8>(layer3_1_bn3_out, layer3_1_binarize2_out);
	layer3_pgconv<64, 64, 8, 8, 8, 8>(layer3_1_binarize2_out, layer3_1_conv2_weight, layer3_1_conv2_threshold, layer3_1_pgconv2_out);
	bn<64, 8, 8>(layer3_1_pgconv2_out, layer3_1_bn2_weight, layer3_1_bn2_bias, layer3_1_bn2_out);
	rprelu<64, 8, 8>(layer3_1_bn2_out, layer3_1_rprelu2_shift_x_bias, layer3_1_rprelu2_shift_y_bias, layer3_1_rprelu2_prelu_weight, layer3_1_rprelu2_out);
	shortcut<64, 8, 8>(layer3_1_rprelu2_out, layer3_1_bn3_out, layer3_1_shortcut2_out);
	bn<64, 8, 8>(layer3_1_shortcut2_out, layer3_1_bn4_weight, layer3_1_bn4_bias, layer3_1_bn4_out);

	quant_sign<64, 8, 8>(layer3_1_bn4_out, layer3_2_binarize1_out);
	layer3_pgconv<64, 64, 8, 8, 8, 8>(layer3_2_binarize1_out, layer3_2_conv1_weight, layer3_2_conv1_threshold, layer3_2_pgconv1_out);
	bn<64, 8, 8>(layer3_2_pgconv1_out, layer3_2_bn1_weight, layer3_2_bn1_bias, layer3_2_bn1_out);
	rprelu<64, 8, 8>(layer3_2_bn1_out, layer3_2_rprelu1_shift_x_bias, layer3_2_rprelu1_shift_y_bias, layer3_2_rprelu1_prelu_weight, layer3_2_rprelu1_out);
	shortcut<64, 8, 8>(layer3_2_rprelu1_out, layer3_1_bn4_out, layer3_2_shortcut1_out);
	bn<64, 8, 8>(layer3_2_shortcut1_out, layer3_2_bn3_weight, layer3_2_bn3_bias, layer3_2_bn3_out);

	quant_sign<64, 8, 8>(layer3_2_bn3_out, layer3_2_binarize2_out);
	layer3_pgconv<64, 64, 8, 8, 8, 8>(layer3_2_binarize2_out, layer3_2_conv2_weight, layer3_2_conv2_threshold, layer3_2_pgconv2_out);
	bn<64, 8, 8>(layer3_2_pgconv2_out, layer3_2_bn2_weight, layer3_2_bn2_bias, layer3_2_bn2_out);
	rprelu<64, 8, 8>(layer3_2_bn2_out, layer3_2_rprelu2_shift_x_bias, layer3_2_rprelu2_shift_y_bias, layer3_2_rprelu2_prelu_weight, layer3_2_rprelu2_out);
	shortcut<64, 8, 8>(layer3_2_rprelu2_out, layer3_2_bn3_out, layer3_2_shortcut2_out);
	bn<64, 8, 8>(layer3_2_shortcut2_out, layer3_2_bn4_weight, layer3_2_bn4_bias, layer3_2_bn4_out);

	for (int c = 0; c < 64; c ++) {
		float m = 0;
		for (int row = 0; row < 8; row ++) {
			for (int col = 0; col < 8; col ++) {
				m += layer3_2_bn4_out[c][row][col];
			}
		}
		avg_pool_out[c] = m/64.0;
	}

	for (int row = 0; row < 10; row ++) {
		float m = 0;
		for (int col = 0; col < 64; col ++) {
			m += avg_pool_out[col]*linear_weight[row][col];
		}
		classifier_out[row] = m + linear_bias[row];
	}
}

//#define SW_TEST
int main(int argc, char **argv)
{
	int correct_sw = 0;
	int correct_hw = 0;

	// Generate the expected result
	// Iterate over the rows of the A matrix
	// int num_tests = 10;

	load_image();
	load_label();

	for (int k = 0; k < NUM_TESTS; k ++) {
		float image[96][32][32] = {0};
		float p;

		get_image(images, k, image);

		////////////////////////////////
		//////// SOFTWARE /////////////
		////////////////////////////////

		FracNet_sw(image);

		int predict = 0;
		p = -1000;
		for (int i = 0; i < 10; i ++) {
			float cl = classifier_out[i];
			cout << classifier_out[i] << "  ";
			if (cl > p) {
				p = cl;
				predict = i;
			}
		}
		cout << endl;
		int label = labels[k];
		if (predict == label) {
			correct_sw ++;
		}
		cout << "Processed " << k + 1 << " pictures. " << endl;
		cout << "Software has "<< correct_sw << "/" << k + 1 << " correct." << endl;

#ifdef LAYER_TEST
		int print_row = 8;
		int print_col = 8;

		cout << "tb output avg_pool_out" << endl;
//		for (int row = 0; row < print_row; row ++) {
//			for (int col = 0; col < print_col; col ++) {
//				cout << layer3_2_bn4_out[0][row][col] << "  ";
//			}
//			cout << endl;
//		}
		for (int i = 0; i < 64; i ++){
			cout << avg_pool_out[i];
		}
		cout << endl;
		cout << "-------------------- above is tb.cc output ---------------------------" << endl;
#endif



		////////////////////////////////
		//////// HARDWARE //////////////
		////////////////////////////////
#ifdef LAYER_TEST
		float accelerator_output[64*32*32];
#else
		float accelerator_output[10];
#endif

		uint64 image_hw[3][32][32] = {0};

		for(int j = 0; j < 3; j ++){
			for(int row = 0; row < 32; row ++){
				for(int col = 0; col < 32; col ++){
					for(int b = 0; b < 32; b ++){
						if (image[j*32 + b][row][col] > 0) {
							image_hw[j][row][col][63 - b] = 1;
						} else {
							image_hw[j][row][col][63 - b] = 0;
						}
					}
				}
			}
		}

		FracNet_T(image_hw, accelerator_output);

#ifdef LAYER_TEST
		cout << endl << "accelerator output: "<< endl;
//		for (int row = 0; row < print_row; row ++) {
//			for(int col = 0; col < print_col; col ++) {
//				cout << accelerator_output[row*32 + col] << "  ";
//			}
//			cout << endl;
//		}
		for (int i = 0; i < 64; i ++){
			cout << accelerator_output[i];
		}
		cout << endl;

		FIX_FM_acc err = 0;
		FIX_FM_acc total_err = 0;
		FIX_FM_acc max_err = 0;
		int err_cnt = 0;
		int total = 0;
		for(int i=0; i<1; i++){
			for(int j=0; j<1; j++){
				for(int k=0; k<64; k++){
					err = hls::absf(accelerator_output[i*32*32+j*32+k] - avg_pool_out[k]);
					if (err > max_err) max_err = err;
					if (err > 0.1) {
						err_cnt += 1;
						cout << "(" << i << ", " << j << ", " << k << ") " << endl;
					}
//					if (err != 0) cout << "(" << i << ", " << j << ", " << k << ") ";
					total_err += err;
					total += 1;
				}
			}
		}
		cout << endl << "Total absolute error: " << total_err << endl;
		cout << "Total number of errors: " << err_cnt << "/" << total << endl;
		cout << "Maximum absolute pixel error: " << max_err << endl;
#else

		int predict_hw = 0;
		p = -1000;
		for (int i = 0; i < 10; i ++) {
			float cl = accelerator_output[i];
			cout << accelerator_output[i] << "  ";
			if (cl > p) {
				p = cl;
				predict_hw = i;
			}
		}
		cout << endl;
		if (predict_hw == labels[k]) {
			correct_hw ++;
		}

		cout << "Hardware has "<< correct_hw << "/" << k + 1 << " correct." << endl;
		cout << "\n" << endl;
#endif
	}

	return 0;
}
