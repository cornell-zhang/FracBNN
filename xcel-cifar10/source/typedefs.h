
#ifndef TYPEDEFS
#define TYPEDEFS

#include <cstddef>
#include <ap_int.h>
#include <ap_fixed.h>

//#define SW_TEST
//#define LAYER_TEST
#define True 1
#define False 0

#ifdef SW_TEST
	typedef float FIX_32_4;	//fix point
	typedef float FIX_32_25;	//fix point
	typedef float FIX_FM;	//fix point for feature map
	typedef float FIX_FM_acc;	//fix point for feature map
	typedef float FIX_FM_last;
	typedef float FIX_WT;	//fix point for weights
	typedef float FIX_32_16;
	typedef float FIX_32_10;
	typedef float FIX_32_12;
	typedef float FIX_16_6;
	typedef float FIX_16_5;
	typedef float FIX_16_4;
	typedef float FIX_16_10;

#else
	typedef ap_fixed<16, 9, AP_RND, AP_SAT> FIX_FM_acc;	//fix point for accumulation (16, 8) (20,9 works)
	typedef ap_fixed<12, 4, AP_RND, AP_SAT> FIX_WT;	//fix point for batchnorm weights (16, 4 works)

	typedef ap_fixed<32,12, AP_RND, AP_SAT> FIX_32_12;
	typedef ap_fixed<32,10, AP_RND, AP_SAT> FIX_32_10;

#endif

	typedef ap_uint<1> uint1;
	typedef ap_uint<2> uint2;
	typedef ap_uint<4> uint4;
	typedef ap_uint<6> uint6;
	typedef ap_uint<8> uint8;
	typedef ap_uint<16> uint16;
	typedef ap_uint<32> uint32;
	typedef ap_uint<64> uint64;
	typedef ap_uint<128> uint128;
	typedef ap_uint<256> uint256;
	typedef ap_uint<512> uint512;

	typedef ap_int<1> int1;
	typedef ap_int<2> int2;
	typedef ap_int<4> int4;
	typedef ap_int<6> int6;
	typedef ap_int<8> int8;
	typedef ap_int<16> int16;
	typedef ap_int<32> int32;
	typedef ap_int<64> int64;
	typedef ap_int<128> int128;
	typedef ap_int<256> int256;
	typedef ap_int<512> int512;

#endif

