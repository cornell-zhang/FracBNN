
#ifndef TYPEDEFS
#define TYPEDEFS

#include <cstddef>
#include <ap_int.h>
#include <ap_fixed.h>

//#define SW_TEST
#define LAYER_TEST

#ifdef SW_TEST

typedef float FIX_FM_acc;   //fix point for feature map
typedef float FIX_WT;       //fix point for weights
typedef float FIX_14_2;
typedef float FIX_14_4;

#else

typedef ap_fixed<13, 9, AP_RND, AP_SAT> FIX_FM_acc; //fix point for accumulation (16, 8)
typedef ap_fixed<14, 6, AP_RND, AP_SAT> FIX_WT; //fix point for weights (14, 6)

typedef ap_fixed<10, 2, AP_RND, AP_SAT> CONST_2;
typedef ap_fixed<6, 4, AP_RND, AP_SAT> CONST_4;
#endif

typedef ap_fixed<32, 6, AP_RND, AP_SAT> FIX_1D_PACK;
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
typedef ap_int<10> int10;
typedef ap_int<16> int16;

#endif

