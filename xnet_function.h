#ifndef BIN_MATH_FUNCTION 
#define BIN_MATH_FUNCTION
#include "xnet_common.h"
#include<cblas.h>
#include<math.h>
#include<stdint.h>
#include<x86intrin.h>
#include<immintrin.h>

#ifdef DEBUG_XNOR
#include<iostream>
using namespace std;
#endif

namespace xnet{

template<typename Dtype>
Dtype xnet_cpu_asum(const int n , const Dtype* x);

template<>
float xnet_cpu_asum<float>(const int n, const float* x){
  return cblas_sasum(n, x, 1);
}

template<>
double xnet_cpu_asum<double>(const int n, const double* x){
  return cblas_dasum(n, x, 1);
}


template<typename Dtype>
inline int8_t xnet_sign(Dtype val){
  return (Dtype(0) < val) - ( val < Dtype(0));
} 

//  
//@code from https://github.com/WojciechMula/sse-popcount


uint8_t lookup8bit[256] = {
	/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
	/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
	/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
	/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
	/* 10 */ 1, /* 11 */ 2, /* 12 */ 2, /* 13 */ 3,
	/* 14 */ 2, /* 15 */ 3, /* 16 */ 3, /* 17 */ 4,
	/* 18 */ 2, /* 19 */ 3, /* 1a */ 3, /* 1b */ 4,
	/* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5,
	/* 20 */ 1, /* 21 */ 2, /* 22 */ 2, /* 23 */ 3,
	/* 24 */ 2, /* 25 */ 3, /* 26 */ 3, /* 27 */ 4,
	/* 28 */ 2, /* 29 */ 3, /* 2a */ 3, /* 2b */ 4,
	/* 2c */ 3, /* 2d */ 4, /* 2e */ 4, /* 2f */ 5,
	/* 30 */ 2, /* 31 */ 3, /* 32 */ 3, /* 33 */ 4,
	/* 34 */ 3, /* 35 */ 4, /* 36 */ 4, /* 37 */ 5,
	/* 38 */ 3, /* 39 */ 4, /* 3a */ 4, /* 3b */ 5,
	/* 3c */ 4, /* 3d */ 5, /* 3e */ 5, /* 3f */ 6,
	/* 40 */ 1, /* 41 */ 2, /* 42 */ 2, /* 43 */ 3,
	/* 44 */ 2, /* 45 */ 3, /* 46 */ 3, /* 47 */ 4,
	/* 48 */ 2, /* 49 */ 3, /* 4a */ 3, /* 4b */ 4,
	/* 4c */ 3, /* 4d */ 4, /* 4e */ 4, /* 4f */ 5,
	/* 50 */ 2, /* 51 */ 3, /* 52 */ 3, /* 53 */ 4,
	/* 54 */ 3, /* 55 */ 4, /* 56 */ 4, /* 57 */ 5,
	/* 58 */ 3, /* 59 */ 4, /* 5a */ 4, /* 5b */ 5,
	/* 5c */ 4, /* 5d */ 5, /* 5e */ 5, /* 5f */ 6,
	/* 60 */ 2, /* 61 */ 3, /* 62 */ 3, /* 63 */ 4,
	/* 64 */ 3, /* 65 */ 4, /* 66 */ 4, /* 67 */ 5,
	/* 68 */ 3, /* 69 */ 4, /* 6a */ 4, /* 6b */ 5,
	/* 6c */ 4, /* 6d */ 5, /* 6e */ 5, /* 6f */ 6,
	/* 70 */ 3, /* 71 */ 4, /* 72 */ 4, /* 73 */ 5,
	/* 74 */ 4, /* 75 */ 5, /* 76 */ 5, /* 77 */ 6,
	/* 78 */ 4, /* 79 */ 5, /* 7a */ 5, /* 7b */ 6,
	/* 7c */ 5, /* 7d */ 6, /* 7e */ 6, /* 7f */ 7,
	/* 80 */ 1, /* 81 */ 2, /* 82 */ 2, /* 83 */ 3,
	/* 84 */ 2, /* 85 */ 3, /* 86 */ 3, /* 87 */ 4,
	/* 88 */ 2, /* 89 */ 3, /* 8a */ 3, /* 8b */ 4,
	/* 8c */ 3, /* 8d */ 4, /* 8e */ 4, /* 8f */ 5,
	/* 90 */ 2, /* 91 */ 3, /* 92 */ 3, /* 93 */ 4,
	/* 94 */ 3, /* 95 */ 4, /* 96 */ 4, /* 97 */ 5,
	/* 98 */ 3, /* 99 */ 4, /* 9a */ 4, /* 9b */ 5,
	/* 9c */ 4, /* 9d */ 5, /* 9e */ 5, /* 9f */ 6,
	/* a0 */ 2, /* a1 */ 3, /* a2 */ 3, /* a3 */ 4,
	/* a4 */ 3, /* a5 */ 4, /* a6 */ 4, /* a7 */ 5,
	/* a8 */ 3, /* a9 */ 4, /* aa */ 4, /* ab */ 5,
	/* ac */ 4, /* ad */ 5, /* ae */ 5, /* af */ 6,
	/* b0 */ 3, /* b1 */ 4, /* b2 */ 4, /* b3 */ 5,
	/* b4 */ 4, /* b5 */ 5, /* b6 */ 5, /* b7 */ 6,
	/* b8 */ 4, /* b9 */ 5, /* ba */ 5, /* bb */ 6,
	/* bc */ 5, /* bd */ 6, /* be */ 6, /* bf */ 7,
	/* c0 */ 2, /* c1 */ 3, /* c2 */ 3, /* c3 */ 4,
	/* c4 */ 3, /* c5 */ 4, /* c6 */ 4, /* c7 */ 5,
	/* c8 */ 3, /* c9 */ 4, /* ca */ 4, /* cb */ 5,
	/* cc */ 4, /* cd */ 5, /* ce */ 5, /* cf */ 6,
	/* d0 */ 3, /* d1 */ 4, /* d2 */ 4, /* d3 */ 5,
	/* d4 */ 4, /* d5 */ 5, /* d6 */ 5, /* d7 */ 6,
	/* d8 */ 4, /* d9 */ 5, /* da */ 5, /* db */ 6,
	/* dc */ 5, /* dd */ 6, /* de */ 6, /* df */ 7,
	/* e0 */ 3, /* e1 */ 4, /* e2 */ 4, /* e3 */ 5,
	/* e4 */ 4, /* e5 */ 5, /* e6 */ 5, /* e7 */ 6,
	/* e8 */ 4, /* e9 */ 5, /* ea */ 5, /* eb */ 6,
	/* ec */ 5, /* ed */ 6, /* ee */ 6, /* ef */ 7,
	/* f0 */ 4, /* f1 */ 5, /* f2 */ 5, /* f3 */ 6,
	/* f4 */ 5, /* f5 */ 6, /* f6 */ 6, /* f7 */ 7,
	/* f8 */ 5, /* f9 */ 6, /* fa */ 6, /* fb */ 7,
	/* fc */ 6, /* fd */ 7, /* fe */ 7, /* ff */ 8
};


uint64_t popcnt_lookup_8bit(const uint64_t* x){
  uint64_t result = 0;
  const uint8_t* data = reinterpret_cast<const uint8_t*>(x);
  for(int i = 0; i < 8; i++)
    result += lookup8bit[data[i]];
  return result;
}

uint64_t popcnt_lookup_8bit(const uint8_t* data, const size_t n) {

    size_t result = 0;

    size_t i = 0;
    while (i + 4 <= n) {
        result += lookup8bit[data[i]]; i++;
        result += lookup8bit[data[i]]; i++;
        result += lookup8bit[data[i]]; i++;
        result += lookup8bit[data[i]]; i++;
    }

    while (i < n) {
        result += lookup8bit[data[i]]; i++;
    }

    return result;
}


#if defined(HAVE_SSE_INSTRUCTIONS)
inline uint64_t popcnt_cpu_64bit(const uint64_t* x){
  uint64_t v = *x;
  return _popcnt64(v);
}

uint64_t popcnt_cpu_64bit(const uint8_t* data, const size_t n) {

    uint64_t result = 0;

    uint64_t v, i = 0;
#define ITER { \
        v = *reinterpret_cast<const uint64_t*>(data + i); \
        result += _popcnt64(v); \
        i += 8; \
    }

    while (i + 4*8 <= n) {
        ITER ITER ITER ITER
    }

#undef ITER

    while (i < n) {
        result += lookup8bit[data[i]];
        i++;
    }

    return result;
}
#endif

#if defined(HAVE_NEON_INSTRUCTIONS)
uint64_t popcnt_neon_vcnt(const uint64_t* x){
  uint64_t result = 0;
  return result;
}
uint64_t popcnt_neon_vcnt(const uint8_t* data, const size_t size)
{
    const size_t chunk_size = 16 * 4 * 2;

    uint8_t* ptr = const_cast<uint8_t*>(data);

    const size_t n = size / chunk_size;
    const size_t k = size % chunk_size;

    uint32x4_t sum = vcombine_u32(vcreate_u32(0), vcreate_u32(0));

    for (size_t i=0; i < n; i++, ptr += chunk_size) {

        uint8x16x4_t input0 = vld4q_u8(ptr + 0 * 16 * 4);
        uint8x16x4_t input1 = vld4q_u8(ptr + 1 * 16 * 4);

        uint8x16_t t0   = vcntq_u8(input0.val[0]);
        t0 = vaddq_u8(t0, vcntq_u8(input0.val[1]));
        t0 = vaddq_u8(t0, vcntq_u8(input0.val[2]));
        t0 = vaddq_u8(t0, vcntq_u8(input0.val[3]));

        t0 = vaddq_u8(t0, vcntq_u8(input1.val[0]));
        t0 = vaddq_u8(t0, vcntq_u8(input1.val[1]));
        t0 = vaddq_u8(t0, vcntq_u8(input1.val[2]));
        t0 = vaddq_u8(t0, vcntq_u8(input1.val[3]));

        const uint16x8_t t1 = vpaddlq_u8(t0);

        sum = vpadalq_u16(sum, t1);
    }

    uint32_t scalar = 0;
    uint32_t tmp[4];

    vst1q_u32(tmp, sum);
    for (int i=0; i < 4; i++) {
        scalar += tmp[i];
    }

    for (size_t j=0; j < k; j++) {
        scalar += lookup8bit[ptr[j]];
    }

    return scalar;
}
#endif 


inline uint64_t xnet_popcnt(uint64_t x){
  uint64_t result = 0;
#if defined(HAVE_NOTHING)
  result = popcnt_lookup_8bit(&x);
#endif

#if defined(HAVE_SSE_INSTRUCTIONS)
  result = popcnt_cpu_64bit(&x);
#endif

#if defined(HAVE_NEON_INSTRUCTIONS)
  result = popcnt_neon_vcnt(&x);
#endif

  return result;
}

/*
 @brief n-2*popcnt(a^b).It's the base operation for XNOR convolution   
 @input:
      - n. actual size of operation binary code. In XNOR convolution, it is c*k^2
      - a. operation binary code.
      - b. operation binary code.
 @return:
      - xnor convolution result of two binary code vector.
 */
inline int xnet_sconv(const uint64_t n, const vector<bitset<BIN_SIZE> >& a, 
                      const vector<bitset<BIN_SIZE> >& b){
  CHECK_EQ(a.size(), b.size());
  uint64_t result = 0;
  //popcnt(a^b)
  for(int i = 0; i < a.size(); i++)
  {
    result += xnet_popcnt(static_cast<uint64_t>((a[i].to_ulong())^(b[i].to_ulong()))); 
  }
  int v = static_cast<int>(result); 
  int size = static_cast<int>(n);
  return (size - 2*v);
}

/*
@brief Binarize convolution weights. 
       This function calculates the scale factor $\alpha$ 
       and times sign(w) to estimate the real value weights. 
@input:
      - weights. BinBlob store both real value weights and  binary value weights.
      - alpha. scale factor. 
*/
template <typename Dtype>
void binarizeWeights(BinBlob<Dtype>& weights, vector<Dtype>& alpha){
  const vector<int>& shape = weights.shape(); 
  int filter_num = shape[0];
  int kernel_count = shape[1] * shape[2] * shape[3];
  unsigned long count = filter_num * kernel_count;
  const Dtype* rv_data = weights.rv_data(); 
  //caculate alpha 
  for(int filter_idx = 0; filter_idx < filter_num; filter_idx++) 
    alpha.push_back(xnet_cpu_asum<Dtype>(kernel_count, rv_data+filter_idx*kernel_count) / kernel_count);
  //sign 
  int bin_block_size = ceil(float(kernel_count)/ BIN_SIZE);  
  vector<BinBlock>& bin_data = weights.mutable_bin_data(); 
  bin_data.resize(filter_num);
  //The store style is NCHW
  for (int filter_idx = 0; filter_idx < filter_num; filter_idx++){
    //store the weights as kernel_size(ck*wk*hk)
    for (int bin_block_idx =0; bin_block_idx < bin_block_size;
         bin_block_idx++){ 
      //remeber the bitset is stored as low-end mode.But this mode may not
      //influence the popcnt
      bitset<BIN_SIZE> temp_bin;
      for(int j = 0 ; j < BIN_SIZE; j++){
        unsigned long weight_idx = filter_idx*kernel_count+ 
          bin_block_idx*BIN_SIZE+j;
        if ( weight_idx > count) break;
        if(rv_data[weight_idx]>0){  
            temp_bin.set(j,1);
        }
      } 
      bin_data[filter_idx].push_back(temp_bin);
    }
    
  }
}  

/*
 @brief binarize the input.store it as its original size 
*/

template <typename Dtype>
void binarizeInput(BinBlob<Dtype>& input)
{
  //const Dtype* rv_data = input.rv_data();
  
  //const vector<int>& shape = input.shape();
  //int data_num = shape[0];
  //int data_channel = shape[1];
  //int data_height = shape[2];
  //int data_width = shape[3]; 
  
  ////the data store as 
}

/*
@brief im2col method after binarizing the input. Convert NCHW image to 
output_h*output_w*CK^2 
*/

template <typename Dtype>
void im2col()
{
} 

/*
 @brief code from caffe.The casting allows to use one condition instead of two.
 */
inline bool is_a_ge_zero_and_a_lt_b(int a, int b){
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
/*
  @brief combine the binarization and im2col. It's easier way than separating 
         binarization and image2column 
  @input 
 
*/
template <typename Dtype>
void binarizeIm2col(BinBlob<Dtype>& input, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w) 
{
  const int output_h = (height + 2*pad_h - (dilation_h * (kernel_h - 1)+1))/stride_h + 1;
  const int output_w = (width + 2*pad_w - (dilation_w*(kernel_h - 1)+1))/stride_w + 1;
  const int channel_size = height * width;
  const int kernel_size = kernel_h * kernel_w;
  const Dtype* rv_data = input.rv_data();
  vector<BinBlock>& bin_data = input.mutable_bin_data();
  //initialize bin_data:
  int bin_block_size = ceil(float(channels * kernel_h * kernel_w)/ BIN_SIZE);
  int bin_block_num = output_h * output_w;
  bin_data.resize(bin_block_num);
  for(int i = 0; i < bin_block_num; i++)
    bin_data[i].resize(bin_block_size);
  //caffe im2col
  int position = 0;
  for(int channel = -1;++channel< channels;rv_data += channel_size){
    for(int kernel_row = 0; kernel_row < kernel_h; kernel_row++){
      for(int kernel_col = 0; kernel_col < kernel_w; kernel_col++){
        //int position = channel*kernel_size + 
                        //kernel_row * kernel_w + kernel_col;
        int bin_block_id = position / BIN_SIZE; 
        int bin_block_size_id = position % BIN_SIZE; 
        position++;
        int input_row = -pad_h + kernel_row * dilation_h;
        int output_offset = 0;
        for(int output_row = 0; output_row < output_h; output_row++){
          if(!is_a_ge_zero_and_a_lt_b(input_row, height)){
              output_offset += output_w;
          }else{
            int input_col = -pad_w + kernel_col*dilation_w;
            for(int output_col = 0;output_col< output_w; output_col++){
              if (is_a_ge_zero_and_a_lt_b(input_col, width)){
                if (rv_data[input_row*width + input_col]>0){
                //Considering the 
                  //int position = channel*kernel_h*kernel_w + 
                                 //kernel_row * kernel_w + kernel_col;
                  //int bin_block_id = position / BIN_SIZE; 
                  //int bin_block_size_id = position % BIN_SIZE; 
                  bin_data[output_offset][bin_block_id].set(
                      bin_block_size_id,1); 
                }
              }else{

              }
              output_offset++;
              input_col += stride_w;
            }
          }
        input_row += stride_h;
        }
      }
    }
  }

}

template <typename Dtype>
void xnorConvolution(BinBlob<Dtype>& input, BinBlob<Dtype>& weights, Dtype* output,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, 
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w)
{
  //init weights parameter
  vector<Dtype> alpha; 
  binarizeWeights(weights, alpha);
  const vector<int>& weights_shape = weights.shape();
  const vector<BinBlock>& bin_weights = weights.bin_data();
  int filter_num = weights_shape[0];
  uint64_t kernel_size = static_cast<uint64_t>(weights_shape[1]*weights_shape[2]*weights_shape[3]);
#if defined(DEBUG_XNOR)
  //LOG(INFO)<<alpha; 
  LOG(INFO)<<"Weights:";
  for(int i = 0; i<bin_weights.size(); i++){
    for(int j = 0; j < bin_weights[i].size(); j++)
      cout<<bin_weights[i][j]<<" ";
    cout<<endl;
  }
#endif 

  //init input parameter
  const vector<int>& input_shape = input.shape();
  int batch_size = input_shape[0];
  int channels = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];
  const Dtype* input_data = input.rv_data();
  BinBlob<Dtype> image_input(1, channels, height, width); 

  //intit output parameter
  int output_offset = 0;

  //xnor convolution 
  for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
    int input_offset = batch_idx*channels*height*width;
    image_input.copyRealValueFrom(input_data+input_offset); 
    binarizeIm2col(image_input, channels, height, width, kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w); 
    const vector<BinBlock>& bin_image = image_input.bin_data();
#if defined(DEBUG_XNOR)
  LOG(INFO)<<"Input:";
  for(int i = 0; i<bin_image.size(); i++){
    for(int j = 0; j < bin_image[i].size(); j++)
      cout<<bin_image[i][j]<<" ";
    cout<<endl;
  }
#endif  
    for(int filter_idx = 0; filter_idx < filter_num; filter_idx++){
      for(int field_idx = 0; field_idx < bin_image.size(); field_idx++){
          *(output++) = (alpha[filter_idx])*static_cast<Dtype>(xnet_sconv(kernel_size, 
                        bin_image[field_idx], bin_weights[filter_idx]));   
      }
    }
  }

}

}
#endif 
