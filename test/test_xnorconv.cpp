#include"../xnet_function.h"
#include<iostream>
#include<stdlib.h>
#include<sys/time.h>
using namespace xnet;
using namespace std;
typedef struct data
{
  data(int n , int c, int h, int w): n_(n), c_(c),
  h_(h), w_(w){
    p_data_ = (float*)malloc(sizeof(float)*n*c*h*w);
    for(int i = 0 ; i < n ; i++)
      for(int j = 0; j < c ; j++)
        for(int k = 0; k < h ; k++)
          for(int l = 0; l < w ; l++)
            p_data_[i*c*h*w+j*h*w+k*w+l] = (i*c*h*w+j*h*w+k*w+l+1)*pow(-1, i*c*h*w+j*h*w+k*w+l); 
    //for(int i = 0; i<n ; i++){
      //for(int j = 0; j < c*h*w; j++)
        //cout<<p_data_[i*c*h*w+j]<<" ";
      //cout<<endl;
    //}
  } 

  int n_;
  int c_;
  int h_;
  int w_;
  float* p_data_;
}Blob;
int main()
{
  //shape 
  int n = 1;
  int c = 64;
  int h = 128;
  int w = 128;
  int k = 3;
  int stride = 1;
  int pad = 0;
  int dilation = 1;

  Blob weights(1,c,k,k);
  BinBlob<float> bin_weights(1,c,k,k);
  bin_weights.copyRealValueFrom(weights.p_data_);

  Blob input(1,c,h,w);
  BinBlob<float> bin_input(1,c,h,w);
  bin_input.copyRealValueFrom(input.p_data_);

  int output_h = (h + 2*pad - (dilation*(k-1)+1))/stride + 1;
  int output_w = (w + 2*pad - (dilation*(k-1)+1))/stride + 1;
  
  float* output = (float*)malloc(sizeof(float)*output_h*output_w*n);

  struct timeval start,end;
  gettimeofday(&start, NULL);
  xnorConvolution(bin_input, bin_weights, output, k, k, pad, pad, stride, stride,
                  dilation, dilation);
  gettimeofday(&end, NULL);
  unsigned long diff = 1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000 ;
  cout<<"xnor time: "<< diff<<"/ms"<<endl;
  
  //for(int i = 0; i<output_h; i++){
    //for(int j = 0; j< output_w; j++)
      //cout<<output[i*output_w+j]<<" ";
    //cout<<endl;
  //}

}
