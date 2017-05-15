#include<iostream>
#include<sys/time.h>
#include<cblas.h>
#include<math.h>
#include<stdlib.h>
#include<stdint.h>
using namespace std;

int main()
{
  
  //int n = 1;
  //int c = 3;
  int h = 126*126;
  int w = 576;
  int k = 1; 

  float* data = (float*)malloc(sizeof(float)*(h*w));
  float* result = (float*)malloc(sizeof(float)*(h*w));
  for(int i = 0 ; i < h ; i++)
    for(int j = 0; j < w ; j++)
      data[i*w+j] = pow(-1, i*w+j); 

  //uint64_t* bin = (uint64_t*)malloc(sizeof(uint64_t)*(h*w)); 
  //for(int i = 0 ; i < h ; i++)
    //for(int j = 0; j < w ; j++)
      //bin[i*w+j] = pow(-1, i*w+j); 


  //cblas
  struct timeval start,end;
  gettimeofday(&start, NULL);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, h,h,h, 1.0, data, h, data, h, 0, result, h);
  gettimeofday(&end, NULL);
  unsigned long diff = 1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000 ;
  cout<<"cblas gemm time: "<< diff<<"/ms"<<endl;

  //add_
  gettimeofday(&start, NULL);
  int num = 10*h;
  for(int i = 0; i< num; i++)
    cblas_sasum(h, data, 1); 
  gettimeofday(&end, NULL);
  diff = 1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000 ;
  cout<<"cblas 1000000 add  time: "<< diff<<"/ms"<<endl;
}
