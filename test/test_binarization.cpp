#include"../xnet_function.h"
#include<iostream>
#include<stdlib.h>
using namespace xnet;
using namespace std;
int main()
{
  //shape 
  int n = 1;
  int c = 3;
  int h = 4;
  int w = 4;

  float* data = (float*)malloc(sizeof(float)*(n*c*h*w));
  for(int i = 0 ; i < n ; i++)
    for(int j = 0; j < c ; j++)
      for(int k = 0; k < h ; k++)
        for(int l = 0; l < w ; l++)
          data[i*c*h*w+j*h*w+k*w+l] = (i*c*h*w+j*h*w+k*w+l+1)*pow(-1, i*c*h*w+j*h*w+k*w+l); 

  for(int i = 0; i<n ; i++){
    for(int j = 0; j < c*h*w; j++)
      cout<<data[i*c*h*w+j]<<" ";
    cout<<endl;
  }
  cout<<endl;
  cout<<endl;
    
  BinBlob<float> weight_blob(n, c, h, w);
  weight_blob.copyRealValueFrom(data);

  vector<float> alpha;
  alpha.resize(n);
  binarizeWeights(weight_blob, alpha);
  const vector<BinBlock>& bin_data = weight_blob.bin_data();
  for(int i = 0; i<bin_data.size(); i++){
    for(int j = 0; j < bin_data[i].size(); j++)
      cout<<bin_data[i][j]<<" ";
    cout<<endl;
  }
  cout<<endl;
  cout<<endl;

  BinBlob<float> input_blob(n, c, h, w);
  input_blob.copyRealValueFrom(data);
  binarizeIm2col(input_blob, 3, 4, 4, 1, 1, 0, 0, 1, 1, 1, 1);
  const vector<BinBlock>& nbin_data = input_blob.bin_data();
  for(int i = 0; i<nbin_data.size(); i++){
    for(int j = 0; j < nbin_data[i].size(); j++)
      cout<<nbin_data[i][j]<<" ";
    cout<<endl;
  }
}
