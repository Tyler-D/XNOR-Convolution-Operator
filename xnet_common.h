#ifndef BINARY_COMMON_HEADER
#define BINARY_COMMON_HEADER

#include<vector>
#include<bitset>
#include<memory>
#include<math.h>
#include<glog/logging.h>


using std::vector;
using std::bitset;
//using namespace std;

//@TODO(Raven) dynamic bitset map. Considering the kernel count and input count cannot
//perfectly fit 64 bit, we may have zero padding in fixed size bitset.
//And the speed of popcnt operations, which relates to the hardware architecture,
//doesn't has linear relation with the bit size. Therefore, maybe we can use
//dynamic bitset here.

//@TODO(Raven): Use uint8_t to store data 
#define BIN_SIZE 64

namespace xnet{

typedef vector<bitset<BIN_SIZE> > BinBlock;

template <typename Dtype>
class BinBlob final{
  public:
    BinBlob()
      :rv_data_(), count_(0) {shape_.resize(4);}

    ~BinBlob(){

    } 

    explicit BinBlob(const int num, const int channels, const int height,
        const int width){
        shape_.resize(4);
        shape_[0] = num;
        shape_[1] = channels;
        shape_[2] = height;
        shape_[3] = width;

        //bin_data_.resize(num);
        
        count_ = num * channels * height * width; 
    }

    void Reshape(const int num, const int channels, const int height,
        const int width){
      shape_.clear();
      shape_[0] = num;
      shape_[1] = channels;
      shape_[2] = height;
      shape_[3] = width;

      count_ = num * channels * height * width; 
      bin_data_.clear();

    }
    
    void copyRealValueFrom(const Dtype* data){
      //CHECK_NE(data, NULL);
      rv_data_ = const_cast<Dtype*>(data); 
    }

    const vector<int>& shape() {
      return shape_;
    }

    const vector<BinBlock>& bin_data(){
      return bin_data_;
    }

    const Dtype* rv_data(){
      return rv_data_;
    }

    const int count(){
      return count_;
    }

    vector<BinBlock>& mutable_bin_data(){
      return bin_data_;
    }

    BinBlob<Dtype>& operator=(const BinBlob<Dtype>&) = delete;


  protected:
       Dtype* rv_data_;
       int count_;
       //real value shape NCHW
       vector<int> shape_;
       //every ck^2 block is stored as a vector<bitset<BIN_SIZE>>
       vector<BinBlock> bin_data_; 

      
};

} 

#endif 
