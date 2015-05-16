#ifndef _VMMR_DATA_LAYER_H_
#define _VMMR_DATA_LAYER_H_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
//#include "pthread.h"
#include "boost/scoped_ptr.hpp"
//#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "opencv2/opencv.hpp"

#define __GXX_EXPERIMENTAL_CXX0X__

#define INIT_SIZE cv::Size( 256, 256 )


namespace caffe {
template <typename Dtype>  
class VMMRImageDataLayer : public VmmrBasePrefetchingDataLayer<Dtype>  {  
 public:  
    explicit VMMRImageDataLayer(const LayerParameter& param)  
      : VmmrBasePrefetchingDataLayer<Dtype> (param) {}  
    virtual ~VMMRImageDataLayer();  
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,  
			const vector<Blob<Dtype>*>& top);  

   virtual inline const char* type() const {return "VMMRImageData";}  
   virtual inline int ExactNumBottomBlobs() const { return 0; }  
   virtual inline int ExactNumTopBlobs() const { return 2; }  

public:
   void SetCurrentImage( cv::Mat& img );		

protected:  
   virtual  void InternalThreadEntry();  

protected:
  vector<std::pair<std::string, int> > lines_;  
  int lines_id_;  
  int datum_channels_;  
  int datum_height_;  
  int datum_width_;  
  int datum_size_;  
  //Blob<Dtype> prefetch_data_;  
  //Blob<Dtype> prefetch_label_; 



  cv::Mat m_current_image;
 };  

} // namespace caffe

#endif //_VMMR_DATA_LAYER_H_
