
#include <fstream>  // NOLINT(readability/streams)  
#include <iostream>  // NOLINT(readability/streams)  
#include <string>  
#include <utility>  
#include <vector>  

#include "caffe/data_layers.hpp"
#include "../../include/caffe/layer.hpp"  
#include "../../include/caffe/util/io.hpp"  
#include "../../include/caffe/util/math_functions.hpp"  
#include "../../include/caffe/util/rng.hpp"  
#include "../../include/caffe/vision_layers.hpp"  
#include "caffe/VMMR_data_layer.h"

//#define _DEBUG_SHOW_

namespace caffe {

template <typename Dtype>  
VMMRImageDataLayer<Dtype>::~VMMRImageDataLayer<Dtype>() 
{  
  this->JoinPrefetchThread();
}  

template <typename Dtype> 
void VMMRImageDataLayer<Dtype>::SetCurrentImage( cv::Mat& img )
{
   this->m_current_image.release();
   this->m_current_image = img.clone();
}

template <typename Dtype>  
void VMMRImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,  
		const vector<Blob<Dtype>*>& top) {  
  //Layer<Dtype>::SetUp(bottom, top);  
   const int new_height  = this->layer_param_.image_data_param().new_height();  
   const int new_width  = this->layer_param_.image_data_param().new_width();  
   const int channels = 3;
   CHECK((new_height == 0 && new_width == 0) ||  
           (new_height > 0 && new_width > 0)) << "Current implementation requires "  
           "new_height and new_width to be set at the same time.";  

    //initilize data blob by 3-channel zero dummy image:
   m_current_image = cv::Mat::zeros( new_height, new_width, CV_8UC3 );
   Datum datum;  
   CHECK( ConvertImageMatToDatum( m_current_image , 0, new_height, new_width, &datum));  

    // image  
   const int crop_size = this->layer_param_.image_data_param().crop_size();   
   const int batch_size = 1;//this->layer_param_.image_data_param().batch_size();  
   const string& mean_file = this->layer_param_.image_data_param().mean_file();  
   if (crop_size > 0) {  
     (top)[0]->Reshape(batch_size, channels, crop_size, crop_size);  
     this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
     this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
   } else {  
     (top)[0]->Reshape(batch_size, channels, new_height, new_width );  
     this->prefetch_data_.Reshape(batch_size, channels, new_height, new_width );  
     this->transformed_data_.Reshape(1, channels, new_height, new_width );  
  }  
  
  LOG(INFO) << "output data size: " << (top)[0]->num() << ","  
         << (top)[0]->channels() << "," << (top)[0]->height() << "," << (top)[0]->width();  
  // label  
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);

  // datum size  
  datum_channels_ = datum.channels();  
  datum_height_ = datum.height();  
  datum_width_ = datum.width();  
  datum_size_ = datum.channels() * datum.height() * datum.width();  
  CHECK_GT(datum_height_, crop_size);  
  CHECK_GT(datum_width_, crop_size);  
 }  

//--------------------------------下面是读取一张图片数据-----------------------------------------------  
 template <typename Dtype>  
 void VMMRImageDataLayer<Dtype>::InternalThreadEntry() 
 {  
   //CPUTimer batch_timer;
   //batch_timer.Start();
   //double read_time = 0;
   //double trans_time = 0;
   //CPUTimer timer;

  Datum datum;  
  CHECK(this->prefetch_data_.count());  
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();  
  const int batch_size = 1;//image_data_param.batch_size(); 这里我们只需要一张图片  
  const int new_height = image_data_param.new_height();  
  const int new_width = image_data_param.new_width();  
  const int crop_size = image_data_param.crop_size();  
  const bool is_color = image_data_param.is_color();

  const bool mirror = image_data_param.mirror();  
  const Dtype scale = image_data_param.scale();//image_data_layer相关参数  

  // datum scales  
  const int channels = datum_channels_;  
  const int height = datum_height_;  
  const int width = datum_width_;  
  const int size = datum_size_;  
  const int lines_size = lines_.size();  

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();  
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();  
		
  int item_id = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id){//读取一图片  

    //#define _DEBUG_SHOW_
#ifdef _DEBUG_SHOW_
     imshow( "Current Fetch Data", m_current_image );
     cv::waitKey();
#endif//_DEBUG_SHOW_

     //timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);

    cv::Mat cv_img;
    cv::resize( m_current_image, cv_img, cv::Size(new_width, new_height ));  
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    //trans_time += timer.MicroSeconds();

    //Any thing here. For we don't use this layer in training. only use it when test or valid!
    prefetch_label[item_id] = 0;

   }

  //batch_timer.Stop();
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms."; 
  }  
	
   INSTANTIATE_CLASS(VMMRImageDataLayer);  
   REGISTER_LAYER_CLASS(VMMRImageData);
  
}  // namespace caffe  



