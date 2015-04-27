#ifndef _VMMR_DATA_LAYER_H_
#define _VMMR_DATA_LAYER_H_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
//#include "pthread.h"
#include <thread>
#include "boost/scoped_ptr.hpp"
//#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "opencvlib.h"

#define INIT_SIZE cv::Size( 256, 256 )


namespace caffe {
	template <typename Dtype>  
	class VMMRImageDataLayer : public Layer<Dtype>  {  
	public:  
		explicit VMMRImageDataLayer(const LayerParameter& param)  
			: Layer<Dtype>(param) {}  
		virtual ~VMMRImageDataLayer();  
		virtual void SetUp(const vector<Blob<Dtype>*>& bottom,  
			vector<Blob<Dtype>*>* top);  

		virtual inline LayerParameter_LayerType type() const {  
			return LayerParameter_LayerType_VMMRIMAGE_DATA;  
		}  
		virtual inline int ExactNumBottomBlobs() const { return 0; }  
		virtual inline int ExactNumTopBlobs() const { return 2; }  

	public:
		void SetCurrentImage( cv::Mat& img );		

	protected:  
		void fetchData();  
		

	protected:
		//virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		//	vector<Blob<Dtype>*>* top) = 0;
		virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
			vector<Blob<Dtype>*>* top);  

		// virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		//const bool propagate_down,
		//vector<Blob<Dtype>*>* bottom) = 0;
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down,vector<Blob<Dtype>*>* bottom)
		{
		}

		vector<std::pair<std::string, int> > lines_;  
		int lines_id_;  
		int datum_channels_;  
		int datum_height_;  
		int datum_width_;  
		int datum_size_;  
		Blob<Dtype> prefetch_data_;  
		Blob<Dtype> prefetch_label_;  
		Blob<Dtype> data_mean_;  
		Caffe::Phase phase_; 

		cv::Mat m_current_image;
	};  

} // namespace caffe

#endif //_VMMR_DATA_LAYER_H_