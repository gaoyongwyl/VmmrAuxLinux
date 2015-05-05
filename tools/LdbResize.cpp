#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include <iostream>


#include <stdlib.h>
//#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>
//#include <io.h>
#include <iomanip>
#include <cv.h>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define _MAX_LINE 2048

bool ImageToDatum(cv::Mat& cv_img_origin, 
					  const int label,
					  const int height, 
					  const int width, 
					  caffe::Datum* datum) ;

int main(int argc, char** argv) 
{

	int defaultStep = 100;
	if( argc != 5 ){
		cout << "Usage: " << endl;
		cout << argv[0] << " LdbFullPath ResizedLdbFullPath Width Height" << endl;
		cout << "Resize images in LDB and save to a new LDB" << endl;

		return 0;
	}

	string strLdbPath = argv[1];
	string strResizeLDBPath = argv[2];
	int imWidth = atoi( argv[3] );
	int imHeight = atoi( argv[4] );
	
	cout << endl << "Parameter list: " << endl;
	cout << endl << "LDB path: " << strLdbPath << endl;
	cout << "Resized LDB Path: " << strResizeLDBPath << endl;
	cout << "New width: " << imWidth << endl;
	cout << "New height: " << imHeight << endl;
	cout << endl;


	// open existing one
	leveldb::DB* m_db;
	leveldb::Options options;
	options.create_if_missing = false;
	options.max_open_files = 100;
	LOG(INFO) << "Opening leveldb " << strLdbPath;
	leveldb::Status status = leveldb::DB::Open( options,strLdbPath,  &m_db );
	CHECK(status.ok()) << "Failed to open leveldb " << strLdbPath << std::endl << status.ToString();
	if( status.ok() == false ) {
		cout << endl << "Failed reopen leveldb : " << strLdbPath << endl;
		cout << status.ToString () << endl;
		return -1;
	} 

	// create new resized ldb
	// create new ldb:
	leveldb::DB* pResizedLdb;
	leveldb::Options options_new;
	options_new.error_if_exists = true;
	options_new.create_if_missing = true;
	options_new.write_buffer_size = 268435456;
	LOG(INFO) << "Opening leveldb " << strResizeLDBPath;
	status = leveldb::DB::Open( options_new, strResizeLDBPath, &pResizedLdb);
	CHECK(status.ok()) << "Failed to open resized leveldb " << strResizeLDBPath;

	//print db info:
	unsigned int numExist = 0;
	leveldb::Iterator* pIterLdb = m_db->NewIterator(leveldb::ReadOptions());	
	
	cout << endl;
	for( pIterLdb->SeekToFirst(); pIterLdb->Valid(); pIterLdb->Next() ) {
		numExist ++;
		if( numExist % 1000 == 0 ) {
			cout << "\r" << "DB record number: " << numExist;
		}
	}
	cout << endl;

	cout << "DB info: " << endl;
	cout << "Total records: " << numExist << endl << endl;

	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];

	caffe::Datum datum_write;
	int count = 0;
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	int data_size;
	bool data_size_initialized = false;

	int label = -1;
	for( pIterLdb->SeekToFirst(); pIterLdb->Valid(); pIterLdb->Next()  ) {

		CHECK(pIterLdb);
		CHECK(pIterLdb->Valid());
		caffe::Datum datum_read;

		datum_read.ParseFromString(pIterLdb->value().ToString());
		const string& data = datum_read.data();
		unsigned int datum_size_ = datum_read.channels() * datum_read.height() * datum_read.width();

		cv::Mat cv_img( datum_read.height(), datum_read.width(), CV_8UC3, Scalar(0,0,0 ) );	
		unsigned int k = 0;
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {					
					cv_img.at<cv::Vec3b>(h, w)[c] = (static_cast<uint8_t>(data[k++]) );
				}
			}
		}


		cv::Mat resizedImage;
		cv::resize( cv_img, resizedImage, Size( imWidth, imHeight ) );//

		label = datum_read.label();
		if( !ImageToDatum( resizedImage, label, -1, -1, &datum_write ) ){
			cout << "Image to Datum failed! " << pIterLdb->key().ToString() << endl;
		}
		//////////////////////////////////////////////////////////////////////////////////////
		///write data:
		if (!data_size_initialized) {
			data_size = datum_write.channels() * datum_write.height() * datum_write.width();
			data_size_initialized = true;
		} else {
			const string& data = datum_write.data();
			CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
				<< data.size();
		}

		// sequential
	   	string value;
		// get the value
		datum_write.SerializeToString(&value);
		batch->Put( pIterLdb->key().ToString(), value);
		if (++count % 1000 == 0) {
			pResizedLdb->Write(leveldb::WriteOptions(), batch);
			LOG(ERROR) << "Processed " << count << " files.";
			delete batch;
			batch = new leveldb::WriteBatch();
		}

#if 0
		char szTitle[_MAX_LINE];
		sprintf( szTitle, "Index: %d", imageIndex );
		cout << "Show image: " << szTitle << endl;
		cv::imshow( "LDB", cv_img );
		int keyValue= cv::waitKey();		
#endif //

		
	}

	// write the last batch
	if (count % 1000 != 0) {
		pResizedLdb->Write(leveldb::WriteOptions(), batch);
		LOG(ERROR) << "Processed " << count << " files.";
	}
	
	assert( pIterLdb->status().ok() );
	delete pIterLdb;

	delete m_db;

	delete batch;
	delete pResizedLdb;
	

	return 0;
}


bool ImageToDatum(cv::Mat& cv_img_origin, 
							   const int label,
					  const int height, 
					  const int width, 
					  caffe::Datum* datum) 
{
	cv::Mat cv_img;
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else {
		cv_img = cv_img_origin.clone();
	}

	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
					static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
			}
		}
	}
	return true;
}
