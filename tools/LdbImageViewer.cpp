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
//
//#include <io.h>
#include <iomanip>
#include <cv.h>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <highgui.h>

using namespace std;
using namespace cv;

#define _MAX_LINE 2048

int main(int argc, char** argv) 
{

	int defaultStep = 100;
	if( argc < 2 || argc > 3 ){
		cout << "Usage: " << endl;
		cout << argv[0] << " LdbFullPath [Step(default=100)]" << endl;
		cout << "View images in LDB one by one with predefined step" << endl;

		return 0;
	}

	string strLdbPath = argv[1];
	
	if( argc == 3 ) {
		string strViewStep = argv[2];
		defaultStep = atoi( strViewStep.c_str() );
	}


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

	unsigned int imageIndex = 0;
	for( pIterLdb->SeekToFirst(); pIterLdb->Valid(); pIterLdb->Next(), imageIndex++ ) {
		bool bComeToEnd = false;
		for( int n = 0; n < defaultStep; n ++ ) {
			if( pIterLdb->Valid() ) {
				pIterLdb->Next();
				imageIndex ++;
			} else {
				bComeToEnd = true;
				break;
			}
		}
		if( bComeToEnd == true ) {
			cout << "Come to the end of LDB !" << endl;
			break;
		}
		CHECK(pIterLdb);
		CHECK(pIterLdb->Valid());
		caffe::Datum datum;
		datum.ParseFromString(pIterLdb->value().ToString());

		const string& data = datum.data();
		unsigned int datum_size_ = datum.channels() * datum.height() * datum.width();

		cv::Mat cv_img( datum.height(), datum.width(), CV_8UC3, Scalar(0,0,0 ) );	
		unsigned int k = 0;
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {					
					cv_img.at<cv::Vec3b>(h, w)[c] = (static_cast<uint8_t>(data[k++]) );
				}
			}
		}

		char szTitle[_MAX_LINE];
		sprintf( szTitle, "Index: %d", imageIndex );
		cout << "Show image: " << szTitle << endl;
		cv::imshow( "LDB", cv_img );
		int keyValue= cv::waitKey();

		if( keyValue == 'q' || keyValue == 'Q' ) {
			break;
		}
	}
	
	assert( pIterLdb->status().ok() );
	delete pIterLdb;

	delete m_db;
	

	return 0;
}
