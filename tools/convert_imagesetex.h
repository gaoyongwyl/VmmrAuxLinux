// Copyright 2014 BVLC and contributors.
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <algorithm>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include <iostream>

using namespace std;

#include <unistd.h>
#include <stdlib.h>
#include <time.h>
//#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>
//#include <io.h>
#include <iomanip>
#include <cv.h>
#include <highgui.h>
//#include <opencv2/imgcodecs.hpp>

#include <opencv.hpp>

using namespace cv;

//#define PATH_MAX 2048
#define _MAX_LINE 4096
#define _MAX_DRIVE 256
#define _MAX_FNAME 2048

#define VMMR_INFLAT_COEFF 0.1f
#define VMMR_NewWidth 380

const int LDB_WRITE_BATCH = 1000; //default 1000

const string TRAIN_LDB="TrainLDB";
const string TEST_LDB="TestLDB";
#define IS_TESTSAVE 1    //whether save images except for leveldb writing. 1 denote save image to directory.
#define IS_TRAINSAVE 0   //the same as the above. 0 denote don't, just write sample data to leveldb.

const string VMMR_DATA_TOPDIR = "/home/ygao/Projects/VehicleRecogntition/Data";
const string VMMR_LIST_FOLDER = "_LabelList/Makemodel/TT0";
const string VMMR_TRAIN_LIST = "Makemodel_TrainLabelList.txt";
const string VMMR_TEST_LIST = "Makemodel_TestLabelList.txt";
const string VMMR_RANDSHUFFLE_LIST_PREFIX = "Rand";
const string VMMR_AUG_LIST_PREFIX = "Aug"; //the name after augment
const string VMMR_DONE_LIST_PREFIX = "Don"; //
const string VMMR_EXCEPTION_PREFIX = "Excep";

const string DBCONV_LOG = "Log_aug_convldb";

const string VMMR_USEDALIGN = "AAuM";
const string ossep = "/";

typedef enum VMMRPPTID{
	PT_COLOR = 0, //original color image
	PT_GRAY = 1,  //convert from the above color image
	PT_EQUALIZEHIST = 2,
	PT_EQUALIZEHISTCOLOR =3
}VMMRPPTID;

typedef enum VMMRKPID{ 
	MAKE_LogoArea = -2, // conduct Make classification crop vehicle logo area
	VF_VehicleFace = -1, // conduct makemodel classification use only face area
	// the following are for multiple path makemodel classification :
	KP_WinGlassLT = 0, // Windshield Glass Left-top
	KP_WinGlassRT = 1, //: Windshield Glass Right-top
	KP_WinGlassLB = 2, //: Windshield Glass Left-bottom
	KP_WinGlassRB = 3, //: Windshield Glass Right-bottom
	KP_LeftHLamp = 4, //: Left Head Lamp center
	KP_RightHLamp = 5, //: Right Head Lamp center
	KP_FrontBumpLB = 6, //: Front Bumper Left Bottom corner
	KP_FrontBumpRB = 7, //: Front Bumper Right Bottom corner
	KP_VehicleLogo = 8,  //: Vehicle Logo center
	KP_LicensePC = 9, //: License Plate center
	KP_MidLineBot = 10 //: Middle line bottom
}VMMRKPID;

class VmmrLevelDB{
public:
	VmmrLevelDB( const string& strDataSetName,\
		const string& strDbName, \
		const VMMRPPTID& enuPptID, \
		const VMMRKPID& enuKpid, \
		const int& NewWidth, \
		const bool& bNewLdb = true,\
		const bool bWithAppendList = false );
	~VmmrLevelDB(){
		delete m_batch;
		delete m_db;
	};

public:
	
	int PushData( cv::Mat& cv_img_origin, \
		int line_id, string& relpathName, \
		const int label, \
		const int height=-1, const int width=-1 );
	void WriteData();
	void WriteDataEnd();
	string& GetLbdParentPath(){
		return this->m_strLdbParentFolder;
	};

	string& GetImageDataPath(){
		return this->m_strDataSetDataFolder; 
	};

	int GetLdbImageRelpathNameToAddedSet();
	int AddRelpathnameToAddedSet( string& strRelpathname ){
		this->m_setImagesAdded.insert( strRelpathname );
		return this->m_setImagesAdded.size();
	};
	bool IsRelpathNameInAddedSet( string& strRelpathName );
	bool IsAppendMode() {
		return !this->m_bNewLdb;
	}

	unsigned int GetAddedImages() {
		return this->m_count;
	}
protected:
	 bool ImageToDatum(cv::Mat& cv_img_origin, const int label,  const int height, const int width, caffe::Datum* datum) ;
	
public: //tools
	 static int CropPatchByKeyPointID( cv::Mat& image, cv::Mat& imagePatchCropped, vector<CvPoint2D32f>& vecPtKeyPoint, int nPatchID );
	 static int FillLicensePlate( cv::Mat& image, CvPoint2D32f& ptLicensePlateCenter, int iNewWidth );
	 static int GetStandarScaleImage( cv::Mat& image, cv::Mat& imageStandScale, vector<CvPoint2D32f>& vecKeyPoints, float INFLAT_COEFF, float NewWidth );

	 string& GetDataSetName() {
		 return this->m_strDataSetName;
	 }
	 string& GetDataSetVerName() {
		 return this->m_strDataSetVersion;
	 }
	 string& GetLDBName() {
		 return this->m_strDbName;
	 }
	 unsigned int GetHeritage(){
		 return this->m_NumHeritage;
	 }
private:
	
	int m_count;
	caffe::Datum m_datum;
	bool m_data_size_initialized;
	int m_data_size;
	leveldb::DB* m_db;
	leveldb::WriteBatch* m_batch ;

	string m_strLdbParentFolder;
	string m_strDataSetDataFolder;

	VMMRPPTID m_enPreprocessType;
	VMMRKPID m_enPatchID; 
	int m_iNewWidth;
	string m_strDataSetVersion;
	string m_strDataSetName;
	string m_strDbName;

	bool m_bNewLdb;
	unsigned int m_NumHeritage; //num of images already in ldb when append mode
	set<string> m_setImagesAdded;
};

string PatchKPIDToStr( int kpID );

string PreprocTypeIDToCode( int pptid );

int icvMkDir( const char* filename );

void TrimSpace( string& str );

int LoadKeypointsFromFileLab( string strAnnotFile,  
							 vector<CvPoint2D32f>& vecKeyPoints );

int FileOrFolderExist( string strFileOrFolder );

int SplitPathFileNameExt( string& strPathFileName, string& strDriver, string& strDir, string& strFileName, string& strExt );

bool compareNoCase(const string &strA,const string &strB);

#define random(x) (rand()%x)
int RandChoice( int N );


#define TRANSTYPE_NUM 4

typedef enum VMMR_AT{
	AUG_ADDNoise = 0,
	AUG_CHColor = 1,
	AUG_CHLight = 2,
	AUG_Smooth = 3
}VMMR_AT;

const VMMR_AT AugType[] = {AUG_ADDNoise, AUG_CHColor, AUG_CHLight, AUG_Smooth };


#define ADDNoise_suffix "N"
#define CHColor_suffix "C"
#define CHLight_suffix "D"
#define Smooth_suffix "S"

void AddNoise( const cv::Mat& image, cv::Mat& outImage );

void ChangeColor( const cv::Mat& image, cv::Mat& outImage );

void DarkenLight( const cv::Mat& image, cv::Mat& outImage );

void SmoothTexture(  const cv::Mat& image, cv::Mat& outImage );

int ConvertImageSet( string& strDataSetName, string& strListFile, vector<int>& vecPreprocTypeID, \
					vector<int>& vecPatchID, int NewWidth, int nAugNum, const string& strLdbName, int isSaveImage, bool bNewLdb, string strAppendList="" );

void _splitpath(const char *path, char *drive, char *dir, char *fname, char *ext);

static void _split_whole_name(const char *whole_name, char *fname, char *ext);

//#define _DEBUG_SHOW
