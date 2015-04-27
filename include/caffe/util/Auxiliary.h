#ifndef _AUXILIARY_H_
#define _AUXILIARY_H_

#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <fstream>
#include <list>
#include <unistd.h>

#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

using namespace std;

namespace VMMR{

#define _MAX_DRIVE 1024
#define _MAX_FNAME 2048
#define _MAX_LINE 4096
#define PATH_MAX 1024
#define _NUM_KEY_POINT_ 11

#define _GLOBAL_LOGGER_FILENAME_ "VMMR_Logger.txt"
#define _HOG_FEAT_FILE_EXT ".hog"
#define _LBP_FEAT_FILE_EXT ".lbp"
#define _IMAGE_FILE_EXT ".jpg"

extern ofstream ofGlobalLogger;

typedef enum FeatNormType{
  FN_Norm01 = 0,   // X_i = X_i / max_k=1( X_k )
  FN_L2 = 1,      // Unit length in high dim sapce
  FN_Norm01_L2 = 2    // first conduct Norm01, then L2
}FeatNormType;

typedef enum HistNormType{
  L2_NORM = 0,
  L2_HYS = 1,
  L1_NORM = 2,
  L1_SQRT = 3
}HistNormType;

typedef enum VMMRFeatDistType{
  L1 = 0,
  L2 = 1,
  COSINE = 2,
  CHISQR = 3,  
  INTERSECT = 4,
  UNKNOWN = -1
}VMMRFeatDistType;

typedef enum FeatureType{
  HOG=0,
  LBP=1,
  SURF=2,
  SIFT=3,
  CURVELET=4,
  DNN = 5
}FeatureType;

//Current classification type. Make or Make and model
typedef enum VRClassType{
  MAKE = 0,
  MAKE_MODEL = 1
}VRClassType;

typedef struct VMMName{
  string strMake;
  string strModel;
}VMMName;

typedef struct VMMGrounTruthTag{
  string strFileName;
  string strMake;
  string strModel;
}VMMGrounTruth;

typedef struct VFKeyPointTag{
  int index;
  CvPoint2D32f keyPoint;
}VFKeyPoint;

typedef struct ModelSet{
  string csModelName;   // make name or make model name
  vector<string> csFileNames;
}ModelSet;

typedef struct ImagePair{
  string csFileNameA;
  string csFileNameB;
}ImagePair;

typedef float (* CompareFeatDistFunc)( vector<float>&, vector<float>&, VMMRFeatDistType );

string strToLower(const string &str);

bool compareNoCase(const string &strA,const string &strB);

void TrimSpace( string& str );

long factorial(int num);

long pnm(int num, int len) ;

int icvMkDir( const char* filename );

unsigned int CountFileLineNum( string& strFilePathname );

int SplitPathFileNameExt( string& strPathFileName, string& strDriver, string& strDir, string& strFileName, string& strExt );

int FileOrFolderExist( string strFileOrFolder );

int InitGlobalLogger( string strLoggerFileName );

int CloseGlobalLogger();

int LoadGroundTruth( string& strGroundTruthFile, map<string, VMMGrounTruth>& mapVmmGroundTruth );

int LoadClassLabelDict( string& strClassLabelDictFile, VRClassType enVRClassType, map<int, VMMName>& mapLabelMakeModel );

int ReadRelpathFileNameFromListFile( string& strListFile, list<string>& lstRelpathFilenames );

int LoadRandSelPairListFromFile( string strFileName, 
				 int nTotalNum, 								 
				 unsigned int nRandSelected,
				 vector<ImagePair>& vecPairs );

int LoadPairListFromFile( string strFileName, 
			  list<ImagePair>& lstPairs );

int ReadAllModelSetFromList( string strListFile, 
			     VRClassType enVRClassType,   //make or make_model
			     list<ModelSet>& lstAllModelSet, 
			     list<string>& lstRelPathFileNames );

// create ModelSet vector
// Add files to into make_model 
int ReadAllModelSetFromList( string strListFileName, 
			     VRClassType enVRClassType,   //make or make_model
			     vector<ModelSet>& vecAllModelSet, 
			     vector<string>& vecFileNames );

int CreateIntraExtraPair( vector<ModelSet>& vecAllModelSet, 
			  list<ImagePair>& vecIntraPairs, 
			  list<ImagePair>& vecExtraPairs,
			  bool bPushToList = true,
			  string csIntravectorFile="", 
			  string csExtravectorFile="" );

int RandomSelectExtraPair( vector<ImagePair>& vecExtraPairs,
			   unsigned int numSelected,
			   vector<ImagePair>& vecSelectedExtraPairs );

float CalcIntraExtraDistHistDist( vector<float>& vecIntraDists, 
				  vector<float>& vecExtraDists, 
				  float& fThreshold, 
				  float& fFalsePosRate, 
				  int method,
				  string strIntraExtraHistImageFile  );

int SaveIntraExtraDists( vector<float>& vecIntraDists, 
			 vector<float>& vecExtraDists, 
			 string csIntraDistFile, 
			 string csExtraDistFile );

int SaveFloatVecFeature( string strFileName, 
			 vector<float>& floatVecFeature );

int LoadFloatVecFeature( string csFileName, 
			 vector<float>& floatVecFeature );

int LoadFloatVecFeaturesFromFiles( vector<string> vecFileNames, 
				   string strFloatVecFeatSaveFolder, 
				   int nFeatDims, 
				   map<string, vector<float> >& mapAllNamedHOGFeats );

int LoadKeypointsFromFile( string strAnnotFile,  
			   vector<VFKeyPoint>& vecKeyPoints );

int LoadKeypointsFromFileLab( string strAnnotFile,  
			      vector<CvPoint2D32f>& vecKeyPoints );

int CalcIntraExtraDistances( map<string, vector<float> >& mapAllNamedHOGFeats,
			     vector<ImagePair>& vecIntraPairs, 
			     vector<ImagePair>& vecExtraPairs, 
			     vector<float>& vecIntraDists, 
			     vector<float>& vecExtraDists, 
			     CompareFeatDistFunc ptrCompareFunc,
			     VMMRFeatDistType method,
			     unsigned int& nValidIntraDist,
			     unsigned int& nValidExtraDist );


float CompareGeneralFeatDistance( vector<float>& vecGeneralFeaturesA, vector<float>& vecGeneralFeaturesB, VMMRFeatDistType dType );

int NormalizeHist( vector<float>& vecHistFeat, HistNormType enuNormType );

int NormalizeFeat( vector<float>& vecFeature, FeatNormType enFeatNormType );

// A list file line may have the following two format:
/* relpath/<filename>
   relpath/<filename>  <label>
   only one level path contained
   there is no any space in  relpath/<filename>.
   relpath is only one level.
   
   string& strPath: has end slash!
*/
int ParseListFileLine( string& strLine, string& strPath, string& strFileName, int& iLabel );

string PatchKPIDToStr( int kpID );

bool IsPathEndWithSlash( string& strPath );



void _splitpath(const char *path, char *drive, char *dir, char *fname, char *ext);

#endif //_AUXILIARY_H_

} //end VMMR namespace
