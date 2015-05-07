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


#include "convert_imagesetex.h"

using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

#define Round(a) (int(a+0.5))

int GetIntegers( string& strColonSepInString, vector<int>& vecIDs ) {
  vecIDs.clear();

  size_t nPosTag = strColonSepInString.find_first_of( ":" );

  if( nPosTag == string::npos ) {
    vecIDs.push_back( atoi( strColonSepInString.c_str() ) );
  } else {
    vecIDs.push_back( atoi( strColonSepInString.substr( 0, nPosTag ).c_str() ) );

    string strRemain = strColonSepInString.substr( nPosTag + 1 );
    nPosTag = strRemain.find_first_of( ":" );
    while( nPosTag != string::npos ) {
      vecIDs.push_back( atoi( strRemain.substr( 0, nPosTag ).c_str() ) );
      strRemain = strRemain.substr( nPosTag + 1 );
      nPosTag = strRemain.find_first_of( ":" );
    }
    vecIDs.push_back( atoi( strRemain.c_str() ) );
  }

  return 0;
}

char szVersionInfo[_MAX_LINE*2];

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 8 || argc > 9 ) {
    printf("Convert a set of images to the leveldb format used\n"
	   "as input for Caffe.\n"
	   "Usage:\n"
	   "    convert_imagesetex [1]DataSetNameVer(eg. V0) [2]PreprocTypeIDs(sep by\":\") [3]PatchIDs(sep by \":\") [4]NewWidth"
           "                       [5]AugmentNum(<AugTrain:AugTest, 0 denote not augmenting, the resulted num is 1+$AugmetNum>) "
           "                       [6]FuncCode( TestOrTrain(0:testonly,1:tainonly,2: both) [7]IsNewLdb <8>[<append list>]\n" 
          );
    return 1;
  }

  memset( szVersionInfo, 0, _MAX_LINE*2 );
  strcpy( szVersionInfo, "[<Version: 2015-03-20 13:54>]"  );

  cout << endl << szVersionInfo << endl << endl;

  bool bNewLdb= true; //default is create new one.
  string strDataSetVer = argv[1];
  string strPreprocTypeIds = argv[2];
  string strPatchIds = argv[3];
  string strNewWidth = argv[4];
  string strTrainTestAugNum = argv[5];
  string strFuncCode = argv[6];
  string strIsNewLdb = argv[7];

  int FuncCode = atoi( strFuncCode.c_str() );		
  int iIsNewLdb = atoi( strIsNewLdb.c_str() );
  if( iIsNewLdb > 0 ) {
    bNewLdb = true;
  }else {
    bNewLdb = false;
  }

  string strAppendList = "";
  if( argc > 8 ) {
    strAppendList = argv[8];
    if( FileOrFolderExist( strAppendList ) < 0 ) {
      std::cout << "The append list not exist: " << strAppendList << endl;
      return -1;
    }
  }

  if( FuncCode < 0 || FuncCode > 2 ) {
    std::cout << "Invalid function code : " << FuncCode << endl;
    std::cout << "0 : Test LDB converstion only !" << endl;
    std::cout << "1 : Train LDB converstion only !" << endl;
    std::cout << "2 : Both Test and Train LDB converstion only !" << endl;
    return -7;
  }
  string strDataSetName = strDataSetVer + "_" + VMMR_USEDALIGN + "Color";

  string strDataSetDataDir = VMMR_DATA_TOPDIR + ossep + strDataSetName + ossep + strDataSetName + ossep;
  string strTrainList = VMMR_DATA_TOPDIR + ossep + strDataSetName + ossep + VMMR_LIST_FOLDER + ossep + VMMR_TRAIN_LIST;
  string strTestList  = VMMR_DATA_TOPDIR + ossep + strDataSetName + ossep + VMMR_LIST_FOLDER + ossep + VMMR_TEST_LIST;
  if( FileOrFolderExist( strDataSetDataDir ) < 0 ) {
    std::cout << endl << "Data set data dir not exist: " << strDataSetDataDir << endl;
    std::cout << "Please check data set: " << strDataSetName << endl << endl;
  }
  if(  FileOrFolderExist( strTestList ) < 0 ) {
    std::cout << endl << "Test list (before aug) not exist: " << strTestList << endl << endl;
    std::cout << "Please check data set: " << strDataSetName << endl << endl;
  } 

  if(  FileOrFolderExist( strTrainList ) < 0 ) {
    std::cout << endl << "Train list (before aug) not exist: " << strTrainList << endl << endl;
    std::cout << "Please check data set: " << strDataSetName << endl << endl;
  }
	
	
  vector<int> vecPreprocTypeID;
  vector<int> vecPatchID;
  vector<int> vecAugNum;
  GetIntegers( strPreprocTypeIds, vecPreprocTypeID );
  GetIntegers( strPatchIds, vecPatchID );
  GetIntegers( strTrainTestAugNum, vecAugNum );

  if( vecAugNum.size() != 2 ) {
    std::cout << "The number of Augment numbers is not two ! but is : " << vecAugNum.size() << endl;
    return -3;
  }
  if( vecAugNum[0] > TRANSTYPE_NUM || vecAugNum[1] > TRANSTYPE_NUM ) {
    cout << "Augment num of train or test cannot large than the number of transform kinds : " << TRANSTYPE_NUM <<endl;
    cout << "This restriction is to assure no repeated images in augmented image sets!" << endl;
    return -5;
  }

  int NewWidth = atoi( strNewWidth.c_str() );

  std::cout << "DataSet name     : " << strDataSetName << endl;
  std::cout << "TrainList file   : " << strTrainList << endl;
  std::cout << "TrainList file   : " << strTestList << endl;
  std::cout << "Preprocess type ids : " << "[ ";
  for( int n = 0; n < vecPreprocTypeID.size(); n ++ ) {
    std::cout << vecPreprocTypeID[n] << ", ";
  }
  std::cout << " ]" << endl;
  std::cout << "Patch ids           : " << "[ ";
  for( int n = 0; n < vecPatchID.size(); n ++ ) {
    std::cout << vecPatchID[n] << ", ";
  }
  std::cout << " ]" << endl;
  std::cout << "Newwidth :" << NewWidth << endl;
  std::cout << "Augment number for train set is : " << vecAugNum[0] << endl;
  std::cout << "Augment number for test set is  : " << vecAugNum[1] << endl;
  std::cout << "Is test images saved: " << IS_TESTSAVE << endl;
  std::cout << "Is train image saved: " << IS_TRAINSAVE << endl;
  std::cout << "Function code : " << FuncCode << endl;
  std::cout << "Is create new LDB : " << ( bNewLdb? " Yes " : " No ") << endl;
  std::cout << "Append list : " << strAppendList << endl;

  //if( !compareNoCase( strLdbName, TRAIN_LDB ) && !compareNoCase( strLdbName, TEST_LDB )  ){
  //	cout << "LDB Name can only be : " << TRAIN_LDB << " or " << TEST_LDB << endl;
  //	return -3;
  //}

  int totalTrain = 0, totalTest = 0;

  try{
    if( FuncCode == 0 || FuncCode == 2 ) {
      //The TEST part. For TEST, Save images!
      std::cout << endl << endl << "Start Test LDB augment and convert ... " << endl;
      totalTest = ConvertImageSet( strDataSetName, strTestList, vecPreprocTypeID, vecPatchID, NewWidth, vecAugNum[1], TEST_LDB, IS_TESTSAVE, bNewLdb, strAppendList );

      if( totalTest < 0 ) {
	cout << "Error happend ! Please check and run again." << endl;
	return -2;
      }
      std::cout << endl << "Complete test ldb augment and convert." << endl;
      std::cout << "Totally " << totalTest << " images are put into test ldb " << endl << endl;
    }

    if( FuncCode == 1 || FuncCode == 2 ) {
      //The train parts. For Train Don't save images!
      std::cout << endl << endl << "Start Train LDB augment and convert ... " << endl;
      totalTrain = ConvertImageSet( strDataSetName, strTrainList, vecPreprocTypeID, vecPatchID, NewWidth, vecAugNum[0], TRAIN_LDB, IS_TRAINSAVE, bNewLdb, strAppendList );

      if( totalTrain < 0 ) {
	cout <<  "Error happend ! Please check and run again." << endl;
	return -2;
      }

      std::cout << endl << "Complete train ldb augment and convert." << endl;
      std::cout << "Totally " << totalTrain << " images are put into train ldb " << endl << endl;
    }
  }
  catch(...) {
    std::cout << "This is Exception happened ! " << endl;
  }
	
  std::cout << endl << endl << "Train and test image-ldb conversion completed!" << endl;
  std::cout << "Totally " << totalTest << " images are converted into test ldb." <<endl;
  std::cout << "Totally " << totalTrain << " images are converted into train ldb." <<endl << endl;
  return 0;
}

//#define FING_BUG
//nAugNum means one image augmented to how many transformed images.
//So original N images well become (1+nAugNum) * N images
//
int ConvertImageSet( string& strDataSetName, string& strListFile, vector<int>& vecPreprocTypeID, \
		     vector<int>& vecPatchID, int NewWidth, int nAugNum, const string& strLdbName, int isSaveImage, bool bNewLdb, string strAppendList )
{
  size_t nNamePos = strListFile.find( strDataSetName );
  if( nNamePos == string::npos ) {
    std::cout << "The list dir don't contain data set name !" << endl;
    std::cout << "The list file should from the same DataSet folder " << endl;
    return -4;
  }

  time_t t = time(NULL); 
  char szCurDateTime[_MAX_LINE];
  std::memset( szCurDateTime, 0, _MAX_LINE );
  strftime( szCurDateTime, sizeof(szCurDateTime), "%Y/%m/%d %X %A",localtime(&t) ); //File name suffix
  std::cout << endl << "System Date and time: " << szCurDateTime << endl << endl;

  std::cout << "++++++++++++++++++++++++++++" << endl;
  std::cout << "Current Augment num : " << nAugNum << endl;
  std::cout << "Is save image       : " << isSaveImage << endl;
  std::cout << "++++++++++++++++++++++++++++" << endl;

  string strColorDataSetDataFolder = VMMR_DATA_TOPDIR + ossep + strDataSetName + ossep + strDataSetName + ossep;

  string strDriver, strDir, strFileName, strExt;
  SplitPathFileNameExt( strListFile, strDriver, strDir, strFileName, strExt );
  
  std::memset( szCurDateTime, 0, _MAX_LINE );
  strftime( szCurDateTime, sizeof(szCurDateTime), "%Y%m%d_%H%M%S",localtime(&t) ); 
  string strCurDateTime = szCurDateTime;
  string strAugListFile     =  strDir + ossep + VMMR_AUG_LIST_PREFIX + "_" + strFileName + "_" +  strCurDateTime + strExt;
  string strDoneListFile    =  strDir + ossep + VMMR_DONE_LIST_PREFIX + "_" + strFileName + "_" + strCurDateTime + strExt; 
  string strRandShuffleList =  strDir + ossep + VMMR_RANDSHUFFLE_LIST_PREFIX + "_" + strFileName + "_" + strCurDateTime + strExt;
  string strExceptInfoFile  =  strDir + ossep + VMMR_EXCEPTION_PREFIX + "_" +  strFileName + "_" + strCurDateTime + strExt;
  string strLogFile         = VMMR_DATA_TOPDIR + ossep + strDataSetName + ossep + \
    strLdbName + "_" + DBCONV_LOG +  + "_" + strCurDateTime + ".txt";

  ofstream ofLogfile( strLogFile.c_str(), ios::out );
  if( !ofLogfile.is_open() ) {
    std::cout << "Open log file failed: " << strLogFile << endl;
    return -2;
  }

  std::memset( szCurDateTime, 0, _MAX_LINE );
  strftime( szCurDateTime, sizeof(szCurDateTime), "%Y/%m/%d %X %A",localtime(&t) ); //File name suffix

  ofLogfile << endl << szVersionInfo << endl << endl;
  ofLogfile << endl << szCurDateTime << endl;

  ofstream ofAugListFile( strAugListFile.c_str(), ios::out );
  if( !ofAugListFile.is_open() ) {
    std::cout << "Open auglist file failed: " << strAugListFile << endl;
    return -3;
  }
  ofstream ofDoneListFile;
  ofDoneListFile.open( strDoneListFile.c_str(), ios::out|ios::app );
  if( !ofDoneListFile.is_open() ) {
    std::cout << "Open done list file failed : " << strDoneListFile << endl;
    return -4;
  }

  bool bWithAppendList = false;
  if( ( bNewLdb == false ) && ( strAppendList.empty() == false ) ) {
    std::cout << "Work in append mode and append list is provided: " << strAppendList << endl;
    std::cout << "We just substitue original list with the input append list and not create in ldb image set" << endl;
    strListFile = strAppendList;
    bWithAppendList = true;
  }
  std::ifstream infile( strListFile.c_str() );
  std::vector<std::pair<string, int> > lines;
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }

  // randomly shuffle data
  LOG(INFO) << "Shuffling data";
  std::cout << "Shuffling data";
  std::random_shuffle(lines.begin(), lines.end());
  std::cout << endl << "++++++++++++++++++++++++++++" << endl;
  LOG(INFO) << "A total of " << lines.size() << " images.";
  std::cout << "A total of " << lines.size() << " images.";
  std::cout << endl << "++++++++++++++++++++++++++++" << endl << endl;

  std::cout << "Save rand_shuffle list to: " << strRandShuffleList << endl;
  std::cout << "You can recovery conversion with this file and line_id of breaking." << endl << endl;
  std::ofstream ofRandShuffleList( strRandShuffleList.c_str(), ios::out );
  for( int k = 0; k < lines.size(); k ++ ) {
    ofRandShuffleList << lines[k].first << " " << lines[k].second << endl;
  }
  ofRandShuffleList.close();

  std::cout <<endl << "Create Exception info file: " << strExceptInfoFile << endl;
  std::ofstream ofExceptInfo( strExceptInfoFile.c_str(), ios::out );
  if( !ofExceptInfo.is_open() ) {
    cout << "Create exception info file failed: " << strExceptInfoFile << endl;
  }

  int count = 0;

  //leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  //Create ldbs:
  map<string, VmmrLevelDB*> mapVmmrLdbs;
  char szKeyVmmrLdbName[_MAX_LINE];
  for( int i = 0; i < vecPreprocTypeID.size(); i ++ ) {
    for( int j = 0; j < vecPatchID.size(); j ++ ) {
      sprintf( szKeyVmmrLdbName, "%d_%d_%d", vecPreprocTypeID[i], vecPatchID[j], NewWidth );
      VMMRPPTID _nppt = (VMMRPPTID)vecPreprocTypeID[i];
      VMMRKPID _np = (VMMRKPID)vecPatchID[j];
      VmmrLevelDB* pVmmrLdb = new VmmrLevelDB( strDataSetName, strLdbName, _nppt, _np, NewWidth, bNewLdb, bWithAppendList );
      mapVmmrLdbs[szKeyVmmrLdbName] = pVmmrLdb;
    }
  }

  srand((unsigned)time(NULL)); 
  int numOriginal = 0;
  int numDarken = 0;
  int numChColor = 0;
  int numAddNoise = 0;
  int numSmooth = 0;
	
  std::cout << endl;
  int m = 0; //augment index
  vector<VMMR_AT> vecVmmrAugType( AugType, AugType+TRANSTYPE_NUM );
  int total_lines = lines.size();

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    try{
      string strImagePathName = strColorDataSetDataFolder + lines[line_id].first;
      int label = lines[line_id].second;
      std::cout << "\r" << "Process image: " << setiosflags( ios::left) << setw( 50 ) << lines[line_id].first \
		<< " [ LineId: " << setiosflags( ios::left) << setw( 10 ) << line_id << "/" << total_lines << "]" \
		<< " Label: " << setiosflags( ios::left) << setw( 5 ) << label;	

      //first time check, if all ldb already added, pass!
      if( ( bNewLdb == false) && ( strAppendList.empty() == true )  ) { //append mode and no append list provided, then check by Image Set:
	bool bAddedInAll = true;
	for( int nppt = 0; nppt < vecPreprocTypeID.size(); nppt ++ ) {
	  for( int np = 0; np < vecPatchID.size(); np ++ ) {
	    sprintf( szKeyVmmrLdbName, "%d_%d_%d", vecPreprocTypeID[nppt], vecPatchID[np], NewWidth );
	    VmmrLevelDB* pVmmrLdb = mapVmmrLdbs[szKeyVmmrLdbName];
	    if( pVmmrLdb == NULL ) {
	      std::cout << endl << "Get vmmr ldb pointer failed : " << szKeyVmmrLdbName << endl;
	    }

	    if( pVmmrLdb->IsRelpathNameInAddedSet( lines[line_id].first ) == false ) { //check name before tansform augment already in added set.
	      // there is one hasn't add this image
	      bAddedInAll = false;
	      break;
	    }
	  }
	  if( bAddedInAll == false ) {
	    //has one hasn't added this image
	    break;
	  }
	}
	if( bAddedInAll == true ){
	  //added in all LDB, so pass directly!
	  std::cout << endl << "Already added in ldb, pass... " << endl;
	  continue;
	}				
      }
			
      string strImageAnnotPathName = strImagePathName.substr( 0, strImagePathName.find_last_of( "." ) );
      strImageAnnotPathName += ".lab";

      //check whether image exist?
      if( FileOrFolderExist( strImagePathName ) < 0 ) {
	std::cout << endl << "Warning: File not exist: " << strImagePathName << endl;
				
	ofExceptInfo << "Image file not exist: " << lines[line_id].first << " " << label << endl;

	continue;
      }

      cv::Mat orig_image = imread( strImagePathName, CV_LOAD_IMAGE_UNCHANGED );
      if( orig_image.empty() ){
	std::cout << endl << "Error! Load image failed: " << strImagePathName << endl;
	std::cout << "Please check. (Exception Info file ) " << endl;

	ofExceptInfo << "Failed ot load image: " << lines[line_id].first << " " << label << endl;

	continue;
      }
      if( orig_image.channels() != 3 ) {
	std::cout << endl << "WARNING: The input image should be 3-channel color image ! nChannels = " << orig_image.channels() <<  endl;
	std::cout << "WARNING: Roll back: Force loading as color image use CV_LOAD_IMAGE_COLOR" << endl;
	orig_image = imread( strImagePathName, CV_LOAD_IMAGE_COLOR );
	if( orig_image.empty() ){
	  std::cout << endl << "Error! Load image failed: " << strImagePathName << endl;
	  std::cout << "Please check. (Exception Info file ) " << endl;

	  ofExceptInfo << "Failed ot load image: " << lines[line_id].first << " " << label << endl;

	  continue;
	}

	ofExceptInfo << "CV read it as not color image. Force color image loading: " << lines[line_id].first << " " << label << endl;
      }

      vector<CvPoint2D32f> vecKeyPoints;
      int nRetVal = LoadKeypointsFromFileLab(strImageAnnotPathName,  vecKeyPoints );
      if( nRetVal < 0 ) {
	std::cout << endl << "Read annot file error: " << strImageAnnotPathName << endl;

	ofExceptInfo << "Load key points from file failed: " << lines[line_id].first << " " << label << endl;

	continue;
      }

      m = 0;
      random_shuffle( vecVmmrAugType.begin(), vecVmmrAugType.end() ); //random_shuffle for each line in list file.			
      for( m = 0; m < (1+nAugNum); m ++ ) {
	cv::Mat image;
	string strCurrRelpathName;
	if( m == 0 ) {
	  image = orig_image.clone();
	  strCurrRelpathName = lines[line_id].first;
	  TrimSpace( strCurrRelpathName );
	  numOriginal ++;
	} else {
	  //Augment:
	  strCurrRelpathName = lines[line_id].first;
	  TrimSpace( strCurrRelpathName );
	  string strRelpathNameNoExt = strCurrRelpathName.substr( 0, strCurrRelpathName.find_last_of("." ) );
	  string strDotExt = strCurrRelpathName.substr( strCurrRelpathName.find_last_of("." ) );
	  char szAugIndex[256];
	  sprintf( szAugIndex, "%d", m );
	  string strAugIndex = szAugIndex;

	  int _TransType = vecVmmrAugType[(m-1)%TRANSTYPE_NUM];

	  if(  _TransType == AUG_ADDNoise ) {						
	    AddNoise( orig_image, image );
	    strCurrRelpathName = strRelpathNameNoExt + "_" + ADDNoise_suffix + 	strAugIndex + strDotExt;
	    numAddNoise ++;
	  } else if ( _TransType ==  AUG_CHColor ) {						
	    ChangeColor( orig_image, image );
	    strCurrRelpathName = strRelpathNameNoExt + "_" + CHColor_suffix + 	strAugIndex + strDotExt;
	    numChColor ++;	
	  }  else if ( _TransType == AUG_CHLight ) {						
	    DarkenLight( orig_image, image );
	    strCurrRelpathName = strRelpathNameNoExt + "_" + CHLight_suffix + 	strAugIndex + strDotExt;
	    numDarken ++;	
	  } else if ( _TransType == AUG_Smooth ) {						
	    SmoothTexture( orig_image, image );
	    strCurrRelpathName = strRelpathNameNoExt + "_" + Smooth_suffix + 	strAugIndex + strDotExt;
	    numSmooth ++;
	  } else {
	    std::cout <<  endl << "ERROR: Unknown transform type id : " << _TransType << endl;

	    ofExceptInfo << "ERROR: Unknown transform type id happened: " << lines[line_id].first << " " << label << endl;
	    ofExceptInfo << "Augment Index: " << m << endl << endl;

	    continue;
	  }
	}

	//Notice: 
	// 1) preprocessing is done on original big color image. So we should do this samely with same sequence. Especially for HE:
	// 2) Input image should always be original big color image.
	cv::Mat grayImage1ch, grayImage, equalHistImage1ch, equalHistImage, equalHistColorImage;
	cv::Mat imageStandScaleColor, imageStdScaleGray, imageStdScaleEqualHist, imageStdScaleEqualHistColor;
	vector<CvPoint2D32f> vecKeyPointsScaled( vecKeyPoints.size() );
	
	//for color image		
	std::copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	nRetVal = VmmrLevelDB::GetStandarScaleImage( image, imageStandScaleColor, vecKeyPointsScaled, VMMR_INFLAT_COEFF, NewWidth );
	if( nRetVal < 0 ) { 
	  std::cout << endl <<  "Error: GetStandarScaleImage for  imageStandScaleColor" << endl; 
	  
	  ofExceptInfo << "Color image GetStandarScaleImage failed: " << lines[line_id].first << " " << label << endl;
	  ofExceptInfo << "Augment ID: " << m << endl << endl;
	  
	  continue;
	}
	//Fill license plat area
	VmmrLevelDB::FillLicensePlate( imageStandScaleColor, vecKeyPointsScaled[KP_LicensePC], NewWidth );
	
	//for gray image:
	cv::cvtColor( image, grayImage1ch, CV_BGR2GRAY );     //conver to 1 channel gray
	cv::cvtColor( grayImage1ch, grayImage, CV_GRAY2BGR ); //Convert three channel to feed dnn
	std::copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	nRetVal = VmmrLevelDB::GetStandarScaleImage( grayImage, imageStdScaleGray, vecKeyPointsScaled, VMMR_INFLAT_COEFF, NewWidth );
	if( nRetVal < 0 ) { 
	  std::cout << endl << "Error: GetStandarScaleImage for  imageStdScaleGray" << endl; 
	  
	  ofExceptInfo << "Gray image GetStandarScaleImage failed: " << lines[line_id].first << " " << label << endl;
	  ofExceptInfo << "Augment ID: " << m << endl << endl;
	  
	  continue;
	}
	//Fill license plat area
	VmmrLevelDB::FillLicensePlate( imageStdScaleGray, vecKeyPointsScaled[KP_LicensePC], NewWidth );
	
	//for equal hist image:
	cv::equalizeHist( grayImage1ch, equalHistImage1ch );
	cv::cvtColor( equalHistImage1ch, equalHistImage, CV_GRAY2BGR );//Convert three channel to feed dnn
	std::copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	nRetVal = VmmrLevelDB::GetStandarScaleImage( equalHistImage, imageStdScaleEqualHist, vecKeyPointsScaled, VMMR_INFLAT_COEFF, NewWidth );
	if( nRetVal < 0 ) { 
	  std::cout << endl << "Error: GetStandarScaleImage for imageStdScaleEqualHist" << endl; 
	  
	  ofExceptInfo << "Equal Hist image GetStandarScaleImage failed: " << lines[line_id].first << " " << label << endl;
	  ofExceptInfo << "Augment ID: " << m << endl << endl;
	  
	  continue;
	}
	//Fill license plat area
	VmmrLevelDB::FillLicensePlate( imageStdScaleEqualHist, vecKeyPointsScaled[KP_LicensePC], NewWidth );

	//for equal hist color image
	vector<cv::Mat> vecImage1Ch;
	cv::split( image, vecImage1Ch );
	cv::equalizeHist( vecImage1Ch[0], vecImage1Ch[0] );
	cv::equalizeHist( vecImage1Ch[1], vecImage1Ch[1] );
	cv::equalizeHist( vecImage1Ch[2], vecImage1Ch[2] );
	cv::merge( vecImage1Ch, equalHistColorImage );
	std::copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	nRetVal = VmmrLevelDB::GetStandarScaleImage( equalHistColorImage, imageStdScaleEqualHistColor, vecKeyPointsScaled, VMMR_INFLAT_COEFF, NewWidth );
	if( nRetVal < 0 ) { 
	  std::cout << endl << "Error: GetStandarScaleImage for imageStdScaleEqualHist" << endl; 
	  
	  ofExceptInfo << "Equal Hist Color image GetStandarScaleImage failed: " << lines[line_id].first << " " << label << endl;
	  ofExceptInfo << "Augment ID: " << m << endl << endl;
	  
	  continue;
	}
	//Fill license plat area
	VmmrLevelDB::FillLicensePlate( imageStdScaleEqualHistColor, vecKeyPointsScaled[KP_LicensePC], NewWidth );

	for( int i = 0; i < vecPreprocTypeID.size(); i ++ ) {
	  for( int j = 0; j < vecPatchID.size(); j ++ ) {

	    sprintf( szKeyVmmrLdbName, "%d_%d_%d", vecPreprocTypeID[i], vecPatchID[j], NewWidth );
	    VmmrLevelDB* pVmmrLdb = mapVmmrLdbs[szKeyVmmrLdbName];
	    if( pVmmrLdb == NULL ) {
	      std::cout << endl << "Get vmmr ldb pointer failed : " << szKeyVmmrLdbName << endl;

	      ofExceptInfo << "Get VMMR LDB pointer failed! " << endl;
	      ofExceptInfo << lines[line_id].first << " " << label << endl;
	      ofExceptInfo << "Augment Index: " << m << "PreprotTypeID: " << vecPreprocTypeID[i] << "Patch ID: " << vecPatchID[j] << endl;
	      ofExceptInfo << endl;

	      continue;
	    }

	    //second time check
	    if( ( bNewLdb == false) && ( strAppendList.empty() == true ) ) { //append mode and no append list, then check by AddedSet:
	      if( pVmmrLdb->IsRelpathNameInAddedSet( lines[line_id].first ) == true ) { //check name before tansform augment already in added set.
		// there is one hasn't add this image
		continue;
	      }
	    }

	    cv::Mat imagePatchCropped;
	    int RetVal = -1;
	    if( vecPreprocTypeID[i] == PT_COLOR ) {
	      RetVal = VmmrLevelDB::CropPatchByKeyPointID( imageStandScaleColor, imagePatchCropped, vecKeyPointsScaled, vecPatchID[j] );
	    } else if( vecPreprocTypeID[i] == PT_GRAY ) {
	      RetVal = VmmrLevelDB::CropPatchByKeyPointID( imageStdScaleGray, imagePatchCropped, vecKeyPointsScaled, vecPatchID[j] );
	    } else if( vecPreprocTypeID[i] == PT_EQUALIZEHIST ) {
	      RetVal = VmmrLevelDB::CropPatchByKeyPointID( imageStdScaleEqualHist, imagePatchCropped, vecKeyPointsScaled, vecPatchID[j] );
	    } else if( vecPreprocTypeID[i] == PT_EQUALIZEHISTCOLOR ){
	      RetVal = VmmrLevelDB::CropPatchByKeyPointID( imageStdScaleEqualHistColor, imagePatchCropped, vecKeyPointsScaled, vecPatchID[j] );
	    } else{
	      std::cout << endl << "Error: Unsupported preprocessing type appread !" << "  Line: " << __LINE__ << endl;
	      return -5;
	    }
	    
	    if( RetVal < 0 ) {
	      std::cout << endl << "Crop patch by key point id failed. Patch area may not fully in Padding image. Pass it !" << endl;
	      
	      ofExceptInfo << "Crop patch by key point id failed. Patch area may not fully in Padding image. Pass it ! " << endl;
	      ofExceptInfo << lines[line_id].first << " " << label << endl;
	      ofExceptInfo << "Augment Index: " << m << "PreprotTypeID: " << vecPreprocTypeID[i] << "Patch ID: " << vecPatchID[j] << endl;
	      ofExceptInfo << endl;
	      
	      continue;
	    }

	    if( imagePatchCropped.empty() || imagePatchCropped.channels() != 3 ) {
	      std::cout << endl << "imagePatchCropped empty or wrong channel number ( " <<  imagePatchCropped.channels() << " ) " << endl;
	      std::cout << "Relpath image: " << strCurrRelpathName << endl;
	      std::cout << "Preprocess type: " << PreprocTypeIDToCode( vecPreprocTypeID[i] ) << "Patch ID: " << PatchKPIDToStr( vecPatchID[j] ) << endl;

	      ofExceptInfo << "imagePatchCropped empty or wrong channel number: " << lines[line_id].first << " " << label << endl;
	      ofExceptInfo << "Augment Index: " << m << "PreprotTypeID: " << vecPreprocTypeID[i] << "Patch ID: " << vecPatchID[j] << endl;
	      ofExceptInfo << endl;

	      continue;
	    }

#ifdef _DEBUG_SHOW
	    char szTitle[1024];
	    sprintf( szTitle, "%s_%s_%d", PreprocTypeIDToCode( vecPreprocTypeID[i]).c_str(), PatchKPIDToStr( vecPatchID[j] ).c_str(), NewWidth );
	    cv::imshow( szTitle, imagePatchCropped );
	    waitKey();
#endif//_DEBUG_SHOW
	    pVmmrLdb->PushData( imagePatchCropped, line_id, strCurrRelpathName, label );
	    pVmmrLdb->WriteData();

	    if( isSaveImage > 0 ) {
	      string strDataSetDataFolder = pVmmrLdb->GetImageDataPath();
	      string strImageSavePathName = strDataSetDataFolder + strCurrRelpathName;

	      //check whether need to mkdir:
	      string strDriver, strDir, strFileName, strExt;
	      SplitPathFileNameExt( strImageSavePathName, strDriver, strDir, strFileName, strExt );
	      string strImageSavePath = strDriver + strDir + "/";
	      if( FileOrFolderExist( strImageSavePath ) < 0 ) {
		icvMkDir( strImageSavePath.c_str() );
	      }

	      imwrite( strImageSavePathName, imagePatchCropped );
	    }
	  }
	} //preprocess type and patch id cycle.

	count ++;

	//Add to Auglist file:
	ofAugListFile << strCurrRelpathName << " " << label << endl;
	if( count % 10000 == 0 ) {
	  t = time(NULL); 
	  std::memset( szCurDateTime, 0, _MAX_LINE );
	  strftime( szCurDateTime, sizeof(szCurDateTime), "%Y/%m/%d %X %A ",localtime(&t) ); 
	  ofLogfile << szCurDateTime << ": Converted " << count << " image into ldb." << endl;
	}
				
      }//end augment cycle

      ofDoneListFile << lines[line_id].first << " " << label << endl;

    } //end all images cycle
    catch(Exception e) {
      std::cout << endl << "Error happened. Exception process: " << __LINE__ << endl;
      std::cout << "Exception Msg: " << e.msg << endl;
      std::cout << "Exception Line: " << e.line << endl;
      std::cout << "Exception Func: " << e.func << endl;
      std::cout << "Exception Error: " << e.err << endl;
      std::cout << "Exception File: " << e.file << endl;
      std::cout << "line_id : " << line_id << endl;
      std::cout << "lines[line_id].first : "  << lines[line_id].first << endl;
      std::cout << "Augment index : " << m << endl;
      std::cout << endl << endl;

      ofExceptInfo << lines[line_id].first << " " << label << "  Line ID: " << line_id << endl;
    }
    catch (...){
      std::cout << endl << "Unknown Error happened. Exception process: " << __LINE__ << endl;
      std::cout << "line_id : " << line_id << endl;
      std::cout << "lines[line_id].first : "  << lines[line_id].first << endl;
      std::cout << "Augment index : " << m << endl;
      std::cout << endl << endl;

      ofExceptInfo << lines[line_id].first << " " << label << "  Line ID: " << line_id << endl;
    }		
  } // end of lines for cycle.

  std::cout << endl;
  ofAugListFile.close();
  ofDoneListFile.close();
  ofExceptInfo.close();

  for( int i = 0; i < vecPreprocTypeID.size(); i ++ ) {
    for( int j = 0; j < vecPatchID.size(); j ++ ) {
      sprintf( szKeyVmmrLdbName, "%d_%d_%d", vecPreprocTypeID[i], vecPatchID[j], NewWidth );
      VmmrLevelDB* pVmmrLdb = mapVmmrLdbs[szKeyVmmrLdbName];
      pVmmrLdb->WriteDataEnd();	
    }
  }
	
  std::cout << "convert image set in " << ( bNewLdb == true ? " Create New " : " Append" ) << " Mode" << endl;
  ofLogfile << "convert image set in " << ( bNewLdb == true ? " Create New " : " Append" ) << " Mode" << endl;
  for( int i = 0; i < vecPreprocTypeID.size(); i ++ ) {
    for( int j = 0; j < vecPatchID.size(); j ++ ) {
      sprintf( szKeyVmmrLdbName, "%d_%d_%d", vecPreprocTypeID[i], vecPatchID[j], NewWidth );
      VmmrLevelDB* pVmmrLdb = mapVmmrLdbs[szKeyVmmrLdbName];

      std::cout << setw(12) <<  pVmmrLdb->GetDataSetName() << " : " << setw(10) << pVmmrLdb->GetLDBName() \
		<< " Legacy: "    << setw(10) << pVmmrLdb->GetHeritage() \
		<< " New Added: " << setw(10) << pVmmrLdb->GetAddedImages() \
		<< " Total: "     << setw(10)<< ( pVmmrLdb->GetAddedImages() +  pVmmrLdb->GetHeritage() ) << endl;
      ofLogfile << setw(12) <<  pVmmrLdb->GetDataSetName() << " : " << setw(10) << pVmmrLdb->GetLDBName() \
		<< " Legacy: "    << setw(10) << pVmmrLdb->GetHeritage() \
		<< " New Added: " << setw(10) << pVmmrLdb->GetAddedImages() \
		<< " Total: "     << setw(10)<< ( pVmmrLdb->GetAddedImages() +  pVmmrLdb->GetHeritage() ) << endl;

      delete pVmmrLdb;
    }
  }
  std::cout << endl << endl;
  ofLogfile<< endl << endl;

  if( count % 10000 != 0 ) {
    t = time(0); 	
    memset( szCurDateTime, 0, _MAX_LINE );
    strftime( szCurDateTime, sizeof(szCurDateTime), "%Y/%m/%d %X %A ",localtime(&t) ); 
    ofLogfile << szCurDateTime << ": Converted " << count << " image into ldb." << endl;
  }
  ofLogfile << szCurDateTime << ": Complete Augment and ldb conversion!" << endl;
  ofLogfile << "Totally added this time: " << count << " images (include augmented images) are converted." << endl << endl;

  std::cout << "++++++++++++++++++++++++++++" << endl;
  std::cout << "Totally convert " << count << " images into ldb for each preprocess type and each patch " << endl;
  std::cout << "Among them: " << endl;
  std::cout << "Original images : " << numOriginal <<endl;
  std::cout << "Noise added aug : " << numAddNoise << endl;
  std::cout << "Darken aug      : " << numDarken << endl;
  std::cout << "Change Color Aug: " << numChColor << endl;
  std::cout << "Smooth Aug      : " << numSmooth << endl;
  std::cout << "            SUM : " << (numOriginal+numAddNoise+numDarken+numChColor+numSmooth )  << endl;
  std::cout << "++++++++++++++++++++++++++++" << endl;

  ofLogfile << "++++++++++++++++++++++++++++" << endl;
  ofLogfile << "Totally convert " << count << " images into ldb for each preprocess type and each patch " << endl;
  ofLogfile << "Among them: " << endl;
  ofLogfile << "Original images : " << numOriginal <<endl;
  ofLogfile << "Noise added aug : " << numAddNoise << endl;
  ofLogfile << "Darken aug      : " << numDarken << endl;
  ofLogfile << "Change Color Aug: " << numChColor << endl;
  ofLogfile << "Smooth Aug      : " << numSmooth << endl;
  ofLogfile << "            SUM : " << (numOriginal+numAddNoise+numDarken+numChColor+numSmooth )  << endl;
  ofLogfile << "Aug list file:    " << strAugListFile << endl;
  ofLogfile << "++++++++++++++++++++++++++++" << endl;

  t = time(NULL); 	
  memset( szCurDateTime, 0, _MAX_LINE );
  strftime( szCurDateTime, sizeof(szCurDateTime), "%Y/%m/%d %X %A ",localtime(&t) ); 
  ofLogfile << endl << szCurDateTime << endl << endl;

  return count;
}

string strToLower(const string &str)
{
  string strTmp = str;
  std::transform(strTmp.begin(),strTmp.end(),strTmp.begin(), ::tolower);
  return strTmp;
}

// return ture : the same
// return false : differenct
bool compareNoCase(const string &strA,const string &strB)
{
  string str1 = strToLower(strA);
  string str2 = strToLower(strB);
  return (str1 == str2);
}

// return 0 if exist
// return -1 if non-exist!
int FileOrFolderExist( string strFileOrFolder )
{
  if( strFileOrFolder.size() <= 0) {
    return -1;
  }

  int status = access(strFileOrFolder.c_str(), F_OK );

  if( status == 0 ) {
    return  0;   // exist
  } else{  
    return -1;   // not existed!
  }	
}

int LoadKeypointsFromFileLab( string strAnnotFile,  
			      vector<CvPoint2D32f>& vecKeyPoints )
{
  ifstream ifAnnotFile;
  ifAnnotFile.open( strAnnotFile.c_str(), ios::in );
  if( !ifAnnotFile.is_open() ) {
    cout << "Error: open file " << strAnnotFile << endl;
    return -2;
  }
  int nKeyPoints = -1;
  bool bStart = false, bEnd = false;
  while( ! ifAnnotFile.eof() ) {
    string strLine;
    getline( ifAnnotFile, strLine );
    
    if( strLine.find( "n_points" ) != string::npos ) {
      string strNum = strLine.substr( strLine.find_last_of( ":" ) );
      TrimSpace( strNum );
      nKeyPoints = atoi( strNum.c_str() );
    }
    if( bStart ) {
      TrimSpace( strLine );
      if( strLine.size() > 3 ) {
	string strXYSep = " ";
	size_t nSepPos = strLine.find_first_of( strXYSep );
	if( nSepPos != string::npos ) {
	  string strX = strLine.substr( 0, nSepPos );
	  string strY = strLine.substr( nSepPos + 1 );
	  TrimSpace( strX );
	  TrimSpace( strY );
	  CvPoint2D32f ptKeyPoint = cvPoint2D32f( atof( strX.c_str()), atof( strY.c_str() ) );
	  vecKeyPoints.push_back( ptKeyPoint );
	}
      }
    }
    
    if( strLine.find( "{" ) != string::npos ) {
      bStart = true;
    }
    if( strLine.find( "}" ) != string::npos ) {
      bEnd = true;
    }
  }
  
  ifAnnotFile.close();
  
  return 0;
}

int VmmrLevelDB::PushData( cv::Mat& cv_img_origin, int line_id, string& relpathName, const int label, \
			   const int height, const int width )
{
  if( !this->ImageToDatum( cv_img_origin, label, height, width, &m_datum ) ){
    return m_count;
  }
  if (!m_data_size_initialized) {
    m_data_size = m_datum.channels() * m_datum.height() * m_datum.width();
    m_data_size_initialized = true;
  } else {
    const string& data = m_datum.data();
    CHECK_EQ(data.size(), m_data_size) << "Incorrect data field size "
				       << data.size();
  }

  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  // sequential
  snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,relpathName.c_str());

  string value;
  // get the value
  m_datum.SerializeToString(&value); // image data serialize to string 'value'
  m_batch->Put(string(key_cstr), value); //add key (line_id and relpathname) and 'value' to batch

  this->m_count += 1;

  return this->m_count;

}
void VmmrLevelDB::WriteData()
{
  if ( this->m_count % LDB_WRITE_BATCH == 0) {
    m_db->Write(leveldb::WriteOptions(), m_batch);
    LOG(ERROR) << "Processed " << m_count << " files.";
    delete m_batch;
    m_batch = new leveldb::WriteBatch();
  }
}

void VmmrLevelDB::WriteDataEnd()
{
  // write the last batch
  if ( m_count % LDB_WRITE_BATCH != 0) {
    m_db->Write(leveldb::WriteOptions(), m_batch);
    LOG(ERROR) << "Processed " << m_count << " files.";
  }
}

bool VmmrLevelDB::ImageToDatum(cv::Mat& cv_img_origin, 
			       const int label,
			       const int height, 
			       const int width, 
			       Datum* datum) 
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

bool VmmrLevelDB::IsRelpathNameInAddedSet( string& strRelpathName )
{
  if( this->m_setImagesAdded.size() <= 0 ) {
    return false;
  }

  string strToFind = strRelpathName;
  set<string>::iterator iterFound = this->m_setImagesAdded.find( strToFind );
  if( iterFound == this->m_setImagesAdded.end() ) {
    return false;
  } else {
    return true;
  }
}

//call me after db opend
int VmmrLevelDB::GetLdbImageRelpathNameToAddedSet()
{
  unsigned int numExist = 0;
  leveldb::Iterator* pIterLdb = this->m_db->NewIterator(leveldb::ReadOptions());
		
  cout << endl;
  //snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id, relpathName.c_str());
  for( pIterLdb->SeekToFirst(); pIterLdb->Valid(); pIterLdb->Next() ) {
    string strKey = pIterLdb->key().ToString();

    //here we extract relpath name before transform-augment and after alignment augment!!!
    size_t nTagPos = strKey.find_first_of( "_" );
    cout << "\r" << setiosflags(ios::left) << setw(60) << strKey \
	 << " relpath name index : " << setiosflags(ios::left) << setw(10) << nTagPos;
    string strLineid = strKey.substr( 0, nTagPos );
    string strRelpathNameAferAug = strKey.substr( nTagPos+1 );

    TrimSpace( strRelpathNameAferAug );		
    string strDriver, strDir, strFileName, strExt;
    SplitPathFileNameExt( strRelpathNameAferAug, strDriver, strDir, strFileName, strExt );

    string strRelpathNameBeforeAug;
    string strRemainder;
    string strMake, strModel, strCoreName, strAugID;
    nTagPos = strFileName.find_first_of( "_" );
    strMake = strFileName.substr( 0, nTagPos );
    strRemainder = strFileName.substr( nTagPos + 1 );

    nTagPos = strRemainder.find_first_of( "_" );
    if( nTagPos == string::npos ) {
      cout << "File name error : " << strRelpathNameAferAug << endl;
    }
    strModel = strRemainder.substr( 0, nTagPos );
    strRemainder = strRemainder.substr( nTagPos + 1 );

    nTagPos = strRemainder.find_first_of( "_" );
    if( nTagPos == string::npos ) {
      cout << "File name error : " << strRelpathNameAferAug << endl;
    }
    strCoreName = strRemainder.substr( 0, nTagPos );
    strRemainder = strRemainder.substr( nTagPos + 1 );

    nTagPos = strRemainder.find_first_of( "_" );
    if( nTagPos == string::npos ) {
      strAugID = strRemainder;
    } else {
      strAugID = strRemainder.substr( 0, nTagPos );
    }
		
    strRelpathNameBeforeAug = strDir + ossep + strMake + "_" + strModel + "_" + strCoreName + "_" + strAugID + strExt;

    this->m_setImagesAdded.insert( strRelpathNameBeforeAug );
    cout << "Set number: " << setiosflags(ios::left ) << setw( 10 ) << this->m_setImagesAdded.size();

    numExist ++;
  }
	
  delete pIterLdb;

  cout << "In " << this->m_strDataSetName << this->m_strDbName << " , There are already " << numExist << " images are put (augmented) into level db. " << endl;
  cout << "They are augmented from " << this->m_setImagesAdded.size() << " Images" << endl;

  return numExist;
}

VmmrLevelDB::VmmrLevelDB( const  string& strDataSetName, const string& strDbName,\
			  const VMMRPPTID& enuPptID, const VMMRKPID& enuKpid, const int& NewWidth, const bool& bNewLdb, const bool bWithAppendList )
{
  m_count = 0;
  m_data_size_initialized = false;
  m_data_size = -1;
  m_bNewLdb = true;
  m_NumHeritage = 0;

  m_enPreprocessType = enuPptID;
  m_enPatchID = enuKpid; 
  m_iNewWidth = NewWidth;
  m_strDataSetVersion = strDataSetName.substr( 0, strDataSetName.find_first_of( "_" ) );
  m_strDataSetName = strDataSetName;
  m_strDbName = strDbName;

  string strCurrDataSetName = "";
  if( enuPptID == PT_COLOR ) {
    strCurrDataSetName = strDataSetName;  //input dataset name is color data set name
  } else if( enuPptID == PT_GRAY ) {
    strCurrDataSetName = strDataSetName + "Gray";
  } else if( enuPptID == PT_EQUALIZEHIST ) {
    strCurrDataSetName = strDataSetName + "equalHist";
  } else if( enuPptID == PT_EQUALIZEHISTCOLOR ){
    strCurrDataSetName = strDataSetName + "equalHistColor";
  } else{
    cout << "Error: Unsupported preprocessing type appread !" << "  Line: " << __LINE__ << endl;
  }

  char szNewWidth[256];
  sprintf( szNewWidth, "%d", NewWidth );
  string strPrepTypeCode = PreprocTypeIDToCode( enuPptID );
  string strPatchIDStr = PatchKPIDToStr( enuKpid );
  string strNewWidth = szNewWidth;
  string strDbParentFolderName = "Caffe_Cropped_" + strPrepTypeCode + "_" + strPatchIDStr + "_" + strNewWidth;
  string strLeveldbFolder = VMMR_DATA_TOPDIR + ossep + strCurrDataSetName + ossep + strDbParentFolderName + ossep + strDbName;
  cout << "ldb path name: " << strLeveldbFolder << endl;

  m_strLdbParentFolder = VMMR_DATA_TOPDIR + ossep + strCurrDataSetName + ossep + strDbParentFolderName + ossep;
  string strCroppedImageFolderName = "Cropped_" + strPrepTypeCode + "_" + strPatchIDStr + "_" + strNewWidth;
  m_strDataSetDataFolder = VMMR_DATA_TOPDIR + ossep + strCurrDataSetName + ossep + strCroppedImageFolderName + ossep;

  this->m_setImagesAdded.clear();

  this->m_bNewLdb = bNewLdb;
  if( this->m_bNewLdb ) {
    // create new ldb:
    if( FileOrFolderExist( strLeveldbFolder ) < 0 ) {
      icvMkDir( strLeveldbFolder.c_str());
    }
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    LOG(INFO) << "Opening leveldb " << strLeveldbFolder;
    leveldb::Status status = leveldb::DB::Open( options, strLeveldbFolder, &m_db);
    CHECK(status.ok()) << "Failed to open leveldb " << strLeveldbFolder;
  } else {
    if( FileOrFolderExist( strLeveldbFolder ) < 0 ) {
      cout << "LDB not exist ! Check :  " << strLeveldbFolder << endl;
      throw "LDB not exist";
      return;
    }

    // open existing one		
    leveldb::Options options;
    options.create_if_missing = false;
    options.max_open_files = 100;
    LOG(INFO) << "Opening leveldb " << strLeveldbFolder;
    leveldb::Status status = leveldb::DB::Open( options,strLeveldbFolder,  &m_db );
    CHECK(status.ok()) << "Failed to open leveldb " << strLeveldbFolder << std::endl << status.ToString();
    if( status.ok() == false ) {
      cout << endl << "Failed reopen leveldb : " << strLeveldbFolder << endl;
      cout << status.ToString () << endl;
    } else {
      if( bWithAppendList == false ) {
	//Append and no append list, we create a havelist from ldb ( this is a little time consuming ... 
	cout << endl << "Get relpath name of images already put into ldb ... " << endl;
	this->m_NumHeritage = this->GetLdbImageRelpathNameToAddedSet();
	cout << "Set LDB image count to : " << this->m_NumHeritage << endl << endl;
      } else {
	cout << endl << "Append list file is provided! Don't create image set from Level db keys." << endl;
      }
    }
  }

  m_batch = new leveldb::WriteBatch();
}
//_makepath
int SplitPathFileNameExt( string& strPathFileName, string& strDriver, string& strDir, string& strFileName, string& strExt )
{
  char szDriver[_MAX_DRIVE];
  char szDir[_MAX_LINE];
  char szFileName[_MAX_LINE];
  char szExt[_MAX_FNAME];
  _splitpath( strPathFileName.c_str(), szDriver, szDir, szFileName, szExt );
  strDriver = szDriver;
  strDir = szDir;
  strFileName = szFileName;
  strExt = szExt;

  return 0;
}

int icvMkDir( const char* filename )
{
  char path[PATH_MAX];
  char* p;
  int pos;

#ifdef _WIN32
  struct _stat st;
#else /* _WIN32 */
  struct stat st;
  mode_t mode;

  mode = 0755;
#endif /* _WIN32 */

  strcpy( path, filename );
  p = path;
  for( ; ; )
    {
      pos = (int)strcspn( p, "/\\" );

      if( pos == (int) strlen( p ) ) break;
      if( pos != 0 )
	{
	  p[pos] = '\0';
#ifdef _WIN32
	  if( p[pos-1] != ':' ) {
	    if( _stat( path, &st ) != 0 ) {
	      if( _mkdir( path ) != 0 ) return 0;
	    }
	  }
#else /* _WIN32 */
	  if( stat( path, &st ) != 0 ) {
	    if( mkdir( path, mode ) != 0 ) return 0;
	  }
#endif /* _WIN32 */
	}

      p[pos] = '/';
      p += pos + 1;
    }
  return 1;
}

/*
  Keep consistent with python version !
*/
string PatchKPIDToStr( int kpID )
{
  switch( kpID ){
  case -2:
    return "LogoArea";   // Make only classify
  case -1:
    return "vface";      // Makemodel classify
  case 0:
    return "WinGlassLT";  //   KP_WinGlassLT = 0 # Windshield Glass Left-top		
  case  1:
    return "WinGlassRT";  //   #KP_WinGlassRT = 1 #: Windshield Glass Right-top
  case  2:
    return "WinGlassLB";  //    KP_WinGlassLB = 2 #: Windshield Glass Left-bottom
  case  3:
    return "WinGlassRB "; //    KP_WinGlassRB =     3 #: Windshield Glass Right-bottom
  case  4:
    return "LeftHLamp";   //    KP_LeftHLamp = 4 #: Left Head Lamp center
  case  5:
    return "RightHLamp";  //    KP_RightHLamp = 5 #: Right Head Lamp center
  case  6:
    return "FrontBumpLB"; //    KP_FrontBumpLB = 6 #: Front Bumper Left Bottom corner
  case  7:
    return "FrontBumpRB"; //    KP_FrontBumpRB = 7 #: Front Bumper Right Bottom corner
  case  8:
    return "VehicleLogo"; //    KP_VehicleLogo = 8 #: Vehicle Logo center
  case  9:
    return "LicensePC";   //    KP_LicensePC = 9 #: License Plate center
  case  10:
    return "MidLineBot";  //    KP_MidLineBot = 10 #: Middle line bottom
  default:
    printf( "Invalid patch or key point id !" );
    return "BAD";
  } 
}


string PreprocTypeIDToCode( int pptid )
{
  switch( pptid ) {
  case 0:
    return "C";
  case 1:
    return "G";
  case 2:
    return "HE";
  case 3:
    return "HEC";
  default:
    cout << "Errro: Unknown preprocess type id : "<< pptid << " !" << endl;
    return "ERROR";
  }
}

void TrimSpace( string& str )
{
  str.erase( 0, str.find_first_not_of("\r\t\n ")); 
  str.erase( str.find_last_not_of("\r\t\n ") + 1);
}

//invoke this after resize and before crop
int VmmrLevelDB::FillLicensePlate( cv::Mat& image, CvPoint2D32f& ptLicensePlateCenter, int iNewWidth )
{
  float fDeltaX = iNewWidth * 0.118;
  float fDeltaY = iNewWidth * 0.04;
  int iRowStart = int( ptLicensePlateCenter.y - fDeltaY );
  int iRowEnd = int( ptLicensePlateCenter.y + fDeltaY );
  int iColStart = int( ptLicensePlateCenter.x - fDeltaX );
  int iColEnd = int( ptLicensePlateCenter.x + fDeltaX );
  
  if( iRowStart < 0 ) {
    iRowStart = 0;
  }
  if( iColStart < 0 ) {
    iColStart = 0;
  }
  if( iRowEnd >= image.rows ) {
    iRowEnd = image.rows - 1;
  }
  if( iColEnd >= image.cols ) {
    iColEnd = image.cols - 1;
  }
  
  for( int r= iRowStart; r < iRowEnd; r ++ ) {
    for( int c = iColStart; c < iColEnd; c++ ) {
      image.at<Vec3b>( r, c ) = Vec3b( 128, 128, 128 );
    }
  }
  return 0;
}

int VmmrLevelDB::GetStandarScaleImage( cv::Mat& image, cv::Mat& imageStandScale, vector<CvPoint2D32f>& vecKeyPoints, float INFLAT_COEFF, float NewWidth )
{
  float im_h = image.rows;
  float im_w = image.cols;
  
  int deflat_h = Round( im_h / ( 10 + 2*INFLAT_COEFF * 10 ) );
  int deflat_w = Round( im_w / ( 10 + 2*INFLAT_COEFF * 10 ) );
  
  int  vf_wid = im_w - deflat_w * 2;
  int  vf_hgt = im_h - deflat_w * 2;
  float im_scale = NewWidth * 1.0 / vf_wid * 1.0;
  int resizeImg_w =  Round( im_w * im_scale ) ;
  int resizeImg_h =  Round( im_h * im_scale ) ;
  resize( image, imageStandScale, Size( resizeImg_w, resizeImg_h ) );
  
  for( int n = 0; n < vecKeyPoints.size(); n++ ) {
    vecKeyPoints[n].x *= im_scale;
    vecKeyPoints[n].y *= im_scale;
  } 
  
  return 0;
}

// cv::Mat& image is standard size image. Same procedure with python code.
int VmmrLevelDB::CropPatchByKeyPointID( cv::Mat& image, cv::Mat& imagePatchCropped, vector<CvPoint2D32f>& vecPtKeyPoint, int nPatchID )
{
  float std_im_h = image.rows;
  float std_im_w = image.cols;
  
  
  // to avoid patch area outside image, padding border first:
  cv::Mat imageWithBorder;
  int pad_width = Round( std_im_w * 0.5f );
  Scalar value = Scalar( 128, 128, 128 );
  copyMakeBorder( image, imageWithBorder, pad_width, pad_width, pad_width, pad_width, BORDER_CONSTANT , value );
  
  vector<CvPoint2D32f> vecPointCoordNew(vecPtKeyPoint.size() );
  for( int n = 0; n < vecPtKeyPoint.size(); n++ ) {
    vecPointCoordNew[n].x = vecPtKeyPoint[n].x + pad_width;
    vecPointCoordNew[n].y = vecPtKeyPoint[n].y + pad_width;		
  }
  
  float pad_im_h = imageWithBorder.rows;
  float pad_im_w = imageWithBorder.cols;
  
  float Half_x_left = -1;
  float Half_x_right = -1;
  float Half_y_up = -1;
  float Helf_y_down = -1;
  float coeff_half_x_l = -1;
  float coeff_half_x_r = -1;
  float coeff_half_y_u = -1;
  float coeff_half_y_d = -1;
  
  if ( nPatchID == MAKE_LogoArea ) 
    {
      coeff_half_x_l = 0.05f;
      coeff_half_x_r = 0.05f;
      coeff_half_y_u = 0.05f;
      coeff_half_y_d = 0.05f;
    } else if ( nPatchID == VF_VehicleFace ) { // crop vehicel face
    coeff_half_x_l = 0.34;
    coeff_half_x_r = 0.34;
    coeff_half_y_u = 0.14;
    coeff_half_y_d = 0.15;
  }else if(  nPatchID == KP_VehicleLogo ) {
    coeff_half_x_l = 0.16;
    coeff_half_x_r = 0.16;
    coeff_half_y_u = 0.12;
    coeff_half_y_d = 0.06;
  } else if ( nPatchID == KP_LeftHLamp ) {
    coeff_half_x_l = 0.09;
    coeff_half_x_r = 0.30;
    coeff_half_y_u = 0.16;
    coeff_half_y_d = 0.09;
  } else if(  nPatchID == KP_RightHLamp ) {
    coeff_half_x_l = 0.30;
    coeff_half_x_r = 0.09;
    coeff_half_y_u = 0.16;
    coeff_half_y_d = 0.09;
  } else if ( nPatchID == KP_FrontBumpLB ) {
    coeff_half_x_l = 0.03;
    coeff_half_x_r = 0.36;
    coeff_half_y_u = 0.18;
    coeff_half_y_d = 0.05;       
  } else if ( nPatchID == KP_FrontBumpRB ) {
    coeff_half_x_l = 0.36;
    coeff_half_x_r = 0.03;
    coeff_half_y_u = 0.18;
    coeff_half_y_d = 0.05;             
  }else if ( nPatchID == KP_LicensePC ) { // # Not used!
    ;
  }else if( nPatchID == KP_MidLineBot ) {
    coeff_half_x_l = 0.18;
    coeff_half_x_r = 0.18;
    coeff_half_y_u = 0.20;
    coeff_half_y_d = 0.03;      
  } else {
    cout << "Invalid crop patch ID " <<  nPatchID << " Name: " << PatchKPIDToStr( nPatchID );
  }
  float xc = -1, yc = -1;
  if ( nPatchID < 0 ) {
    xc = vecPointCoordNew[KP_VehicleLogo].x;
    yc = vecPointCoordNew[KP_VehicleLogo].y;
  } else {
    xc = vecPointCoordNew[nPatchID].x;
    yc = vecPointCoordNew[nPatchID].y;
  }
  
  Half_x_left = Round( coeff_half_x_l * std_im_w );
  Half_x_right = Round( coeff_half_x_r * std_im_w ) ;
  Half_y_up = Round( coeff_half_y_u * std_im_w ) ;
  Helf_y_down = Round( coeff_half_y_d * std_im_w ) ;
  
  cv::Mat croppedPatch;
  
  if ( ( yc-Half_y_up < 0 ) || (yc+Helf_y_down > pad_im_h ) || (  xc-Half_x_left  < 0 ) || ( xc+Half_x_right > pad_im_w )  ) {
    cout << "Patch area outside padding image ! pass ... " << endl;
    return -1;
  } else {
    imagePatchCropped = imageWithBorder( Rect(  xc-Half_x_left,  yc-Half_y_up, Half_x_left + Half_x_right, 
						Half_y_up + Helf_y_down ) );
    return 0;
  }
}

//select a number [0, N-1]
#define random(x) (rand()%x)
int RandChoice( int N )
{
  return random(N);
}

#define NT_NUM 4
void AddNoise( const cv::Mat& image, cv::Mat& outImage )
{
  int L_noiseList[ NT_NUM ] = {-60, -45, -30, -25};
  int H_noiseList[ NT_NUM ] = {60, 45, 30, 25};
  int high = H_noiseList[RandChoice( NT_NUM ) ];
  int low = L_noiseList[RandChoice( NT_NUM ) ];
  
  cv::Mat matNoise( image.size(), CV_8UC3, cv::Scalar( 0, 0, 0 ) );
  CvRNG rng = cvRNG( cv::getTickCount() );
  IplImage iplNoise = matNoise;
  cvRandArr( &rng, &iplNoise, CV_RAND_UNI, cvScalar(L_noiseList[RandChoice( NT_NUM ) ], L_noiseList[RandChoice( NT_NUM ) ],L_noiseList[RandChoice( NT_NUM ) ]),\
	     cvScalar(H_noiseList[RandChoice( NT_NUM ) ], H_noiseList[RandChoice( NT_NUM ) ],H_noiseList[RandChoice( NT_NUM ) ] ) );
  cv::add( matNoise, image, outImage );
}

#define CC_NUM 5
void ChangeColor( const cv::Mat& image, cv::Mat& outImage )
{
  int nChannels[] = {0, 1, 2};
  float coefficients[] = {0.7, 0.6, 0.5, 0.4, 0.3};
  
  int _curCh = RandChoice(3);
  float _curCoeff = coefficients[ RandChoice(CC_NUM)];
  
  cv::Mat matColorChImage = image.clone();
  
  vector<cv::Mat> vecImage1Ch;
  cv::split( matColorChImage, vecImage1Ch );
  cv::convertScaleAbs( vecImage1Ch[_curCh], vecImage1Ch[_curCh], _curCoeff, 15 );
  cv::merge( vecImage1Ch, outImage );
}

#define DL_NUM 5
void DarkenLight( const cv::Mat& image, cv::Mat& outImage )
{
  int nChannels[] = {0, 1, 2};
  float coefficients[] = {0.8, 0.7, 0.6, 0.5, 0.4};
  
  float _curCoeff = coefficients[ RandChoice(DL_NUM)];
  
  cv::Mat matDarkImage = image.clone();
  cv::convertScaleAbs( matDarkImage, outImage, _curCoeff, 10 );
}

#define SMK_NUM 4
void SmoothTexture(  const cv::Mat& image, cv::Mat& outImage )
{
  float sm_sizes[SMK_NUM] = {5, 7, 9, 11};
  
  cv::Mat matSmoothImage( image.size(), CV_8UC3, cv::Scalar( 0, 0, 0 ) );
  cv::GaussianBlur( image, outImage, cv::Size( sm_sizes[RandChoice(SMK_NUM)], sm_sizes[RandChoice(SMK_NUM)] ), 0 );
}


#ifndef WIN32

void _splitpath(const char *path, char *drive, char *dir, char *fname, char *ext)
{
  char *p_whole_name;

  drive[0] = '\0';
  if (NULL == path)
    {
      dir[0] = '\0';
      fname[0] = '\0';
      ext[0] = '\0';
      return;
    }

  if ('/' == path[strlen(path)])
    {
      strcpy(dir, path);
      fname[0] = '\0';
      ext[0] = '\0';
      return;
    }

  char _szPath[_MAX_LINE];
  strcpy( _szPath, path );
  p_whole_name = rindex(_szPath, '/');
  if (NULL != p_whole_name)
    {
      p_whole_name++;
      _split_whole_name(p_whole_name, fname, ext);

      snprintf(dir, strlen(path) - strlen( p_whole_name)/*p_whole_name - path*/, "%s", path);
    }
  else
    {
      _split_whole_name(path, fname, ext);
      dir[0] = '\0';
    }
}

static void _split_whole_name(const char *whole_name, char *fname, char *ext)
{
  char *p_ext;

  // p_ext = rindex(whole_name, '.');
  char szName[_MAX_LINE];
  strcpy( szName, whole_name );
  p_ext = rindex(szName, '.');

  if (NULL != p_ext)
    {
      strcpy(ext, p_ext);
      snprintf(fname, p_ext - whole_name + 1, "%s", whole_name);
    }
  else
    {
      ext[0] = '\0';
      strcpy(fname, whole_name);
    }
}

#endif
