// TestDNNPerformance.cpp : 定义控制台应用程序的入口点。
//

#include <string>
#include <caffe/vmmr/ExtractDNNFeature.h>
#include <vector>
#include <iostream>
#include "cv.h"
#include "opencv2/opencv.hpp"

#include <fstream>
#include <algorithm> 
#include <numeric>
#include "caffe/util/Auxiliary.h"


using namespace cv;
using namespace std;
using namespace VMMR;

#define MAKEMODEL_CLASS_LABEL_DICT_FILE_NAME "Makemodel_ClassLabelDict.txt"
#define MAKE_CLASS_LABEL_DICT_FILE_NAME "Make_ClassLabelDict.txt"
#define GROUND_TRUTH_FILE_NAME "GroundTruth.txt"
#define RANDK_ACCURACY_CURVE_PIC "RankAccuracyCurve.jpg"

typedef struct LabelProb{
	int index;
	VMMRDType prob; 
}LabelProb;


int FeatureReprentTest( VmmrDNN* pDnnFeatureBoy, VRClassType enClassType, string strFeatName, string strSampleFolder, string strListFile, \
					   VMMRFeatDistType distMethod, string strWorkFolder, bool bSaveFeats = false );
int ClassificationTest( VmmrDNN* pDnnFeatureBoy, string strClassLayerName, VRClassType enVRClassType, string strSampleFolder, string strListFile, \
					   string strWorkFolder, bool bSaveResultShowImages = false );
int ClassificationTestEx( VmmrDNN* pDnnFeatureBoy, string strClassLayerName, VRClassType enVRClassType, string strSampleFolder, string strListFile, \
                 			  string strWorkFolder, int batch_num, bool bSaveResultShowImages );

int ExtractDnnFeatures( VmmrDNN* pDnnFeatureBoy, list<string> lstRelpathFileNames, string strSampleFolder,  
					   string strFeatName, map<string, vector<float> >& mapRelpathFileNamedFeats, string strFeatSaveFolder="" );
int SaveVMMRankAccuracyAndDraw( VRClassType enVRClassType, vector<float>& vecMakeAccumRankAccuray, vector<float>& vecMakemodelAccumRankAccuray,
							   string& strMakeRankAccuFile, string& strMakeModelRankAccuFile, string& strRankAccuCurvePicFile );

bool CompareLabelProbs( const LabelProb& v1, const LabelProb& v2) ;
void Usage( const char* szAppName );
void ParseFeatNames( string& strFeatNameParam, list<string>& lstFeatNames );
float CompareDnnFeatDist( vector<float>& vFeatA, vector<float>& vFeatB, VMMRFeatDistType dist_type );
VMMRFeatDistType ParseFeatDistMethod( string& strDistMethod );

//Given a DNN model conduct the following tests.
//1. Test Feature of difference layers intra-extra class separability;
//2. Test DNN classification accuracy.
void Usage( const char* szAppName )
{
	printf( "Usage of %s\n", szAppName );
	printf( "%s <NNProtoFile> <NNBinaryModel> \
		    <SampleFolder> <ListFile> <<featName1>:<featName2>; ..<featNamen>> \
			<distType> <classLayerName> <classType> \
			<workFolder> <IsShowResultImage> <ComputMod> \
			<device_id> <BatchImageNum>\n", szAppName );

	printf( "<classType> : \"make\" or \"makemodel\" " );
	printf( "<IsShowResultImage>: > 0 will save; <= 0 will not save show images \n" );
	printf( "<ComputMod>: <=0 means use cpu only; > 0 will use GPU  \n" );
	printf( "<BatchImageNum> should be less than batch size.\n" );
	printf( "\n" );
}

int main(int argc, char* argv[])
{
	const int input_num = 14;
	if( argc != input_num ) {
		cout << endl << "The number of parameter not correct: " << argc << endl;
		cout << "Should be : " << input_num << endl;
		exit(-1);
	}

	//Naiive parameter parsing: 
	string strProto = argv[1];
	string strModel = argv[2];
	string strSampleFolder = argv[3];
	string strListFile = argv[4];
	
	string strFeatNameParam = argv[5];
	list<string> lstFeatNames;
	ParseFeatNames( strFeatNameParam, lstFeatNames );
	string strDistType = argv[6];
	string strClassLayerName = argv[7];
	string strClassType = argv[8];
	string strWorkFolder = argv[9];
	string strIsShowResultImages = argv[10];
	string strComputMode = argv[11];   //0 denote cpu, 1 denote GPU
        string strGPUDevID = argv[12];
	int batch_num = atoi( argv[13] );

	if( ! IsPathEndWithSlash(strSampleFolder ) ) strSampleFolder += "/";
	if( ! IsPathEndWithSlash(strWorkFolder) ) strWorkFolder += "/";
	//echo params:
	cout << "Proto file: " << strProto << endl;
	cout << "Model file: " << strModel << endl;
	cout << "Sample folder: " << strSampleFolder << endl;
	cout << "List file: " << strListFile << endl;
	cout << "Feat names (" << lstFeatNames.size() << ") : ";
	list<string>::iterator lstStringIter = lstFeatNames.begin();
	for( int n = 0; n < lstFeatNames.size(); n++, lstStringIter++ ) {
		cout << *lstStringIter << "|";
	}
	cout  << endl;
	cout << "Classification Layer Name: " << strClassLayerName << endl;
	cout << "Recognition Class type: " << strClassType << endl;
	cout << "Work folder: " << strWorkFolder << endl;
	cout << "Whether show result images: " << strIsShowResultImages << endl << endl << endl;
	int iComputMode = atoi( strComputMode.c_str() );
	cout << "Compute mode: " << iComputMode << endl;
	int iGPUDevID = atoi( strGPUDevID.c_str() );
	cout << "GPU device ID : " << iGPUDevID << endl;
	cout << "Batch num : " << batch_num << endl;

	DNNFeature::COMPUTE_MOD enComputMode = DNNFeature::GPU;
	if( iComputMode <= 0 ) {
		enComputMode = DNNFeature::CPU;
	}
	VmmrDNN* pDnnFeatureBoy = NULL;		
	if(  compareNoCase( strModel, "MP" ) == true ){
		if( !IsPathEndWithSlash( strProto ) ) {
			strProto += "/";  //For multi-patch should be a directory!
		}
		pDnnFeatureBoy = new DNNFeatMulti();
		int nRetVal = ((DNNFeatMulti*)pDnnFeatureBoy)->InitializeDNN( strProto, enComputMode, iGPUDevID );
		if( nRetVal < 0 )
		{
			cout << "Dnn multi initialization failed ! " << endl;
			return -1;
		}
	} else {
		pDnnFeatureBoy = new DNNFeature();
		int nRetVal = ((DNNFeature*)pDnnFeatureBoy)->InitializeDNN( strProto, strModel, enComputMode, iGPUDevID );	
		if( nRetVal < 0 )
		{
			cout << "Dnn initialization failed ! " << endl;
			return -1;
		}
	}

	int net_batch_size = pDnnFeatureBoy->GetBatchSize();
	cout << "Net's batch size : " << net_batch_size << endl;

	if( batch_num > net_batch_size ) {
		cout << "Error: batch num larger than net batch size ! " << endl;
		return -2;
	}
	
	VRClassType enVRClassType;
	if( compareNoCase( strClassType, "make" ) == true ) {
		enVRClassType = MAKE;
	} else if ( compareNoCase( strClassType, "makemodel" ) == true ) {
		enVRClassType = MAKE_MODEL;
	} else if ( compareNoCase( strClassType, "make_model" ) == true ) {
		enVRClassType = MAKE_MODEL;
	} else {
		cout << "Unknown recognized class type : " << strClassType << endl;
		Usage( argv[0] );
		exit(-1);
	}

	bool bShowResultImages = atoi( strIsShowResultImages.c_str() ) > 0 ? true : false;
	if( bShowResultImages ) {
		cout << endl << "Will show recognized reulst images. " << endl << endl;
	}

	bool bSaveFeats = bShowResultImages;

	VMMRFeatDistType enVMMRFeatDistType = ParseFeatDistMethod( strDistType );

#if 0
	//TestFeatureReprent discriminality (Repeatly computation there is !!! )	
	cout << endl << "Start dnn featuer representation analysis ... " << endl;
	lstStringIter = lstFeatNames.begin();
	for( int n = 0; n < lstFeatNames.size(); n++, lstStringIter++ ) {
		FeatureReprentTest( pDnnFeatureBoy, enVRClassType, *lstStringIter, strSampleFolder, strListFile, enVMMRFeatDistType, strWorkFolder, bSaveFeats );
	}
	cout << "Complete dnn feature representation analysis . " << endl << endl;
#endif //0

	string strResultListFile = strWorkFolder + "Recog_TimeStat.txt"; //print first five in it
	ofstream ofTimeStat( strResultListFile.c_str(), ios::out );

	//Classfication accuracy test:
	int nImages = 0;

	//Classfication accuracy test:
	cout << endl << "Start dnn classification accuracy test ... " << endl;
	double t = (double)getTickCount();

	if( batch_num <= 1 ) {
	     cout << endl << "Batch num <= 1, use single image mode." << endl;
     	     nImages = ClassificationTest( pDnnFeatureBoy, strClassLayerName, enVRClassType, strSampleFolder, strListFile, strWorkFolder, bShowResultImages );
	}else {
	      cout << endl << "Batch num > 1, use batch image set mode." << endl;
	     nImages = ClassificationTestEx( pDnnFeatureBoy, strClassLayerName, enVRClassType, strSampleFolder, strListFile, strWorkFolder, batch_num, bShowResultImages );
	}

	t = ((double)getTickCount() - t)/getTickFrequency();

	cout << "Complete dnn classification accuracy test." << endl << endl;

	cout << "Complete " << nImages << " images' dnn classification accuracy test in " << t << " seconds." << endl;
	cout << "Average time for each image (seconds): " << t / nImages << endl;
	cout << "Average image number per second : " << nImages / t << endl << endl;

	ofTimeStat << "Complete " << nImages << " images' dnn classification accuracy test in "<< t << " seconds." << endl;
	ofTimeStat << "Average time for each image (seconds): " << t / nImages << endl;
	ofTimeStat << "Average image number per second : " << nImages / t << endl;
	ofTimeStat << "Net batch size: " << net_batch_size << endl;
	ofTimeStat << "Actual batch num: " << batch_num << endl;

	ofTimeStat.close();

	return 0;
}

/* 
   copy( lstIntraPairs.begin(), lstIntraPairs.end(), back_inserter( vecIntraPairs ) );
*/
// 1) feature saved in subfolder "Feature_<strFeatName>"
// 2) classification separability curve save in strWorkFolder with name "<strFeatName>_rep_sep.jpg"
// <Experiment>
//      |____ <Make>
//      |_________|__Make_IntraPair.txt
//      |_________|__Make_ExtraPair.txt
//      |         |__<Exp_0> (work folder)
//      |
//      |____<Makemodel>
//      |_________|__Makemodel_IntraPair.txt
//      |_________|__Makemodel_ExtraPair.txt
//                |__<Exp_0> (work folder)
//                |__<Exp_1> (work folder)

int FeatureReprentTest( VmmrDNN* pDnnFeatureBoy, VRClassType enVRClassType, string strFeatName, string strSampleFolder, string strListFile, 
					   VMMRFeatDistType distMethod, string strWorkFolder, bool bSaveFeats )
{
	string strDriver, strDir, strFileName, strExt;
	SplitPathFileNameExt( strListFile, strDriver, strDir, strFileName, strExt );
	string strTrainTestListDir = strDriver + strDir;

	string strIntraListFileR = "";
	string strExtraListFileR = ""; 

	if( enVRClassType == MAKE ) {
		strIntraListFileR = strTrainTestListDir + "Make_IntraPair.txt";
		strExtraListFileR = strTrainTestListDir + "Make_ExtraPair.txt";
	} else if( enVRClassType == MAKE_MODEL  ) {
		strIntraListFileR = strTrainTestListDir + "Makemodel_IntraPair.txt";
		strExtraListFileR = strTrainTestListDir + "Makemodel_ExtraPair.txt";
	} else {
		std::cout << "Unknow class type ! " << endl;
		exit(-1);
	}
	
	string strDnnFeatSaveFolder = strWorkFolder + "DnnFeat_" + strFeatName + "/";
	string strIntraDistFile = strWorkFolder +  strFeatName + "_intra_dist.txt"; 
    string strExtraDistFile = strWorkFolder +  strFeatName + "_extra_dist.txt";
	string strIntraExtraHistImageFile = strWorkFolder +  strFeatName + "_IntraExtraDistHist.jpg" ;

	if( FileOrFolderExist( strDnnFeatSaveFolder )  < 0 ) { icvMkDir( strDnnFeatSaveFolder.c_str() ); }

	if( bSaveFeats == false ) { strDnnFeatSaveFolder = ""; }

	//list<ModelSet> lstAllModelSet;
	list<string> lstRelpathFileNames;
	ReadRelpathFileNameFromListFile( strListFile, lstRelpathFileNames );

	// From model sets create intra and extra name pair sets:
	list<ImagePair> lstIntraPairs;
	//list<ImagePair> lstExtraPairs;
	vector<ImagePair> vecIntraPairs;
	vector<ImagePair> vecExtraPairs;
	
	if( ofGlobalLogger.is_open() )
	{
		ofGlobalLogger << "Load intra pair list ... \n ";
	}
	//Load intra pair randomly	
	int nHopeIntrapairNum = 40000;
	int nExtraIntraRatio = 2;

	unsigned int numIntraPairTotal = 0;
	cout << "Count intra file list line number ... " << endl;
	numIntraPairTotal = CountFileLineNum( strIntraListFileR );
	cout << "Total line number : " << numIntraPairTotal << endl;
	if( nHopeIntrapairNum > numIntraPairTotal ) {
		nHopeIntrapairNum = numIntraPairTotal;
		LoadPairListFromFile( strIntraListFileR, lstIntraPairs );
	    std::copy( lstIntraPairs.begin(), lstIntraPairs.end(), back_inserter( vecIntraPairs ) );
	} else {
		vecIntraPairs.resize( nHopeIntrapairNum );
		if( ofGlobalLogger.is_open() ) {
			ofGlobalLogger << "Load intra pair list (randomly select) ... \n ";
		}	
		LoadRandSelPairListFromFile( strIntraListFileR, numIntraPairTotal, nHopeIntrapairNum, vecIntraPairs );
	}
	
	//Load extra pair randomly:
	unsigned int numExtraPairTotal = 0;
	cout << endl << "Count extra file list line number ... " << endl;
	numExtraPairTotal = CountFileLineNum( strExtraListFileR );
	cout << "Total line number : " << numExtraPairTotal << endl;

	int nSelectedExtra = nExtraIntraRatio * nHopeIntrapairNum;

	if( nSelectedExtra > numExtraPairTotal ) {
		nSelectedExtra = numExtraPairTotal;
		cout << "Randomly selected number is changed to : " << nSelectedExtra << endl;
	}

	vecExtraPairs.resize( nSelectedExtra );
	if( ofGlobalLogger.is_open() ) {
		ofGlobalLogger << "Load extra pair list (randomly select) ... \n ";
	}
	
	LoadRandSelPairListFromFile( strExtraListFileR, numExtraPairTotal, nSelectedExtra, vecExtraPairs );
	
	// Extract component DNN features from all images in image list
	cout << endl << "Extract DNN features ... " << endl;
	map<string, vector<float> > mapRelpathFileNamedFeats;
	ExtractDnnFeatures( pDnnFeatureBoy, lstRelpathFileNames, strSampleFolder,  
					   strFeatName, mapRelpathFileNamedFeats, strDnnFeatSaveFolder );

	// Compute HOG feature distance for each intra or extra pair:
	printf( "\nStart calculate intra and extra distance ... \n" );
	vector<float> vecIntraDists( vecIntraPairs.size(), -1 );
	vector<float> vecExtraDists( vecExtraPairs.size(), -1 );
	
	bool IsWeighted = false;
	unsigned int nValidIntraDist = 0;   //some images' HOG feature may not exist! so cannot calculate distance. remove them
	unsigned int nValidExtraDist = 0;
	if( ofGlobalLogger.is_open() ){
		ofGlobalLogger << "Start calculate intra and extra model pair distance ... \n ";
	}
	CalcIntraExtraDistances( mapRelpathFileNamedFeats, vecIntraPairs, vecExtraPairs, 
							vecIntraDists, vecExtraDists, CompareDnnFeatDist, distMethod, nValidIntraDist, nValidExtraDist );

	sort( vecIntraDists.begin(), vecIntraDists.end() );
	sort( vecExtraDists.begin(), vecExtraDists.end() );
	int numIntraDistNeg = 0, numExtraDistNeg = 0;
	for( int n = 0; n < vecIntraDists.size(); n ++ ) {
		if( vecIntraDists[n] < 0 ) { numIntraDistNeg++; }
	}
	for( int n = 0; n < vecExtraDists.size(); n ++ ) { 
		if( vecExtraDists[n] < 0 ) {  numExtraDistNeg++; }
	}
	if( numIntraDistNeg != (vecIntraDists.size()-nValidIntraDist) ||
		numExtraDistNeg !=  ( vecExtraDists.size() - nValidExtraDist  ) )
	{
		printf( "ERROR: invalid distance number not correct !\n" );
	}
	vector<float> vecValidIntraDists;
	vector<float> vecValidExtraDists;
	copy( vecIntraDists.begin() + (vecIntraDists.size()-nValidIntraDist), vecIntraDists.end(), 
		back_inserter(vecValidIntraDists) );
	copy( vecExtraDists.begin() + ( vecExtraDists.size() - nValidExtraDist ), vecExtraDists.end(),
		back_inserter( vecValidExtraDists ) );
	printf( "\nOver :calculate intra and extra distance.\n" );
	if( ofGlobalLogger.is_open() ){
		ofGlobalLogger << "\nOver :calculate intra and extra distance.\n\n";
	}

	// Save all intra and extra distances to file:]
	printf( "\nSave intra and extra distances to files ... \n" );
	if( ofGlobalLogger.is_open() ){
		ofGlobalLogger << "\nSave intra and extra distances to files ... \n";
	}
	
	SaveIntraExtraDists( vecValidIntraDists, vecValidExtraDists, strIntraDistFile, strExtraDistFile );
	printf( "\nOver.\n" );
	if( ofGlobalLogger.is_open() ){
		ofGlobalLogger << "\nOver.\n";
	}

	// Computer intra extra distance score histogram distance and false pos rate, threshold
	printf( "\nStart calculate intra and extra distance histogram distance and threshold ... \n" );
	if( ofGlobalLogger.is_open() ){
		ofGlobalLogger << "\nStart calculate intra and extra distance histogram distance and threshold ... \n";
	}
	float fThreshold = 0;
	float fFalsePosRate = 0;
	int CV_HD_Method = CV_COMP_CORREL; // CV_COMP_CORREL        =0,  CV_COMP_CHISQR        =1,
     // CV_COMP_INTERSECT     =2,    CV_COMP_BHATTACHARYYA =3,  CV_COMP_HELLINGER     =CV_COMP_BHATTACHARYYA
	CalcIntraExtraDistHistDist( vecValidIntraDists, vecValidExtraDists, fThreshold, fFalsePosRate, CV_HD_Method, strIntraExtraHistImageFile );
	printf( "\nOver.\n" );	
	if( ofGlobalLogger.is_open() ){
		ofGlobalLogger << "\nOver.\n\n";	
	}

	return 0;
}

// 1) save image with make-model drawn on in subfolder "Classification"
// 2) draw rank curve for make and model and save in strWorkFolder with name "make_model_rank.jpg"
int ClassificationTest( VmmrDNN* pDnnFeatureBoy, string strClassLayerName, 
					   VRClassType enVRClassType, string strSampleFolder, string strListFile, string strWorkFolder, bool bSaveResultShowImages )
{	
	string strDriver, strDir, strFileName, strExt;
	SplitPathFileNameExt( strListFile, strDriver, strDir, strFileName, strExt );
	string strTrainTestListDir = strDriver + strDir;
	string strTemp = strTrainTestListDir.substr( 0, strTrainTestListDir.size()-1);
	SplitPathFileNameExt( strTemp , strDriver, strDir, strFileName, strExt );
	string strTrainTestListDirParent = strDriver + strDir;

	string strClassLabelDictFile = ""; 
	string strGroundTruthFile = strTrainTestListDirParent + "/" + GROUND_TRUTH_FILE_NAME;

	string strResultShowFolder = strWorkFolder + "ClassShow/"; //
	string strResultListFile = strWorkFolder + "RecogResult.txt"; //print first five in it
	string strResultErrorListFile = strWorkFolder + "RecogResultError.txt"; //Not rank 1, print first five in it
	string strMakeRankAccuFile = strWorkFolder + "RankAccuracy_Make.txt";
	string strMakeModelRankAccuFile = strWorkFolder + "RankAccuracy_Makemodel.txt";
	string strRankAccuCurvePicFile = strWorkFolder + RANDK_ACCURACY_CURVE_PIC;
	
	if( enVRClassType == MAKE ) {
		strClassLabelDictFile = strTrainTestListDirParent + "/" + MAKE_CLASS_LABEL_DICT_FILE_NAME;
	} else if ( enVRClassType == MAKE_MODEL ) {
		strClassLabelDictFile = strTrainTestListDirParent + "/" + MAKEMODEL_CLASS_LABEL_DICT_FILE_NAME;
	} else {
		cout << "Error: Unknow class type ! " << endl;
		return -1;
	}

	if( FileOrFolderExist( strResultShowFolder )  < 0 ) {
		icvMkDir( strResultShowFolder.c_str() );
	}

	//load class label - make -model dictionary
	map<int, VMMName> mapLabelMakeModel;
	LoadClassLabelDict( strClassLabelDictFile, enVRClassType, mapLabelMakeModel );

	//load list file
	list<string> lstRelpathFilenames;
	ReadRelpathFileNameFromListFile( strListFile, lstRelpathFilenames );

	//load ground truth
	map<string, VMMGrounTruth> mapVmmGroundTruth;
	LoadGroundTruth( strGroundTruthFile, mapVmmGroundTruth );
	if( mapVmmGroundTruth.size() <= 0  ) {
		cout << "Load ground truth failed ! " << endl;
		exit(-1);
	}

	//init dnn
	int dimFeat = pDnnFeatureBoy->GetClassLabelNum();
	vector<VMMRDType> vfDnnFeat( dimFeat );

	vector<int> vecMakeRankHits( dimFeat, 0 );
	vector<int> vecMakemodelRankHits( dimFeat, 0 );
	
	int count = 0;
	ofstream ofResultListFile( strResultListFile.c_str(), ios::out );
	ofstream ofResultErrorListFile( strResultErrorListFile.c_str(), ios::out );

	for( list<string>::iterator lstIter = lstRelpathFilenames.begin(); lstIter != lstRelpathFilenames.end(); lstIter ++ ) {
		string strImgPathFilename = strSampleFolder + *lstIter;
		cv::Mat img = imread( strImgPathFilename, CV_LOAD_IMAGE_UNCHANGED);//CV_LOAD_IMAGE_COLOR );
		//The input source image must be original color image! so we check her.
		if( img.empty() ) {
			cout << "*Error: load image : " <<  strImgPathFilename << endl;			
			continue;
		}

		if( img.channels() != 3 ) {
			cout << "The input source image must be original color image! " << endl << endl;
			return -1;
		}

		// Get ground truth of current test vehicle face or logo area image:
		VMMName stTrueMakemodel; 
		stTrueMakemodel.strMake = mapVmmGroundTruth[*lstIter].strMake;
		stTrueMakemodel.strModel = mapVmmGroundTruth[*lstIter].strModel;

		if( stTrueMakemodel.strMake == "MG" ||  stTrueMakemodel.strMake == "黑豹" ) {  //first: MG, last: 黑豹
			cout << stTrueMakemodel.strMake << endl;
		}

		//For each test vehicle face image:
		if( pDnnFeatureBoy->GetVmmrDnnType() == VmmrDNN::SINGLE ) {
			( (DNNFeature*) pDnnFeatureBoy ) ->DNNClassLabelProb( img, vfDnnFeat );
		} else {
			//load key point
			vector<CvPoint2D32f> vecKeyPoints;
			string strKeyPointFileLab = strImgPathFilename;
			strKeyPointFileLab = strKeyPointFileLab.substr( 0, strKeyPointFileLab.find_last_of( "." ) ) + ".lab";
			LoadKeypointsFromFileLab( strKeyPointFileLab, vecKeyPoints );

			( (DNNFeatMulti*) pDnnFeatureBoy ) ->DNNClassLabelProb( img, vecKeyPoints, vfDnnFeat );
		}

		vector<LabelProb> vecLabelProbs( dimFeat );
		for( int n = 0; n < dimFeat; n ++ ) {
			vecLabelProbs[n].index = n; //label, zero based
			vecLabelProbs[n].prob = vfDnnFeat[n];
		}
		std::sort( vecLabelProbs.begin(), vecLabelProbs.end(), CompareLabelProbs ); //sort result class-label
		
		bool bFirstRecogOneCorrect = false;
		VMMName stVmmNameRecogedFirst = mapLabelMakeModel[vecLabelProbs[0].index]; //recognized first make
		if( enVRClassType == MAKE ) {
			if( compareNoCase( stVmmNameRecogedFirst.strMake, stTrueMakemodel.strMake ) ) {
				bFirstRecogOneCorrect = true; //True, for make recognition
			}
		} else if( enVRClassType == MAKE_MODEL ) {
			if( compareNoCase( stVmmNameRecogedFirst.strMake, stTrueMakemodel.strMake ) && 
				compareNoCase( stVmmNameRecogedFirst.strModel, stTrueMakemodel.strModel)  ) {
					bFirstRecogOneCorrect = true;  //True, for make and model.
			}
		}

		//Draw firt three classes on image and save 
		//TODO: [opencv not support chinese !]		
		string strDriver,strDir, strFileName, strExt;
		SplitPathFileNameExt( *lstIter, strDriver, strDir, strFileName, strExt );
		string ShowResultImageFile = strResultShowFolder;// + strFileName + ".jpg";

		ofResultListFile << *lstIter << " ";
		if( enVRClassType == MAKE_MODEL ) {
			ofResultListFile << "[" << stTrueMakemodel.strMake << "_" << stTrueMakemodel.strModel << "] =》";
		} else if( enVRClassType == MAKE ) {
			ofResultListFile << "[" << stTrueMakemodel.strMake << "] =》";
		}
		if( !bFirstRecogOneCorrect ) {
			ofResultErrorListFile << *lstIter << " ";
			if( enVRClassType == MAKE_MODEL ) {
				ofResultErrorListFile << "[" << stTrueMakemodel.strMake << "_" << stTrueMakemodel.strModel << "] =》";
			} else if( enVRClassType == MAKE ) {
				ofResultErrorListFile << "[" << stTrueMakemodel.strMake << "] =》";
			}
		}


		if( !bFirstRecogOneCorrect && bSaveResultShowImages ) { strFileName = "_ERR_" +	strFileName; }    //first recognized result not correct
		int nPrintResults = 5;
		for( int i = 0; i < nPrintResults; i ++ ) {
			if( enVRClassType == MAKE_MODEL ) {
				ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "_";
				ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strModel << "; ";
				if( !bFirstRecogOneCorrect ) {
					ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "_";
					ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strModel << "; ";

				}
			} else if( enVRClassType == MAKE ) {
				ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "; ";
				if( !bFirstRecogOneCorrect ) {
					ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "; ";
				}
			}

			if( bSaveResultShowImages ) {
				strFileName += "_";
				strFileName += mapLabelMakeModel[ vecLabelProbs[i].index ].strMake;
				if( enVRClassType == MAKE_MODEL ) {
					strFileName += "_";
					strFileName += mapLabelMakeModel[ vecLabelProbs[i].index ].strModel;
				} 
			}
		}		
		if( bSaveResultShowImages ) {
			ShowResultImageFile = ShowResultImageFile + strFileName + ".jpg";
			imwrite( ShowResultImageFile, img );
		}
		ofResultListFile << endl;
		if( !bFirstRecogOneCorrect ) {
			ofResultErrorListFile << endl;
		}

		//Find it's rank in recognition result:
		unsigned int uRank = 0;
		int uTrueMakePos = -1;
		int uTrueMakemodelPos = -1;
		for( vector<LabelProb>::iterator labelProbIter = vecLabelProbs.begin();
			labelProbIter != vecLabelProbs.end(); labelProbIter ++, uRank ++ ) {
			VMMName stVmmNameRecoged = mapLabelMakeModel[labelProbIter->index];
			if( uTrueMakePos < 0 ) {
				if( compareNoCase( stVmmNameRecoged.strMake, stTrueMakemodel.strMake ) ) {
					uTrueMakePos = uRank; //True make position.
				}
			}
			if( uTrueMakemodelPos < 0 && enVRClassType == MAKE_MODEL ) {
				if( compareNoCase( stVmmNameRecoged.strMake, stTrueMakemodel.strMake ) && 
					compareNoCase( stVmmNameRecoged.strModel, stTrueMakemodel.strModel)  ) {
					uTrueMakemodelPos = uRank;  //True make and model position,  position one based.
				}
			}
			if( uTrueMakePos >= 0 && enVRClassType == MAKE ) break;
			if( uTrueMakePos >= 0 && uTrueMakemodelPos >= 0 ) break;
		}

		if( uTrueMakePos >= 0 ) { 
			vecMakeRankHits[uTrueMakePos] ++;
		}
		if( uTrueMakemodelPos >= 0 && enVRClassType == MAKE_MODEL ) {
			vecMakemodelRankHits[uTrueMakemodelPos] ++;
		}
		count ++;
		printf( "Check recognition result [%d/%d][%d %d]: %s \r", count, lstRelpathFilenames.size(), uTrueMakePos,  uTrueMakemodelPos, lstIter->c_str() );
	}
	ofResultListFile.close();
	ofResultErrorListFile.close();

	vector<float> vecMakeAccumRankHitAccuray( dimFeat, 0 );
	vector<float> vecMakemodelAccumRankHitAccuray( dimFeat, 0 );

	cout << endl << endl<< "Compute accumulate rank accuracy ... " << endl << endl;
	for( int n = 0; n < mapLabelMakeModel.size(); n ++ ) {
		vecMakeAccumRankHitAccuray[n] = std::accumulate( vecMakeRankHits.begin(), vecMakeRankHits.begin()+n+1, 0 );	
		vecMakeAccumRankHitAccuray[n] /= ( 1.0f* lstRelpathFilenames.size() );
		
		if(  enVRClassType == MAKE_MODEL ) {
			vecMakemodelAccumRankHitAccuray[n] = std::accumulate( vecMakemodelRankHits.begin(), vecMakemodelRankHits.begin()+n+1, 0 );		
			vecMakemodelAccumRankHitAccuray[n] /= ( 1.0f* lstRelpathFilenames.size() );
		}
	}

	//draw and save rank accuracy curve:
	cout << "Draw and save make and model rank accuracy data and curve picture ... " << endl;
	SaveVMMRankAccuracyAndDraw( enVRClassType, vecMakeAccumRankHitAccuray, vecMakemodelAccumRankHitAccuray,
							   strMakeRankAccuFile, strMakeModelRankAccuFile, strRankAccuCurvePicFile );
	cout << "Complete draw and save make and model rank accuracy data and curve picture." << endl << endl;

	return count;
}

int SaveVMMRankAccuracyAndDraw( VRClassType enVRClassType, vector<float>& vecMakeAccumRankAccuray, vector<float>& vecMakemodelAccumRankAccuray,
							   string& strMakeRankAccuFile, string& strMakeModelRankAccuFile, string& strRankAccuCurvePicFile )
{
	ofstream ofMakeAccuFile, ofMakemodelAccuFile;
	ofMakeAccuFile.open( strMakeRankAccuFile.c_str(), ios::out );
	if( enVRClassType == MAKE_MODEL ) {
	  ofMakemodelAccuFile.open( strMakeModelRankAccuFile.c_str(), ios::out );
	}
	for( int n = 0; n < vecMakeAccumRankAccuray.size(); n ++ ) {
		ofMakeAccuFile << n+1 << " " << vecMakeAccumRankAccuray[n] << endl;
		if( enVRClassType == MAKE_MODEL ) {
			ofMakemodelAccuFile << n+1 << " " << vecMakemodelAccumRankAccuray[n] << endl;
		}
	}
	ofMakeAccuFile.close();
	if( enVRClassType == MAKE_MODEL ) {
		ofMakemodelAccuFile.close();
	}
	
	//draw rank accuracy curve by gnuplot
	FILE* gid=popen( "/usr/bin/gnuplot","w");
	
	fprintf(gid,"set terminal png \n");
	fprintf(gid,"set output '%s'\n", strRankAccuCurvePicFile.c_str() );
	if( enVRClassType == MAKE_MODEL ) {
		fprintf(gid,"plot '%s'w lines,'%s'w lines \n", strMakeRankAccuFile.c_str(), strMakeModelRankAccuFile.c_str() );	
	} else {
		fprintf(gid,"plot '%s'w lines\n", strMakeRankAccuFile.c_str() );	
	}
	fprintf(gid,"set xlabel 'Rank'\n");
	fprintf(gid,"set ylabel 'Accuracy'\n");
	if( enVRClassType == MAKE_MODEL ) {
		fprintf(gid,"set title \"Make and model rank accuracy\"\n");
	} else {
		fprintf(gid,"set title \"Make rank accuracy\"\n");
	}
	fprintf(gid,"show xlabel\n");
	fprintf(gid,"show ylabel\n");
	fprintf(gid,"show title\n");
	fflush(gid);

	pclose( gid );

	return 0;
}

bool CompareLabelProbs( const LabelProb& v1, const LabelProb& v2) 
{  
	return v1.prob > v2.prob;
}  


void ParseFeatNames( string& strFeatNameParam, list<string>& lstFeatNames )
{
	int start = 0;
	int colon_pos = -1;
		
	colon_pos = strFeatNameParam.find_first_of( ":" );
	while( colon_pos != string::npos ) {
		lstFeatNames.push_back( strFeatNameParam.substr( start, colon_pos - start ) );
		start = colon_pos + 1;
		colon_pos = strFeatNameParam.find_first_of( ":", start );
	}
	lstFeatNames.push_back( strFeatNameParam.substr( start ) );
}

// 1) feature saved in subfolder "Feature_<strFeatName>"
int ExtractDnnFeatures( VmmrDNN* pDnnFeatureBoy, list<string> lstRelpathFileNames, string strSampleFolder,  
					   string strFeatName, map<string, vector<float> >& mapRelpathFileNamedFeats, string strFeatSaveFolder )
{
	int dimFeat = pDnnFeatureBoy->GetFeatureDim( strFeatName );
	vector<VMMRDType> vfDnnFeat( dimFeat );
		
	int count = 0;
	for( list<string>::iterator relPathnameIter = lstRelpathFileNames.begin(); 
		relPathnameIter != lstRelpathFileNames.end(); relPathnameIter ++ ) {
			if( relPathnameIter->size() < 4 ) continue;
		cv::Mat img = imread( strSampleFolder + *relPathnameIter, CV_LOAD_IMAGE_COLOR );
		if( img.empty() ) {
			cout << "Empty image loaded: " << *relPathnameIter << endl << endl;
			continue;
		}
		if( pDnnFeatureBoy->GetVmmrDnnType() == VmmrDNN::SINGLE ) {
			((DNNFeature*)pDnnFeatureBoy) ->ExtractDNNFeat( img, strFeatName, vfDnnFeat );
		} else {
			//load key point
			vector<CvPoint2D32f> vecKeyPoints;
			string strKeyPointFileLab = strSampleFolder + *relPathnameIter;
			strKeyPointFileLab = strKeyPointFileLab.substr( 0, strKeyPointFileLab.find_last_of( "." ) ) + ".lab";
			LoadKeypointsFromFileLab( strKeyPointFileLab, vecKeyPoints );
			((DNNFeatMulti*)pDnnFeatureBoy) ->ExtractDNNFeat( img, vecKeyPoints, strFeatName, vfDnnFeat );
		}
		NormalizeFeat( vfDnnFeat, FN_Norm01_L2 ); //normalized

		mapRelpathFileNamedFeats[*relPathnameIter] = vfDnnFeat;

		if( strFeatSaveFolder.size() > 0 ) {  //save feature
			string strDnnFeatFile = strFeatSaveFolder + *relPathnameIter;
			string strDriver, strDir, strFilename, strExt;
			SplitPathFileNameExt( strDnnFeatFile,  strDriver, strDir, strFilename, strExt );
			if( FileOrFolderExist( strFeatSaveFolder + strDir ) < 0 ) {
				icvMkDir( strDnnFeatFile.c_str() );
			}
			//change extension name:
			char szDnnFeatFile[_MAX_LINE];
			//_makepath( szDnnFeatFile,  strDriver.c_str(), strDir.c_str(), strFilename.c_str(), ".dnn" );
			strDnnFeatFile = strDir + "/" + strFilename + ".dnn";

			pDnnFeatureBoy->WriteDNNFeatToFile( vfDnnFeat, strDnnFeatFile );
		}

		count ++;
		printf( "Extract dnn feature [%d]: %s \r", count, relPathnameIter->c_str() );
	}
	cout << "Complete DNN feature extraction. " << endl << endl;

	return 0;
}


float CompareDnnFeatDist( vector<float>& vFeatA, vector<float>& vFeatB, VMMRFeatDistType dist_type )
{
	float fFeatDist = 0.0f;

	fFeatDist = CompareGeneralFeatDistance( vFeatA, vFeatB, dist_type );

	return fFeatDist;
}

VMMRFeatDistType ParseFeatDistMethod( string& strDistMethod )
{
	TrimSpace( strDistMethod );

	if( compareNoCase( "L1", strDistMethod ) == true ){
	  return VMMR::L1;
	} else if ( compareNoCase( "L2", strDistMethod ) == true ){
	  return VMMR::L2;
	} else if ( compareNoCase( "COSINE", strDistMethod ) == true ){
		return COSINE;
	} else if ( compareNoCase( "CHISQR", strDistMethod ) == true ){
		return CHISQR;
	} else if ( compareNoCase( "INTERSECT", strDistMethod ) == true ){
		return INTERSECT;
	} else {
		cout << "Unknow distance measure type !" << endl;
		return UNKNOWN;
	}
}



/*
Support batch mode test.
*/
int ClassificationTestEx( VmmrDNN* pDnnFeatureBoy, string strClassLayerName, VRClassType enVRClassType, string strSampleFolder, string strListFile, \
						 string strWorkFolder, int batch_num, bool bSaveResultShowImages )
{
	string strDriver, strDir, strFileName, strExt;
	SplitPathFileNameExt( strListFile, strDriver, strDir, strFileName, strExt );
	string strTrainTestListDir = strDriver + strDir;
	string strTemp = strTrainTestListDir.substr( 0, strTrainTestListDir.size()-1);
	SplitPathFileNameExt( strTemp , strDriver, strDir, strFileName, strExt );
	string strTrainTestListDirParent = strDriver + strDir;

	string strClassLabelDictFile = ""; 
	string strGroundTruthFile = strTrainTestListDirParent + "/" + GROUND_TRUTH_FILE_NAME;

	string strResultShowFolder = strWorkFolder + "ClassShow/"; //
	string strResultListFile = strWorkFolder + "RecogResult.txt"; //print first five in it
	string strResultErrorListFile = strWorkFolder + "RecogResultError.txt"; //Not rank 1, print first five in it
	string strMakeRankAccuFile = strWorkFolder + "RankAccuracy_Make.txt";
	string strMakeModelRankAccuFile = strWorkFolder + "RankAccuracy_Makemodel.txt";
	string strRankAccuCurvePicFile = strWorkFolder + RANDK_ACCURACY_CURVE_PIC;

	if( enVRClassType == MAKE ) {
		strClassLabelDictFile = strTrainTestListDirParent + "/" + MAKE_CLASS_LABEL_DICT_FILE_NAME;
	} else if ( enVRClassType == MAKE_MODEL ) {
		strClassLabelDictFile = strTrainTestListDirParent + "/" + MAKEMODEL_CLASS_LABEL_DICT_FILE_NAME;
	} else {
		cout << "Error: Unknow class type ! " << endl;
		return -1;
	}

	if( FileOrFolderExist( strResultShowFolder )  < 0 ) {
		icvMkDir( strResultShowFolder.c_str() );
	}

	//load class label - make -model dictionary
	map<int, VMMName> mapLabelMakeModel;
	LoadClassLabelDict( strClassLabelDictFile, enVRClassType, mapLabelMakeModel );

	//load list file
	list<string> lstRelpathFilenames;
	ReadRelpathFileNameFromListFile( strListFile, lstRelpathFilenames );

	//load ground truth
	map<string, VMMGrounTruth> mapVmmGroundTruth;
	LoadGroundTruth( strGroundTruthFile, mapVmmGroundTruth );
	if( mapVmmGroundTruth.size() <= 0  ) {
		cout << "Load ground truth failed ! " << endl;
		exit(-1);
	}

	int count = 0;
	ofstream ofResultListFile( strResultListFile.c_str(), ios::out );
	ofstream ofResultErrorListFile( strResultErrorListFile.c_str(), ios::out );

	int batch_size =  pDnnFeatureBoy->GetBatchSize(); // number of images in each batch
	cout << endl << "Max batch size is : " << batch_size << endl;	
	cout << "Use batch size is : " << batch_num << endl;

	if( batch_num > batch_size ) {
		cout << "Image batch num cannot larger than batch size !" << endl;
		return -1;
	}

	vector<cv::Mat> vecImageBatch(batch_num);	
	vector<vector<CvPoint2D32f> > keyPointsBatch(batch_num);
	vector<string> vecRelPathNameBatch( batch_num );

	int dimFeat = pDnnFeatureBoy->GetClassLabelNum();
	vector<vector<VMMRDType> > vfDnnFeatBatch( batch_num );
	for( int i =0; i < batch_num; i++ ) {
		vector<VMMRDType> temp(dimFeat);
		vfDnnFeatBatch[i] = temp;
	}

	vector<int> vecMakeRankHits( dimFeat, 0 );
	vector<int> vecMakemodelRankHits( dimFeat, 0 );

	int batch_id = 0;

	for( list<string>::iterator lstIter = lstRelpathFilenames.begin(); lstIter != lstRelpathFilenames.end(); lstIter ++ ) {

		double t = (double)getTickCount();
		string strImgPathFilename = strSampleFolder + *lstIter;
		cv::Mat img = imread( strImgPathFilename, CV_LOAD_IMAGE_UNCHANGED);//CV_LOAD_IMAGE_COLOR );
		//The input source image must be original color image! so we check her.
		if( img.empty() ) {
			cout << "*Error: load image : " <<  strImgPathFilename << endl;			
			continue;
		}
		if( img.channels() != 3 ) {
			cout << "The input source image must be original color image! " << endl << endl;
			return -1;
		}		

		//load key point
		string strKeyPointFileLab = strImgPathFilename;		
		strKeyPointFileLab = strKeyPointFileLab.substr( 0, strKeyPointFileLab.find_last_of( "." ) ) + ".lab";
		if( LoadKeypointsFromFileLab( strKeyPointFileLab, keyPointsBatch[batch_id] )  < 0 ) {
			cout << "Load key points failed ! Annot file: " << strKeyPointFileLab << endl;
			continue;
		}

		vecImageBatch[batch_id] = img;
		vecRelPathNameBatch[batch_id] = *lstIter;
		

		batch_id ++;
		if( batch_id < batch_num ){
			continue;  //not full, continue to eat ...
		} 

		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << endl << "Loaded " << batch_id << " images using " << t << " seconds." << " Now classify in batch mode... " << endl;

		//full, then process in batch
		batch_id = 0; //indicate this batch is processed. can placed it at the end.

		//dnn batch forward:
		//For each test vehicle face image:
		int valRet = -1;
		if( pDnnFeatureBoy->GetVmmrDnnType() == VmmrDNN::SINGLE ) {
			valRet = ( (DNNFeature*) pDnnFeatureBoy ) ->DNNClassLabelProbEx( vecImageBatch, vfDnnFeatBatch );
		} else {
			valRet = ( (DNNFeatMulti*) pDnnFeatureBoy ) ->DNNClassLabelProbEx( vecImageBatch, keyPointsBatch, vfDnnFeatBatch );
		}

		if( valRet < 0 ) {
			cout << "Error: Dnn classify. " << endl;
			continue;
		}

		for( int k = 0; k < batch_num; k ++ )
		{

			// Get ground truth of current test vehicle face or logo area image:
			VMMName stTrueMakemodel; 
			stTrueMakemodel.strMake = mapVmmGroundTruth[ vecRelPathNameBatch[k] ].strMake;
			stTrueMakemodel.strModel = mapVmmGroundTruth[ vecRelPathNameBatch[k] ].strModel;

			vector<LabelProb> vecLabelProbs( dimFeat );
			for( int n = 0; n < dimFeat; n ++ ) {
				vecLabelProbs[n].index = n; //label, zero based
				vecLabelProbs[n].prob = (vfDnnFeatBatch[k])[n];  // the k-th image's dnn prob vector copy
			}
			std::sort( vecLabelProbs.begin(), vecLabelProbs.end(), CompareLabelProbs ); //sort result class-label

			bool bFirstRecogOneCorrect = false;
			VMMName stVmmNameRecogedFirst = mapLabelMakeModel[vecLabelProbs[0].index]; //recognized first make
			if( enVRClassType == MAKE ) {
				if( compareNoCase( stVmmNameRecogedFirst.strMake, stTrueMakemodel.strMake ) ) {
					bFirstRecogOneCorrect = true; //True, for make recognition
				}
			} else if( enVRClassType == MAKE_MODEL ) {
				if( compareNoCase( stVmmNameRecogedFirst.strMake, stTrueMakemodel.strMake ) && 
					compareNoCase( stVmmNameRecogedFirst.strModel, stTrueMakemodel.strModel)  ) {
						bFirstRecogOneCorrect = true;  //True, for make and model.
				}
			}

			string strDriver,strDir, strFileName, strExt;
			SplitPathFileNameExt( vecRelPathNameBatch[k], strDriver, strDir, strFileName, strExt );
			string ShowResultImageFile = strResultShowFolder;// + strFileName + ".jpg";

			//recog result
			ofResultListFile << vecRelPathNameBatch[k] << " ";
			if( enVRClassType == MAKE_MODEL ) {
				ofResultListFile << "[" << stTrueMakemodel.strMake << "_" << stTrueMakemodel.strModel << "] =》";
			} else if( enVRClassType == MAKE ) {
				ofResultListFile << "[" << stTrueMakemodel.strMake << "] =》";
			}

			//error list
			if( !bFirstRecogOneCorrect ) {
				ofResultErrorListFile << vecRelPathNameBatch[k] << " ";
				if( enVRClassType == MAKE_MODEL ) {
					ofResultErrorListFile << "[" << stTrueMakemodel.strMake << "_" << stTrueMakemodel.strModel << "] =》";
				} else if( enVRClassType == MAKE ) {
					ofResultErrorListFile << "[" << stTrueMakemodel.strMake << "] =》";
				}
			}


			if( !bFirstRecogOneCorrect && bSaveResultShowImages ) 
			{
				strFileName = "_ERR_" +	strFileName; 
			}    //first recognized result not correct

			const int nPrintResults = 5;
			for( int i = 0; i < nPrintResults; i ++ ) {
				if( enVRClassType == MAKE_MODEL ) {
					ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "_";
					ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strModel << "; ";
					if( !bFirstRecogOneCorrect ) {
						ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "_";
						ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strModel << "; ";

					}
				} else if( enVRClassType == MAKE ) {
					ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "; ";
					if( !bFirstRecogOneCorrect ) {
						ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "; ";
					}
				}

				if( bSaveResultShowImages ) {
					strFileName += "_";
					strFileName += mapLabelMakeModel[ vecLabelProbs[i].index ].strMake;
					if( enVRClassType == MAKE_MODEL ) {
						strFileName += "_";
						strFileName += mapLabelMakeModel[ vecLabelProbs[i].index ].strModel;
					} 
				}
			}

			if( bSaveResultShowImages ) {
				ShowResultImageFile = ShowResultImageFile + strFileName + ".jpg";
				imwrite( ShowResultImageFile, vecImageBatch[k] );
			}
			ofResultListFile << endl;
			if( !bFirstRecogOneCorrect ) {
				ofResultErrorListFile << endl;
			}

			//Find it's rank in recognition result:
			unsigned int uRank = 0;
			int uTrueMakePos = -1;
			int uTrueMakemodelPos = -1;
			for( vector<LabelProb>::iterator labelProbIter = vecLabelProbs.begin();
				labelProbIter != vecLabelProbs.end(); labelProbIter ++, uRank ++ ) {
					VMMName stVmmNameRecoged = mapLabelMakeModel[labelProbIter->index];
					if( uTrueMakePos < 0 ) {
						if( compareNoCase( stVmmNameRecoged.strMake, stTrueMakemodel.strMake ) ) {
							uTrueMakePos = uRank; //True make position.
						}
					}
					if( uTrueMakemodelPos < 0 && enVRClassType == MAKE_MODEL ) {
						if( compareNoCase( stVmmNameRecoged.strMake, stTrueMakemodel.strMake ) && 
							compareNoCase( stVmmNameRecoged.strModel, stTrueMakemodel.strModel)  ) {
								uTrueMakemodelPos = uRank;  //True make and model position,  position one based.
						}
					}
					if( uTrueMakePos >= 0 && enVRClassType == MAKE ) break;
					if( uTrueMakePos >= 0 && uTrueMakemodelPos >= 0 ) break;
			}

			if( uTrueMakePos >= 0 ) { 
				vecMakeRankHits[uTrueMakePos] ++;
			}
			if( uTrueMakemodelPos >= 0 && enVRClassType == MAKE_MODEL ) {
				vecMakemodelRankHits[uTrueMakemodelPos] ++;
			}
			count ++;
			printf( "Check recognition result [%d/%d][%d %d]: %s \r", \
				count, lstRelpathFilenames.size(), uTrueMakePos,  uTrueMakemodelPos, vecRelPathNameBatch[k].c_str() );
		}
	}

	if( batch_id > 0 ) //process remain
	{
		//dnn batch forward:
		//For each test vehicle face image:
		int valRet = -1;
		if( pDnnFeatureBoy->GetVmmrDnnType() == VmmrDNN::SINGLE ) {
			valRet = ( (DNNFeature*) pDnnFeatureBoy ) ->DNNClassLabelProbEx( vecImageBatch, vfDnnFeatBatch );
		} else {

			valRet = ( (DNNFeatMulti*) pDnnFeatureBoy ) ->DNNClassLabelProbEx( vecImageBatch, keyPointsBatch, vfDnnFeatBatch );
		}

		if( valRet < 0 ) {
			cout << "Error: Dnn classify. " << endl;
			return -3;
		}

		for( int k = 0; k < batch_id; k ++ )
		{

			// Get ground truth of current test vehicle face or logo area image:
			VMMName stTrueMakemodel; 
			stTrueMakemodel.strMake = mapVmmGroundTruth[ vecRelPathNameBatch[k] ].strMake;
			stTrueMakemodel.strModel = mapVmmGroundTruth[ vecRelPathNameBatch[k] ].strModel;

			vector<LabelProb> vecLabelProbs( dimFeat );
			for( int n = 0; n < dimFeat; n ++ ) {
				vecLabelProbs[n].index = n; //label, zero based
				vecLabelProbs[n].prob = (vfDnnFeatBatch[k])[n];  // the k-th image's dnn prob vector copy
			}
			std::sort( vecLabelProbs.begin(), vecLabelProbs.end(), CompareLabelProbs ); //sort result class-label

			bool bFirstRecogOneCorrect = false;
			VMMName stVmmNameRecogedFirst = mapLabelMakeModel[vecLabelProbs[0].index]; //recognized first make
			if( enVRClassType == MAKE ) {
				if( compareNoCase( stVmmNameRecogedFirst.strMake, stTrueMakemodel.strMake ) ) {
					bFirstRecogOneCorrect = true; //True, for make recognition
				}
			} else if( enVRClassType == MAKE_MODEL ) {
				if( compareNoCase( stVmmNameRecogedFirst.strMake, stTrueMakemodel.strMake ) && 
					compareNoCase( stVmmNameRecogedFirst.strModel, stTrueMakemodel.strModel)  ) {
						bFirstRecogOneCorrect = true;  //True, for make and model.
				}
			}

			string strDriver,strDir, strFileName, strExt;
			SplitPathFileNameExt( vecRelPathNameBatch[k], strDriver, strDir, strFileName, strExt );
			string ShowResultImageFile = strResultShowFolder;// + strFileName + ".jpg";

			//recog result
			ofResultListFile << vecRelPathNameBatch[k] << " ";
			if( enVRClassType == MAKE_MODEL ) {
				ofResultListFile << "[" << stTrueMakemodel.strMake << "_" << stTrueMakemodel.strModel << "] =》";
			} else if( enVRClassType == MAKE ) {
				ofResultListFile << "[" << stTrueMakemodel.strMake << "] =》";
			}

			//error list
			if( !bFirstRecogOneCorrect ) {
				ofResultErrorListFile << vecRelPathNameBatch[k] << " ";
				if( enVRClassType == MAKE_MODEL ) {
					ofResultErrorListFile << "[" << stTrueMakemodel.strMake << "_" << stTrueMakemodel.strModel << "] =》";
				} else if( enVRClassType == MAKE ) {
					ofResultErrorListFile << "[" << stTrueMakemodel.strMake << "] =》";
				}
			}


			if( !bFirstRecogOneCorrect && bSaveResultShowImages ) 
			{
				strFileName = "_ERR_" +	strFileName; 
			}    //first recognized result not correct

			const int nPrintResults = 5;
			for( int i = 0; i < nPrintResults; i ++ ) {
				if( enVRClassType == MAKE_MODEL ) {
					ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "_";
					ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strModel << "; ";
					if( !bFirstRecogOneCorrect ) {
						ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "_";
						ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strModel << "; ";

					}
				} else if( enVRClassType == MAKE ) {
					ofResultListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "; ";
					if( !bFirstRecogOneCorrect ) {
						ofResultErrorListFile << mapLabelMakeModel[ vecLabelProbs[i].index ].strMake << "; ";
					}
				}

				if( bSaveResultShowImages ) {
					strFileName += "_";
					strFileName += mapLabelMakeModel[ vecLabelProbs[i].index ].strMake;
					if( enVRClassType == MAKE_MODEL ) {
						strFileName += "_";
						strFileName += mapLabelMakeModel[ vecLabelProbs[i].index ].strModel;
					} 
				}
			}

			if( bSaveResultShowImages ) {
				ShowResultImageFile = ShowResultImageFile + strFileName + ".jpg";
				imwrite( ShowResultImageFile, vecImageBatch[k] );
			}
			ofResultListFile << endl;
			if( !bFirstRecogOneCorrect ) {
				ofResultErrorListFile << endl;
			}

			//Find it's rank in recognition result:
			unsigned int uRank = 0;
			int uTrueMakePos = -1;
			int uTrueMakemodelPos = -1;
			for( vector<LabelProb>::iterator labelProbIter = vecLabelProbs.begin();
				labelProbIter != vecLabelProbs.end(); labelProbIter ++, uRank ++ ) {
					VMMName stVmmNameRecoged = mapLabelMakeModel[labelProbIter->index];
					if( uTrueMakePos < 0 ) {
						if( compareNoCase( stVmmNameRecoged.strMake, stTrueMakemodel.strMake ) ) {
							uTrueMakePos = uRank; //True make position.
						}
					}
					if( uTrueMakemodelPos < 0 && enVRClassType == MAKE_MODEL ) {
						if( compareNoCase( stVmmNameRecoged.strMake, stTrueMakemodel.strMake ) && 
							compareNoCase( stVmmNameRecoged.strModel, stTrueMakemodel.strModel)  ) {
								uTrueMakemodelPos = uRank;  //True make and model position,  position one based.
						}
					}
					if( uTrueMakePos >= 0 && enVRClassType == MAKE ) break;
					if( uTrueMakePos >= 0 && uTrueMakemodelPos >= 0 ) break;
			}

			if( uTrueMakePos >= 0 ) { 
				vecMakeRankHits[uTrueMakePos] ++;
			}
			if( uTrueMakemodelPos >= 0 && enVRClassType == MAKE_MODEL ) {
				vecMakemodelRankHits[uTrueMakemodelPos] ++;
			}
			count ++;
			printf( "Check recognition result [%d/%d][%d %d]: %s \r", \
				count, lstRelpathFilenames.size(), uTrueMakePos,  uTrueMakemodelPos, vecRelPathNameBatch[k].c_str() );
		}
	}


	ofResultListFile.close();
	ofResultErrorListFile.close();

	vector<float> vecMakeAccumRankHitAccuray( dimFeat, 0 );
	vector<float> vecMakemodelAccumRankHitAccuray( dimFeat, 0 );

	cout << endl << endl<< "Compute accumulate rank accuracy ... " << endl << endl;
	for( int n = 0; n < mapLabelMakeModel.size(); n ++ ) {
		vecMakeAccumRankHitAccuray[n] = std::accumulate( vecMakeRankHits.begin(), vecMakeRankHits.begin()+n+1, 0 );	
		vecMakeAccumRankHitAccuray[n] /= ( 1.0f* lstRelpathFilenames.size() );

		if(  enVRClassType == MAKE_MODEL ) {
			vecMakemodelAccumRankHitAccuray[n] = std::accumulate( vecMakemodelRankHits.begin(), vecMakemodelRankHits.begin()+n+1, 0 );		
			vecMakemodelAccumRankHitAccuray[n] /= ( 1.0f* lstRelpathFilenames.size() );
		}
	}

	//draw and save rank accuracy curve:
	cout << "Draw and save make and model rank accuracy data and curve picture ... " << endl;
	SaveVMMRankAccuracyAndDraw( enVRClassType, vecMakeAccumRankHitAccuray, vecMakemodelAccumRankHitAccuray,
		strMakeRankAccuFile, strMakeModelRankAccuFile, strRankAccuCurvePicFile );
	cout << "Complete draw and save make and model rank accuracy data and curve picture." << endl << endl;

	return count;
}
