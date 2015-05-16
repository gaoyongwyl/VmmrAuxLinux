// ExtractDNNFeature.cpp : 定义 DLL 应用程序的导出函数。
//


#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <string>
#include <vector>
#include <fstream>
//#include <io.h>
#include <math.h>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include <iomanip>


#include "caffe/vmmr/ExtractDNNFeature.h"
#include "caffe/VMMR_data_layer.h"

//#define _DEBUG_SHOW_SAVE

#define Round(a) (int(a+0.5))

string PatchKPIDToStr( int kpID ); 
void TrimSpace( string& str );

int DNNFeature::ExtractDNNFeat( cv::Mat& image, string& strFeatName,  vector<VMMRDType>& vfFeature )
{
	const boost::shared_ptr<Layer<VMMRDType> > data_layer = 
		this->m_DNNFeatureExtractionNet->layer_by_name( DATA_LAYTER_NAME );	
	VMMRImageDataLayer<VMMRDType>* pVmmrDataLayer = (VMMRImageDataLayer<VMMRDType>*)data_layer.get();

	if( pVmmrDataLayer == NULL ) {
		cout << "Invalid data layer ! Check you data layer name. It MUST be " << DATA_LAYTER_NAME << endl << endl;		
		return -1;
	}

	pVmmrDataLayer->SetCurrentImage( image );
	m_DNNFeatureExtractionNet->Forward(m_input_vec);

	const boost::shared_ptr<Blob<VMMRDType> > feature_blob = 
		this->m_DNNFeatureExtractionNet->blob_by_name( strFeatName );

	VMMRDType* pFeatBlobData;
	int nOffset = 0;
	int num_DimFeat = this->GetFeatureDim( strFeatName ); 
	int num_features = feature_blob->num();   // the image number in a batch.
	pFeatBlobData = feature_blob->mutable_cpu_data() +
          feature_blob->offset(nOffset);

	for( int n = 0; n < num_DimFeat; n ++ ) {
		vfFeature[n] = pFeatBlobData[n];
	}

	return 0;
}

int DNNFeature::ExtractDNNFeatToFile( cv::Mat& image, string& strFeatName,  string& strFeatFile )
{
	int dimFeature = this->GetFeatureDim( strFeatName );
	vector<VMMRDType> vfFeature(dimFeature );

	ExtractDNNFeat( image, strFeatName, vfFeature );
	this->WriteDNNFeatToFile( vfFeature, strFeatFile );

	return 0;
}

int VmmrDNN::ReadDNNFeatFromFile( string& strFeatFile, vector<VMMRDType>& vfFeature )
{
	FILE *hFeatFile = fopen( strFeatFile.c_str(), "r" );

	if( hFeatFile != NULL ) {
		int nFeatDims = 0;
		int count = 0;		
		if( fscanf( hFeatFile, "%d", &nFeatDims ) != 1 ) {
			printf( "ERROR: Read feature dims !	\n; " );
			return -1;
		}
		if( nFeatDims > 0 )	{
			float tmp = 0;			
			while( fscanf( hFeatFile, "%f", &tmp ) != 0 ) {
				vfFeature[count] = tmp;
				count ++;
				if( count >= nFeatDims ) {
					break;
				}
			}
		} else {
			printf( "ERROR: HOG Feat dims is %d in feature file %s. \n", strFeatFile.c_str() );
		}
		fclose( hFeatFile ); hFeatFile = NULL;

		if( count != nFeatDims ) {
			printf( "ERROR: the number of read value is not equal to the HOG dims \n" );
			return -2;
		}
	} else {
		printf( "ERROR: load feature file %s !\n", strFeatFile.c_str() );
		return -1;
	}

	return 0;
}

int DNNFeature::GetFeatureDim( string& strFeatName )
{
	const boost::shared_ptr<Blob<VMMRDType> > feature_blob = m_DNNFeatureExtractionNet
		->blob_by_name( strFeatName );
	int num_features = feature_blob->num();
	int dim_features = feature_blob->count() / num_features;  //should be 1

	return dim_features;
}

int VmmrDNN::WriteDNNFeatToFile( vector<VMMRDType>& vfFeature, string& strFeatFile )
{
	FILE *hFeatFile = fopen( strFeatFile.c_str(), "w" );

	if( hFeatFile != NULL )
	{
		fprintf( hFeatFile, "%d\n", vfFeature.size() );
		for( int n = 0; n < vfFeature.size(); n ++ )
		{
			fprintf( hFeatFile, "%f\n", vfFeature[n] );
		}
	}
	else
	{
		printf( "ERROR: save feature file %s !\n", strFeatFile.c_str() );
		return -1;
	}

	fclose( hFeatFile ); hFeatFile = NULL;

	return 0;

}

int DNNFeature::GetClassLabelNum()
{	
	string strClassLabelProbLayerName = CLASS_PROB_LAYER_NAME;
	int numClassLabels = this->GetFeatureDim( strClassLabelProbLayerName );
	return numClassLabels;	
}

int DNNFeature::DNNClassLabelProb( cv::Mat& image, vector<VMMRDType>& vfClassLabelProb, bool bWeighted )
{
	string strClassLabelProbLayerName = CLASS_PROB_LAYER_NAME;
	int nRetVal = this->ExtractDNNFeat( image, strClassLabelProbLayerName, vfClassLabelProb );
	return nRetVal;
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
		cout << "Errro: Unknown preprocess type id ! " << endl;
		return "ERROR";
	}
}

string PreprocTypeIDToStr( int pptid )
{
	switch( pptid ) {
	case VmmrDNN::PT_COLOR:
		return "Color";
	case VmmrDNN::PT_GRAY:
		return "Gray";
	case VmmrDNN::PT_EQUALIZEHIST:
		return "EqualizeHist";
	case VmmrDNN::PT_EQUALIZEHISTCOLOR:
		return "EqualizeHistColor";
	default:
		cout << "Errro: Unknown preprocess type id ! " << endl;
		return "ERROR";
	}

}

/*
Keep consistent with python version !
*/
string PatchKPIDToStr( int kpID )
{
	switch( kpID ){
	case VmmrDNN::MAKE_LogoArea:
		return "LogoArea";   // Make only classify
	case VmmrDNN::VF_VehicleFace:
		return "vface";      // Makemodel classify
	case  VmmrDNN::KP_WinGlassLT:
        return "WinGlassLT";  //   KP_WinGlassLT = 0 # Windshield Glass Left-top		
	case  VmmrDNN::KP_WinGlassRT:
        return "WinGlassRT";  //   #KP_WinGlassRT = 1 #: Windshield Glass Right-top
	case  VmmrDNN::KP_WinGlassLB:
        return "WinGlassLB";  //    KP_WinGlassLB = 2 #: Windshield Glass Left-bottom
	case  VmmrDNN::KP_WinGlassRB:
        return "WinGlassRB "; //    KP_WinGlassRB =     3 #: Windshield Glass Right-bottom
	case  VmmrDNN::KP_LeftHLamp:
        return "LeftHLamp";   //    KP_LeftHLamp = 4 #: Left Head Lamp center
	case  VmmrDNN::KP_RightHLamp:
        return "RightHLamp";  //    KP_RightHLamp = 5 #: Right Head Lamp center
	case  VmmrDNN::KP_FrontBumpLB:
        return "FrontBumpLB"; //    KP_FrontBumpLB = 6 #: Front Bumper Left Bottom corner
	case  VmmrDNN::KP_FrontBumpRB:
        return "FrontBumpRB"; //    KP_FrontBumpRB = 7 #: Front Bumper Right Bottom corner
	case  VmmrDNN::KP_VehicleLogo:
        return "VehicleLogo"; //    KP_VehicleLogo = 8 #: Vehicle Logo center
	case  VmmrDNN::KP_LicensePC:
        return "LicensePC";   //    KP_LicensePC = 9 #: License Plate center
	case  VmmrDNN::KP_MidLineBot:
        return "MidLineBot";  //    KP_MidLineBot = 10 #: Middle line bottom
    default:
        printf( "Invalid patch or key point id !" );
        return "BAD";
	} 
}

void IntToString (std::string& strInt, const int value)  
{  
    char buf[32];  
    snprintf (buf, sizeof (buf), "%d", value);  // snprintf is thread safe #include <stdio.h>  
    strInt.append (buf);  
}  

int DNNFeatMulti::ReadMakemodelMakeLabelMap( string& strMakemodelMakeLabelMapFile )
{
	this->m_vecMakemodelMakeLabelMap.clear();

	ifstream ifMakemodelMakeMap( strMakemodelMakeLabelMapFile.c_str() );
	if( !ifMakemodelMakeMap.is_open() ) {
		cout << "Open makemodel make file failed ! " << strMakemodelMakeLabelMapFile << endl;
		return -1;
	}
	int count = 0;
	int maxMakeLabel = 0;
	while( !ifMakemodelMakeMap.eof() ) {
		string strLine;
		getline( ifMakemodelMakeMap, strLine );
		TrimSpace( strLine );
		if( strLine.empty() ) {
			continue;
		}
		string strLabesPart = strLine.substr( strLine.find_first_of( " " ) + 1 );
		TrimSpace( strLabesPart );
		string strMakemodelLabel = strLabesPart.substr( 0, strLabesPart.find_first_of( " " ) );
		string strMakeLabel = strLabesPart.substr( strLabesPart.find_first_of( " " ) + 1 );
		TrimSpace( strMakemodelLabel );
		TrimSpace( strMakeLabel );
		MakemodelMakeLabelMap labelMap;
		labelMap.nMakemodelLabel = atoi( strMakemodelLabel.c_str() );
		labelMap.nMakeLabel = atoi( strMakeLabel.c_str() );

		if( maxMakeLabel < labelMap.nMakeLabel ) {
			maxMakeLabel = labelMap.nMakeLabel;
		}
		if( labelMap.nMakemodelLabel != count ) {
			cout << "Make model label is not sequential ! which will result in error later ! " << endl;
			cout << "Current makemodel label : " << labelMap.nMakemodelLabel << " vs. " << " count number: " << count << endl << endl;
			return -2;
		}

		this->m_vecMakemodelMakeLabelMap.push_back( labelMap );

		count ++;
	}
	ifMakemodelMakeMap.close();

	cout << endl << "Complete makemodel - make label map file read " << endl;	
	cout << "Max make label is " << maxMakeLabel << "( " <<maxMakeLabel+1 << " classes ) " <<  endl;
	cout << "Total makemodel label number is " << count;
	
	return 0;
}


int DNNFeatMulti::ReadConfigInfo( string& strDnnConfigFile )
{
	ifstream ifDnnConfigFile;

	ifDnnConfigFile.open( strDnnConfigFile.c_str(), ios::in );
	if( !ifDnnConfigFile.is_open() ) {
		cout << "Failed to open Dnn config file: " << strDnnConfigFile << endl;
		return -1;
	}
	
	int nTotalPatches = -1;
	bool bPatchIDStart = false;
	bool bModelSuffixGotten = false;
	while( !ifDnnConfigFile.eof() ) {
		string strLine;
		getline( ifDnnConfigFile, strLine );
		
		//remove comment or empty lines firstly 
		size_t nCommentTag = strLine.find_first_of( "#" );
		if( nCommentTag != string::npos ) {
			strLine = strLine.substr( 0, nCommentTag );
		}
		TrimSpace( strLine );
		if( strLine.size() <= 0 ) {
			continue;
		}

		//parse line content
		//------------------------------------------------
		if( bModelSuffixGotten == false ) {
			size_t nTagPos = strLine.find( "Suffix" );
			if( nTagPos != string::npos ) {
				nTagPos = strLine.find( ":" );
				if( nTagPos != string::npos ) {
					string strIterNum = strLine.substr( nTagPos + 1 );
					this->m_nDnnModelIterNum = atoi( strIterNum.c_str() );
					bModelSuffixGotten = true;
				}
			}
		}

		if( bPatchIDStart == false ) {
			size_t nTagPos = strLine.find( "Total" );
			if( nTagPos != string::npos ) {
				nTagPos = strLine.find( ":" );
				if( nTagPos == string::npos ) {
					cout << "In config file, Total format not correct ! " << strLine << endl;
					return -1;
				}
				string strTotal = strLine.substr( nTagPos+1 );
				nTotalPatches = atoi( strTotal.c_str() );
				bPatchIDStart = true;
			}
		} else {
			int nPreprocTypeID = -128;
			int nPatchID = -128;
			float fModelWeight = -FLT_MIN;
			int nStdVFWdith = 0;
			
			string strPreprocTypeID = "", strPatchID = "", strStdVFWidth = "", strModelWeight = "";
			size_t nSpaceSepPos = strLine.find_first_of( " " );
			if( nSpaceSepPos == string::npos ) {
				continue;
			}			

			// preprocess type id:
			strPreprocTypeID = strLine.substr( 0, nSpaceSepPos );
			nPreprocTypeID = atoi( strPreprocTypeID.c_str() );
			
			if( nPreprocTypeID == VmmrDNN::PT_GRAY ) {
				this->m_bHasGrayDnn = true;
			} else if( nPreprocTypeID == VmmrDNN::PT_EQUALIZEHIST ) {
				this->m_bHasEqualHistDnn = true;
			}

			string strRemainLine = strLine.substr( nSpaceSepPos + 1 );
			TrimSpace( strRemainLine );
			nSpaceSepPos = strRemainLine.find_first_of( " " ); 
			if( nSpaceSepPos != string::npos ) {
				strPatchID = strRemainLine.substr( 0, nSpaceSepPos );
				nPatchID = atoi( strPatchID.c_str() );

				strRemainLine = strRemainLine.substr(  nSpaceSepPos + 1 );
				TrimSpace( strRemainLine );
				nSpaceSepPos = strRemainLine.find( " " );  //"<NewWidth> <Weight>" or "<NewWidth>"
				if( nSpaceSepPos != string::npos ) {
					strStdVFWidth = strRemainLine.substr( 0, nSpaceSepPos );
					strModelWeight = strRemainLine.substr ( nSpaceSepPos + 1 );

					nStdVFWdith = atoi( strStdVFWidth.c_str() );
					fModelWeight = atof( strModelWeight.c_str() );					
				} else {
					cout << "Error: Not found weight when parse config line !" << endl << endl;
				}

			} else {
				cout << "Error parse config line ! " << endl << endl;
				return -2;
			}
			
			if( nPatchID <= 10 && nPatchID >= -2 ) {
				this->m_vecPreprocTypeIDs.push_back( nPreprocTypeID );
				this->m_vecPatchIDs.push_back( nPatchID );
				this->m_vecfModelWeights.push_back( fModelWeight );
				m_vecStdVFWidths.push_back( nStdVFWdith );
			}
		}
	}
	
	ifDnnConfigFile.close();

	if( this->m_vecPatchIDs.size() != nTotalPatches ) {
		cout << "Warning : total patch number and actually read id num different! " << m_vecPatchIDs.size() <<
			" vs. " << nTotalPatches << endl;
		return -1;
	}

	// print config info here:
	cout << endl << "Total patch number is " << m_vecPatchIDs.size() << endl;

	cout << endl << endl << "Multi DNN makemodel ( or make if provided ) information : " << endl;
	cout << "=========================================================================" << endl;
	cout << " " << setw(5) << "PTID" << " " << setw(10) << "Preproc Type" <<  " " << setw(5) << "ID"<< " : " << setw( 16 ) << "Model Name"  \
			<< "  (" << setw( 5 ) << "std VF width" << ", "  << setw( 8 ) << "Weight"  << ")" << endl;
	cout << "-------------------------------------------------------------------------" << endl;
	for( int n = 0; n < this->m_vecPatchIDs.size(); n++ ) {
		cout << " " << setw(5) << this->m_vecPreprocTypeIDs[n] << " " << setw(10) << PreprocTypeIDToStr( this->m_vecPreprocTypeIDs[n]) \
			<< " " << setw(5) << this ->m_vecPatchIDs[n] << " : " << setw( 16 ) <<  PatchKPIDToStr( this->m_vecPatchIDs[n])  \
			<< "  (" << setw( 5 ) << this->m_vecStdVFWidths[n] << ", " << setw( 8 ) << this->m_vecfModelWeights[n] << ")" << endl;
	}
	cout << endl<<endl;	
	cout << "Model iteration number : " << this->m_nDnnModelIterNum << endl << endl;

	return 0;
}

int DNNFeatMulti::InitializeDNN(string strProtoModelFolder, 
		DNNFeature::COMPUTE_MOD enCompMod, int devid )
{
    //key point list
	m_nDnnModelIterNum = 0;
	this->m_vecPatchIDs.clear();
	this->m_vecfModelWeights.clear();
	this->m_vecStdVFWidths.clear();
	this->m_vecPreprocTypeIDs.clear();

	m_bHasGrayDnn = false;  //preprocess type
	m_bHasEqualHistDnn = false;

	//Read configuration info:
	string strVmmrConfigInfoFile = strProtoModelFolder + "VmmrConfig.ini";
	int nRetVal = this->ReadConfigInfo( strVmmrConfigInfoFile );
	if( nRetVal < 0 ) {
		cout << endl << endl << "Failed to read DNN configuration file !" << endl <<endl;
		return -1;
	}

	char szIterNum[100];
	sprintf( szIterNum, "%d", this->m_nDnnModelIterNum );
	string strIterNum = szIterNum;

	for( int n = 0; n < m_vecPatchIDs.size(); n ++ ) {
		string strPatchName = PatchKPIDToStr( m_vecPatchIDs[n] );
		string strPreprocCode = PreprocTypeIDToCode( this->m_vecPreprocTypeIDs[n] );
		char szStdVFWidth[100];
		sprintf( szStdVFWidth, "_%d", this->m_vecStdVFWidths[n]);
		string strStdVFWdith = szStdVFWidth;
		string strProtoFile = strProtoModelFolder + "vmakemodel_" + strPreprocCode + "_" +  strPatchName + strStdVFWdith + "_train_test_val.prototxt";
		string strModelFile = strProtoModelFolder + "vmakemodel_" + strPreprocCode + "_" +  strPatchName + strStdVFWdith + "_iter_" +  strIterNum + ".caffemodel";
		if( m_vecPatchIDs[n] == VmmrDNN::MAKE_LogoArea ) {
			strProtoFile = strProtoModelFolder + "vmake_" + strPreprocCode + "_" + strPatchName + strStdVFWdith + "_test_val.prototxt";
			strModelFile = strProtoModelFolder + "vmake_" +strPreprocCode + "_" +  strPatchName + strStdVFWdith + "_iter_" +  strIterNum;

			string strMakemodelMakeLabeMapFile = strProtoModelFolder + "MakemodelMakeLabelMap.txt";
			int nRV = this->ReadMakemodelMakeLabelMap( strMakemodelMakeLabeMapFile );
			if( nRV < 0 ) {
				return nRV;
			}
		}

		if( access( strProtoFile.c_str(), 0 ) == -1 ) {
			cout << "Proto file not exist! : " << strProtoFile << endl;
			return -1;
		}
		if( access( strModelFile.c_str(), 0 ) == -1 ) {
			cout << "MOdel file not exist! : " << strModelFile << endl;
			return -1;
		}

		DNNFeature dnnFeat;
		dnnFeat.InitializeDNN( strProtoFile, strModelFile, enCompMod, devid );

		this->m_VmmrDnnFeats.push_back( dnnFeat );
	}
	
	cout << endl << endl << "Initialize dnn model successfully ! " << endl << endl;

	m_enVmmrDnnType = MULTI;

	return 0;
}


//should feed different image patch to their DNN
int DNNFeatMulti::ExtractDNNFeat( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, string& strFeatName,  vector<VMMRDType>& vfFeature )
{
	cout << "Not update now. Don't use it ! " << endl;
	return -2;

	int nFeatDim = this->GetFeatureDim( strFeatName );
	if( vfFeature.size() < nFeatDim ) {
		vfFeature.resize( nFeatDim );
	}
	cv::Mat imageStandScale;
	vector<CvPoint2D32f> vecKeyPointsScaled( vecKeyPoints.size() );
	copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	int nRetVal = 
		GetStandarScaleImage( image, imageStandScale, vecKeyPointsScaled, VMMR_INFLAT_COEFF, VMMR_NewWidth );
	if( nRetVal < 0 ) {
		cout << "Error: GetStandarScaleImage " << endl;
		return -1;
	}

	int index = 0;
	for( int n = 0; n < this->m_VmmrDnnFeats.size(); n ++ ) {
		int nCurrFeatDim = m_VmmrDnnFeats[n].GetFeatureDim( strFeatName );
		vector<VMMRDType> vfTempFeat( nCurrFeatDim );

		cv::Mat imagePatchCropped;
		int RetVal = CropPatchByKeyPointID( imageStandScale, imagePatchCropped, vecKeyPointsScaled,this->m_vecPatchIDs[n] );

		m_VmmrDnnFeats[n].ExtractDNNFeat( imagePatchCropped, strFeatName, vfTempFeat );
		memcpy( &(vfFeature[index]), &(vfTempFeat[0]), sizeof(VMMRDType) * nCurrFeatDim );
		index += nCurrFeatDim;
	}

	return 0;
}


int DNNFeatMulti::ExtractDNNFeatToFile( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, string& strFeatName,  string& strFeatFile )
{
	int dimFeature = this->GetFeatureDim( strFeatName );
	vector<VMMRDType> vfFeature(dimFeature );

	ExtractDNNFeat( image, vecKeyPoints, strFeatName, vfFeature );
	this->WriteDNNFeatToFile( vfFeature, strFeatFile );

	return 0;
}

int DNNFeatMulti::GetClassLabelNum()
{	
	int numClassLabels = this->m_VmmrDnnFeats[0].GetClassLabelNum();

	for( int n = 1; n < this->m_VmmrDnnFeats.size(); n++ ) {
		int _numClassLabels = this->m_VmmrDnnFeats[n].GetClassLabelNum();
		if( _numClassLabels > numClassLabels ) {
			numClassLabels = _numClassLabels;
		}
	}

	return numClassLabels;
}

//should feed different image patch to their DNN
int DNNFeatMulti::DNNClassLabelProb( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, vector<VMMRDType>& vfClassLabelProb, bool bWeighted )
{
	int numLabelNum = this->GetClassLabelNum();
	if( vfClassLabelProb.size() < numLabelNum ) {
		vfClassLabelProb.resize( numLabelNum, 0 );
	}
	//Notice: 
	// 1) preprocessing is done on original big color image. So we should do this samely with same sequence. Especially for HE:
	// 2) Input image should always be original big color image.
	cv::Mat grayImage1ch, grayImage, equalHistImage1ch, equalHistImage, equalHistColorImage;
	cv::Mat imageStandScaleColor, imageStdScaleGray, imageStdScaleEqualHist, imageStdScaleEqualHistColor;
	vector<CvPoint2D32f> vecKeyPointsScaled( vecKeyPoints.size() );

	//for color image
	int nRetVal = -1;
	copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	nRetVal = GetStandarScaleImage( image, imageStandScaleColor, vecKeyPointsScaled, VMMR_INFLAT_COEFF, VMMR_NewWidth );
	if( nRetVal < 0 ) { cout << "Error: GetStandarScaleImage for  imageStandScaleColor" << endl; return -1; }
	//Fill license plat area
	this->FillLicensePlate( imageStandScaleColor, vecKeyPointsScaled[KP_LicensePC], VMMR_NewWidth );

#if 0
	imshow( "color std img", imageStandScaleColor );
	waitKey();
#endif
	//for gray image:
	cv::cvtColor( image, grayImage1ch, CV_BGR2GRAY );     //conver to 1 channel gray
	cv::cvtColor( grayImage1ch, grayImage, CV_GRAY2BGR ); //Convert three channel to feed dnn
	copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	nRetVal = GetStandarScaleImage( grayImage, imageStdScaleGray, vecKeyPointsScaled, VMMR_INFLAT_COEFF, VMMR_NewWidth );
	if( nRetVal < 0 ) { cout << "Error: GetStandarScaleImage for  imageStdScaleGray" << endl; return -1; }
	//Fill license plat area
	this->FillLicensePlate( imageStdScaleGray, vecKeyPointsScaled[KP_LicensePC], VMMR_NewWidth );

	//for equal hist image:
	cv::equalizeHist( grayImage1ch, equalHistImage1ch );
	cv::cvtColor( equalHistImage1ch, equalHistImage, CV_GRAY2BGR );//Convert three channel to feed dnn
	copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	nRetVal = GetStandarScaleImage( equalHistImage, imageStdScaleEqualHist, vecKeyPointsScaled, VMMR_INFLAT_COEFF, VMMR_NewWidth );
	if( nRetVal < 0 ) { cout << "Error: GetStandarScaleImage for imageStdScaleEqualHist" << endl; return -1; }
	//Fill license plat area
	this->FillLicensePlate( imageStdScaleEqualHist, vecKeyPointsScaled[KP_LicensePC], VMMR_NewWidth );

	//for equal hist color image
	vector<cv::Mat> vecImage1Ch;
	cv::split( image, vecImage1Ch );
	cv::equalizeHist( vecImage1Ch[0], vecImage1Ch[0] );
	cv::equalizeHist( vecImage1Ch[1], vecImage1Ch[1] );
	cv::equalizeHist( vecImage1Ch[2], vecImage1Ch[2] );
	cv::merge( vecImage1Ch, equalHistColorImage );
	copy(vecKeyPoints.begin(), vecKeyPoints.end(), vecKeyPointsScaled.begin() );
	nRetVal = GetStandarScaleImage( equalHistColorImage, imageStdScaleEqualHistColor, vecKeyPointsScaled, VMMR_INFLAT_COEFF, VMMR_NewWidth );
	if( nRetVal < 0 ) { cout << "Error: GetStandarScaleImage for imageStdScaleEqualHist" << endl; return -1; }
	//Fill license plat area
	FillLicensePlate( imageStdScaleEqualHistColor, vecKeyPointsScaled[KP_LicensePC], VMMR_NewWidth );

	//note: here will only resize image in VMMR_NewWidth scale. Other resize will be done by our vmmr data layer! 
	memset( &(vfClassLabelProb[0]), 0, sizeof( VMMRDType ) * numLabelNum );

	for( int n = 0; n < this->m_VmmrDnnFeats.size(); n ++ ) {
		vector<VMMRDType> vfTempLabelProbs( numLabelNum );
		vector<VMMRDType> vfModelWeights( numLabelNum, this->m_vecfModelWeights[n] );

		cv::Mat imagePatchCropped;
		int RetVal = -1;
		if( this->m_vecPreprocTypeIDs[n] == VmmrDNN::PT_COLOR ) {
			RetVal = CropPatchByKeyPointID( imageStandScaleColor, imagePatchCropped, vecKeyPointsScaled,this->m_vecPatchIDs[n] );
		} else if( this->m_vecPreprocTypeIDs[n] == VmmrDNN::PT_GRAY ) {
			RetVal = CropPatchByKeyPointID( imageStdScaleGray, imagePatchCropped, vecKeyPointsScaled,this->m_vecPatchIDs[n] );
		} else if( this->m_vecPreprocTypeIDs[n] == VmmrDNN::PT_EQUALIZEHIST ) {
			RetVal = CropPatchByKeyPointID( imageStdScaleEqualHist, imagePatchCropped, vecKeyPointsScaled,this->m_vecPatchIDs[n] );
		} else if( this->m_vecPreprocTypeIDs[n] == VmmrDNN::PT_EQUALIZEHISTCOLOR ) {
			RetVal = CropPatchByKeyPointID( imageStdScaleEqualHistColor, imagePatchCropped, vecKeyPointsScaled,this->m_vecPatchIDs[n] );
		} else {
			cout << "Error: Unsupported preprocessing type appread !" << "  Line: " << __LINE__ << endl;
			return -2;
		}
		if( RetVal < 0 ) {
			cout << "Crop patch by key point id failed !" << endl;
			return -3;
		}
	
		//#define _DEBUG_SHOW_SAVE
#ifdef _DEBUG_SHOW_SAVE
		imwrite( "/home/ygao/Projects/vmmrTest0.bmp", image );
		imwrite( "/home/ygao/Projects/vmmrTest1.bmp", imageStandScaleColor );
		imwrite( "/home/ygao/Projects/vmmrTest2.bmp", imageStdScaleGray );
		imwrite( "/home/ygao/Projects/vmmrTest3.bmp", imageStdScaleEqualHist );
		imwrite( "/home/ygao/Projects/vmmrTest4.bmp", imagePatchCropped );
		char szTitle[2048];
		sprintf( szTitle, "%s", PatchKPIDToStr( this->m_vecPatchIDs[n] ).c_str() );
		imshow( szTitle, imagePatchCropped  );
		waitKey();
#endif //_DEBUG

		this->m_VmmrDnnFeats[n].DNNClassLabelProb( imagePatchCropped, vfTempLabelProbs );

		if( m_vecPatchIDs[n] == VmmrDNN::MAKE_LogoArea ) {  // current is make dnn classifier 
			for( int m = 0; m < vfClassLabelProb.size(); m ++ ) {
				int _makeLabel = this->m_vecMakemodelMakeLabelMap[m].nMakeLabel;
				float fWScore = vfTempLabelProbs[_makeLabel] * this->m_vecfModelWeights[n];
				vfClassLabelProb[m] += fWScore;
			}

		} else {  // makemodel dnn classifier
			//multiply weights:
			transform( vfModelWeights.begin(), vfModelWeights.end(), vfTempLabelProbs.begin(), vfTempLabelProbs.begin(), multiplies<VMMRDType>());

			//Add to vfClassLabelProb;
			transform( vfTempLabelProbs.begin(), vfTempLabelProbs.end(), vfClassLabelProb.begin(),vfClassLabelProb.begin(), plus<VMMRDType>());
		}
	}

	vector<VMMRDType> vfTempLabelProbs( numLabelNum, 1.0f/this->m_VmmrDnnFeats.size() );
	transform( vfClassLabelProb.begin(), vfClassLabelProb.end(), vfTempLabelProbs.begin(), vfClassLabelProb.begin(), multiplies<VMMRDType>());

	return 0;
}

//invoke this after resize and before crop
int DNNFeatMulti::FillLicensePlate( cv::Mat& image, CvPoint2D32f& ptLicensePlateCenter, int iNewWidth )
{
	float fDeltaX = iNewWidth * 0.118;
    float fDeltaY = iNewWidth * 0.04;
	int iRowStart = int( ptLicensePlateCenter.y - fDeltaY );
	int iRowEnd = int( ptLicensePlateCenter.y + fDeltaY );
	int iColStart = int( ptLicensePlateCenter.x - fDeltaX );
	int iColEnd = int( ptLicensePlateCenter.x + fDeltaX );

	for( int r= iRowStart; r < iRowEnd; r ++ ) {
		for( int c = iColStart; c < iColEnd; c++ ) {
			image.at<Vec3b>( r, c ) = Vec3b( 128, 128, 128 );
		}
	}
	return 0;
}

int DNNFeatMulti::GetStandarScaleImage( cv::Mat& image, cv::Mat& imageStandScale, vector<CvPoint2D32f>& vecKeyPoints, float INFLAT_COEFF, float NewWidth )
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
 int DNNFeatMulti::CropPatchByKeyPointID( cv::Mat& image, cv::Mat& imagePatchCropped, vector<CvPoint2D32f>& vecPtKeyPoint, int nPatchID )
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

	VMMRKPID enPatchID = (VMMRKPID)nPatchID;
	if ( enPatchID == MAKE_LogoArea ) 
	{
		coeff_half_x_l = 0.05f;
		coeff_half_x_r = 0.05f;
		coeff_half_y_u = 0.05f;
		coeff_half_y_d = 0.05f;
	} else if ( enPatchID == VF_VehicleFace ) { // crop vehicel face
		coeff_half_x_l = 0.34;
		coeff_half_x_r = 0.34;
		coeff_half_y_u = 0.14;
		coeff_half_y_d = 0.15;
	}else if(  enPatchID == KP_VehicleLogo ) {
		coeff_half_x_l = 0.16;
		coeff_half_x_r = 0.16;
		coeff_half_y_u = 0.12;
		coeff_half_y_d = 0.06;
	} else if ( enPatchID == KP_LeftHLamp ) {
		coeff_half_x_l = 0.09;
		coeff_half_x_r = 0.30;
		coeff_half_y_u = 0.16;
		coeff_half_y_d = 0.09;
	} else if(  enPatchID == KP_RightHLamp ) {
		coeff_half_x_l = 0.30;
		coeff_half_x_r = 0.09;
		coeff_half_y_u = 0.16;
		coeff_half_y_d = 0.09;
	} else if ( enPatchID == KP_FrontBumpLB ) {
		coeff_half_x_l = 0.03;
		coeff_half_x_r = 0.36;
		coeff_half_y_u = 0.18;
		coeff_half_y_d = 0.05;       
	} else if ( enPatchID == KP_FrontBumpRB ) {
		coeff_half_x_l = 0.36;
		coeff_half_x_r = 0.03;
		coeff_half_y_u = 0.18;
		coeff_half_y_d = 0.05;             
	}else if ( enPatchID == KP_LicensePC ) { // # Not used!
		;
	}else if( enPatchID == KP_MidLineBot ) {
		coeff_half_x_l = 0.18;
		coeff_half_x_r = 0.18;
		coeff_half_y_u = 0.20;
		coeff_half_y_d = 0.03;      
	} else {
		cout << "Invalid crop patch ID " <<  nPatchID << " Name: " << PatchKPIDToStr( nPatchID );
	}
	float xc = -1, yc = -1;
	if ( nPatchID < 0 ) {
		xc = vecPointCoordNew[VmmrDNN::KP_VehicleLogo].x;
		yc = vecPointCoordNew[VmmrDNN::KP_VehicleLogo].y;
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
		return -1;;
	} else {
		imagePatchCropped = imageWithBorder( Rect(  xc-Half_x_left,  yc-Half_y_up, Half_x_left + Half_x_right, 
			Half_y_up + Helf_y_down ) );
   	    return 0;
	}
}

 void TrimSpace( string& str )
{
	str.erase( 0, str.find_first_not_of("\r\t\n ")); 
	str.erase( str.find_last_not_of("\r\t\n ") + 1);
}
