#ifndef _EXTRACT_DNN_FEATURE_H_
#define _EXTRACT_DNN_FEATURE_H_

#include <string>
#include "opencv2/opencv.hpp"
#include "caffe/net.hpp"
#include "cv.h"

using namespace std;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)

#ifdef EXTRACTDNNFEATURE_EXPORTS
#define API_DECLSPEC    __declspec(dllexport)
#else
#define API_DECLSPEC    __declspec(dllimport)
#endif

//#define TIME_PROFILE

#ifdef TIME_PROFILE
	#define _TS_          double  t = (double)getTickCount();
	#define _TE_(action)  t = ((double)getTickCount() - t)/getTickFrequency(); \
						  cout << endl << #action << " time :" << t << " seconds" << endl;
#else
	#define _TS_ 
	#define _TE_(name) 
#endif//

#define DATA_LAYTER_NAME "data"   // data layer name should be this in proto file!!!
#define CLASS_PROB_LAYER_NAME "prob" // class prob layer name should be this in proto file!!!

#define VMMR_INFLAT_COEFF 0.1f
#define VMMR_NewWidth 380

typedef float VMMRDType; //caffe abstract instantiate float and double two types. USE float for current test bed!!!

class VmmrDNN{
public:
	typedef enum VMMRDNN_TYPE{
		SINGLE = 0,
		MULTI = 1
	}VMMRDNN_TYPE;
		typedef enum COMPUTE_MOD{
		CPU=0,
		GPU=1
	}COMPUTE_MOD;
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
	// for feature extraction
	//virtual int ExtractDNNFeat( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, string& strFeatName,  vector<VMMRDType>& vfFeature ) = 0;
	//virtual int ExtractDNNFeatToFile( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, string& strFeatName,  string& strFeatFile ) = 0;
	virtual int GetFeatureDim( string& strFeatName ) = 0;
	virtual int GetBatchSize() = 0;

	// for classification
	virtual int GetClassLabelNum() = 0;
	//virtual int DNNClassLabelProb( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, vector<VMMRDType>& vfClassLabelProb, bool bWeighted = false ) = 0;

	// common utilities
	virtual COMPUTE_MOD GetComputeMode() = 0;
	virtual int GetDevID()  = 0;

public:
	static int WriteDNNFeatToFile( vector<VMMRDType>& vfFeature, string& strFeatFile );
	static int ReadDNNFeatFromFile( string& strFeatFile, vector<VMMRDType>& vfFeature );

	virtual VMMRDNN_TYPE GetVmmrDnnType() = 0;		
	
protected:
	VMMRDNN_TYPE m_enVmmrDnnType;
};

class DNNDiagnose{
public:
	~DNNDiagnose() {

	}
public:
	DNNDiagnose( string strNetProto ) {
		boost::shared_ptr<Net<VMMRDType> > feature_extraction_net(
					new Net<VMMRDType>( strNetProto ) );
			m_DNNFeatureExtractionNet = feature_extraction_net;		
	}	
	int GetFeatureDim( string& strFeatName )
	{
		const boost::shared_ptr<Blob<VMMRDType> > feature_blob = m_DNNFeatureExtractionNet
			->blob_by_name( strFeatName );
		int num_features = feature_blob->num();
		int dim_features = feature_blob->count() / num_features;  //should be 1

		return dim_features;
	}

	boost::shared_ptr<Net<VMMRDType> > m_DNNFeatureExtractionNet; //automatically delete
};

class  DNNFeature : public VmmrDNN{
public:
	/*
	Usage: 
	string strDNNProtoFile: feature_extraction_proto_file  
	string strPretrainedModel: pretrained_net_param
	string strFeatName: extract_feature_blob_name
	COMPUTE_MOD enCompMod: [CPU/GPU]  
	int devid: [DEVICE_ID=0]";
	*/
	int InitializeDNN(string strDNNProtoFile, 
		string strPretrainedModel, 		
		COMPUTE_MOD enCompMod = CPU, 
		int devid = 0 ){
			m_strDNNProtoFile = strDNNProtoFile;
			m_strPretrainedModel = strPretrainedModel;			
			m_enCompMod = enCompMod;
			m_devid = devid;

			switch( m_enCompMod ) {
			case CPU:
				Caffe::set_mode(Caffe::CPU);
				break;
			case GPU:
				Caffe::set_mode(Caffe::GPU);
				Caffe::SetDevice(m_devid);
				break;
			default:
				Caffe::set_mode(Caffe::CPU);
			}

			boost::shared_ptr<Net<VMMRDType> > feature_extraction_net(
                                new Net<VMMRDType>( m_strDNNProtoFile, caffe::TEST ) );
			m_DNNFeatureExtractionNet = feature_extraction_net;
			m_DNNFeatureExtractionNet->CopyTrainedLayersFrom( m_strPretrainedModel );

			m_enVmmrDnnType = SINGLE;

			return 0;
	};

	int ExtractDNNFeat( cv::Mat& image, string& strFeatName,  vector<VMMRDType>& vfFeature );
	int ExtractDNNFeatEx( vector<cv::Mat>& imageBatch, string& strFeatName,  vector<vector<VMMRDType> >& vfFeatureBatch );
	
	int ExtractDNNFeatToFile( cv::Mat& image, string& strFeatName,  string& strFeatFile );
	int GetFeatureDim( string& strFeatName );
	int GetBatchSize( );

	int GetClassLabelNum();
	int DNNClassLabelProb( cv::Mat& image, vector<VMMRDType>& vfClassLabelProb, bool bWeighted = false );
	int DNNClassLabelProbEx( vector<cv::Mat>& imageBatch, vector<vector<VMMRDType> >& vfClassLabelProbBatch, bool bWeighted = false );
	
	string& GetDNNProtoFile() { return m_strDNNProtoFile; };
	string& GetPretrainedModel() { return m_strPretrainedModel; };
	COMPUTE_MOD GetComputeMode() { return m_enCompMod; };
	int GetDevID() { return m_devid; };


	VMMRDNN_TYPE GetVmmrDnnType() {
		return this->m_enVmmrDnnType;
	}
private:	
	string m_strDNNProtoFile;
	string m_strPretrainedModel;
	COMPUTE_MOD m_enCompMod;
	int m_devid;
	vector<Blob<float>*> m_input_vec;
	boost::shared_ptr<Net<VMMRDType> > m_DNNFeatureExtractionNet;
}; //end class




class  DNNFeatMulti : public VmmrDNN {
public:
	typedef struct MakemodelMakeLabelMap{
		int nMakemodelLabel;
		int nMakeLabel;
	}MakemodelMakeLabelMap;
public:
	int InitializeDNN(string strProtoModelFolder, 
		DNNFeature::COMPUTE_MOD enCompMod = DNNFeature::CPU, 
		int devid = 0 );

public:
	int ExtractDNNFeat( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, string& strFeatName,  vector<VMMRDType>& vfFeature );
	int ExtractDNNFeatToFile( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, string& strFeatName,  string& strFeatFile );
	int GetFeatureDim( string& strFeatName ){
		int totalDim = 0;
		for( int n = 0; n < this->m_VmmrDnnFeats.size(); n ++ ){
			totalDim += m_VmmrDnnFeats[n].GetFeatureDim( strFeatName );
		}
		return totalDim;
	};

	int GetBatchSize(){
		string strProbLayerName = CLASS_PROB_LAYER_NAME;
		int batch_size = m_VmmrDnnFeats[0].GetBatchSize();
		for( int n =0; n < this->m_VmmrDnnFeats.size(); n++ ) {
			if( batch_size != m_VmmrDnnFeats[n].GetBatchSize() )
			{
				cout << "Batch size not equal for " << n << "-th DNN net" << endl;
				cout << "Please check it !" << endl;
				return -1;
			}
		}
		return batch_size;
	}

	int GetClassLabelNum();
	int DNNClassLabelProb( cv::Mat& image, vector<CvPoint2D32f>& vecKeyPoints, vector<VMMRDType>& vfClassLabelProb, bool bWeighted = false );
	int DNNClassLabelProbEx( vector<cv::Mat>& imageBatch, vector<vector<CvPoint2D32f> >& vecKeyPoints, vector<vector<VMMRDType> >& classLabelProbBatch, bool bWeighted = false );

	DNNFeature::COMPUTE_MOD GetComputeMode() { return m_enComputeMod; };
	int GetDevID() { return m_computeDevId; };
	string& GetProtoModelFolder() {
		return m_strProtoModelFolder;
	};

	VMMRDNN_TYPE GetVmmrDnnType() {
		return this->m_enVmmrDnnType;
	}

protected:
	int FillLicensePlate( cv::Mat& image, CvPoint2D32f& ptLicensePlateCenter, int iNewWidth );
	int CropPatchByKeyPointID( cv::Mat& image, cv::Mat& imagePatchCropped, vector<CvPoint2D32f>& vecPtKeyPoint, int nPatchID );
	int GetStandarScaleImage( cv::Mat& image, cv::Mat& imageStandScale, vector<CvPoint2D32f>& vecKeyPoints, float INFLAT_COEFF, float NewWidth );
	int ReadConfigInfo( string& strDnnConfigFile );
	int ReadMakemodelMakeLabelMap( string& strMakemodelMakeLabelMapFile );
private:
	string m_strProtoModelFolder;
	vector<DNNFeature> m_VmmrDnnFeats;
	DNNFeature::COMPUTE_MOD m_enComputeMod; //GPU or CPU
	int m_computeDevId;

	vector<float> m_vecfModelWeights;
	vector<int> m_vecPatchIDs;
	vector<int> m_vecStdVFWidths;
	vector<int> m_vecPreprocTypeIDs;
	unsigned int m_nDnnModelIterNum;
	vector<MakemodelMakeLabelMap> m_vecMakemodelMakeLabelMap;

	bool m_bHasGrayDnn;  //preprocess type, used to remove un-necessary preprocessing.
	bool m_bHasEqualHistDnn;
	bool m_bHasEqualHistColorDnn;
}; //end clas 
#endif //_EXTRACT_DNN_FEATURE_H_
