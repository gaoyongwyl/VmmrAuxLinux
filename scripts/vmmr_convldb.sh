#!/usr/bin/env sh
#
# Notice: All python script should be called from the python vmmr directory.
#         or else there will be error when converting the relative path to 
#         absolute path!
#
# Do two things:
# 1. crop patch or multi-patches and save into leveldb;
# 2. compute their image means
#
#

export GLOG_logtostderr=1

function Usage {
   echo "Usage:"
   echo $0 "DataSetVer(eg:V1) TransformType(e.g. AAuM）PrepTypeIDs(eg. 1:3) PatchIds(eg. -1:7) NewWidth(e.g. 150) FuncCode(1 both, 0 only comput mean)"
   echo .
}

if [ $# -ne 6 ]
then
   echo "The paramter number not correct!"
   echo $0 receive 6 parameters.
   echo .
   Usage
   exit 3  
fi

DATASET_VERSION=$1
TRANFORM_TYPE=$2

PREP_TYPEIDS=$3
PATCHIDS=$4
NEW_WIDTH=$5
DO_CONVERTLDBEX=$6

if [ $DO_CONVERTLDBEX -eq 0 ]
then
   echo.
   echo Onley compute image mean. 
   echo.
fi


#DATASET_VERSION=V1
#TRANFORM_TYPE=AAuM

#PREP_TYPEIDS="1:3"
#PATCHIDS="-1:7"
#NEW_WIDTH=150

#
# Function control
#DO_CONVERTLDBEX=0

#
#convert_imagesetex parameters:
TRAIN_TEST_AUG="0:0"
FUNC_CODE=2 #0:test only, 1:train, 2 : both
IS_NEWLDB=1 #0: append, 1: new one

#
# path
VMMR_DATA_PATH=/home/ygao/Projects/VehicleRecogntition/Data
CAFFE_TOOLS_PATH=/home/ygao/Projects/VehicleRecogntition/Code/caffe/build/tools
VMMR_PYTHON_PATH=/home/ygao/Projects/python/vmmr

#
# app tools
CONVERT_LDBEX=convert_imagesetex.bin


#
# python scripts
COMPUT_MEAN=DNNMakemodel_PrepareData.py



#####################################################
# function definition
#####################################################
function GetPatchCode {
    if [ $# -eq 1 ]
    then
	   case $1 in
	       -1)
		   echo "vface";;
	       4)
                   echo "LeftHLamp";;
	       5)
	           echo "RightHLamp";;
               6)
                   echo "FrontBumpLB";;
               7)
                   echo "FrontBumpRB";;
	       8)
                   echo "VehicleLogo";;
	       10)
                   echo "MidLineBot";;
               *)
	           echo "Unkown"
                   return 3;;
            esac
    else
	   echo "PatchCode function need at least 1 input."
	   echo ""
	   return 2
    fi
}

function GetPrepTypeCode {
    if [ $# -eq 1 ]
    then
	   case $1 in
	       0)
		   echo "";;
	       1)
                   echo "Gray";;
	       2)
	           echo "equalHist";;
               3)
                   echo "equalHistColor";;
               *)
	           echo "Unkown"
                   return 3;;
            esac
    else
	   echo "PatchCode function need at least 1 input."
	   echo ""
	   return 2
    fi 
}

#####################################################
## crop and save to ldb
####################################################
if [$DO_CONVERTLDBEX -gt 0 ]
then
  echo Now crop muliple-patches and save to ldb  ...
  echo How are : you

  $CAFFE_TOOLS_PATH/$CONVERT_LDBEX $DATASET_VERSION $PREP_TYPEIDS $PATCHIDS $NEW_WIDTH $TRAIN_TEST_AUG $FUNC_CODE $IS_NEWLDB

  echo "Complete patch cropping and ldb saving. :)"
fi

IFS=$':'
#####################################################
## compute image mean from train ldb
####################################################
echo Now compute image mean ...
OLD_PWD=`pwd`
cd $VMMR_PYTHON_PATH
echo Now current work dir is `pwd` . "(should be vmmr python dir)"
for ppti  in $PREP_TYPEIDS
do
	for pi in $PATCHIDS
	do
	        echo $PREP_TYPEIDS
		echo $PATCHIDS
	        echo new preprocess type `GetPrepTypeCode $ppti` and patch `GetPatchCode $pi`
		DataSetName=$DATASET_VERSION"_"$TRANFORM_TYPE"Color"`GetPrepTypeCode $ppti`
		echo Process computing image mean for $DataSetName
         	./$COMPUT_MEAN $DataSetName $pi $NEW_WIDTH
	done
done
cd $OLD_PWD
echo Now current work dir is `pwd`.
echo "Complete image mean computing. :)"


#
# Restore system IFS
#
IFS=$IFS.OLD
