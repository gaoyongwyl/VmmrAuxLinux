#!/usr/bin/env sh
#
# Notice: All python script should be called from the python vmmr directory.
#         or else there will be error when converting the relative path to 
#         absolute path!
#
# train dnn and test on validation set.
# 
#
# usage: DNNMakemodel_Train.py [-h] [-rst ResSolverStateIterNum] [-ttlid TrainTestListID] [-tin TrainIterNum] [-cm COMPUT_MODE] [-gid GPU_DevID]
#                             [-shri ShowResultImages] [-expid CurExprID] [-doDnn DoDnnTrain]
#                             DATASET_NAME PATCH_ID NewWidth
# 

export GLOG_logtostderr=1

DATASET_VERSION=V1
TRANFORM_TYPE=AAuM

PREP_TYPEIDS="1:3"
PATCHIDS="-1:7"
NEW_WIDTH=150

COMPUT_MODE=1
GPU_DevID=0
TrainIterNum=120000
CurExprID=""
DoDnnTrain=1

#
# path
VMMR_DATA_PATH=/home/ygao/Projects/VehicleRecogntition/Data
VMMR_EXPR_PATH=/home/ygao/Projects/VehicleRecogntition/DNN_Experiments/
CAFFE_TOOLS_PATH=/home/ygao/Projects/VehicleRecogntition/Code/caffe/build/tools
VMMR_PYTHON_PATH=/home/ygao/Projects/python/vmmr

# python scripts
PY_DNNTrain=DNNMakemodel_Train.py

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
                   echo "equalColorHist";;
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

IFS=$':'
#####################################################
## Train DNN one by one
####################################################
echo Now compute image mean ...
OLD_PWD=`pwd`

# all python script should be called from python/vmmr path
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
         	./$PY_DNNTrain $DataSetName $pi $NEW_WIDTH
	done
done
cd $OLD_PWD
echo Now current work dir is `pwd`.
echo "Complete image mean computing. :)"


#
# Restore system IFS
#
IFS=$IFS.OLD



