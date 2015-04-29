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

function Usage {
    echo "Usage :"
    echo "%1 DATASET_NAME PATCH_ID NewWidth TrainIterNum ResSolverStateIterNum TrainTestListID COMPUT_MODE GPU_DevID CurExprID DoDnnTrain"
    echo "    DATASET_NAME, str, the dataset name"
    echo "    PATCH_ID, str, patch id seperated by ':', e.g. 6：7:-1 "
    echo "    NewWidth, int, New width from standard Vehicle Face before cropping"
    echo "    ResSolverStateIterNum, int, the iter num from which to restore training. -1 denotes new training."
    echo "    TrainTestListID, int, the id of train test list set"
    echo "    TrainIterNum, int, the training iter num"
    echo "    COMPUT_MODE, int, 1 denote use GPU, 0 denotes only CPU"
    echo "    GPU_DevID, int, gpu device id"
    echo "    CurExprID，int,  the experiment id, mainly use for error handling"
    echo "    DoDnnTrain, int, 1 denotes training dnn, 0 denotes only carry out validation test"
    echo
    echo "More detailed information, you can look at the source code :)"
    echo
}

# internal parameters:
DATASET_NAME="V1_AAuMColor"
PATCHIDS="7:-1"
NEW_WIDTH=150

TrainIterNum=120000
ResSolverStateIterNum=-1 # <0 means new training
TrainTestListID=0  # default is 0
COMPUT_MODE=1
GPU_DevID=0
CurExprID=""  # for error handling only
DoDnnTrain=1

echo parameter number = $#
if [ $# -eq 10 ]
then
    echo Get parameter from command line.
    DATASET_NAME=$1
    PATCHIDS=$2
    NEW_WIDTH=$3

    TrainIterNum=$4
    ResSolverStateIterNum=$5  # <0 means new training
    TrainTestListID=$6  # default is 0
    COMPUT_MODE=$7
    GPU_DevID=$8
    CurExprID=$9  # for error handling only
    DoDnnTrain=${10}
else
    Usage
    echo "The number of input parameter is : " $#, Not correct.
    echo Required parameter number is 10
    read -p "The default parameters will be used. Are you sure (y/n)?" repl
    if [ "$repl" != "y" ] && [ "$repl" != "Y" ]
    then
        echo "You don't accept default parameters, so exit program!"
	exit -2
    fi
    echo
    echo Ok, you accept default parameters to run this program! Go ...
    echo
fi

#list parameters:
echo Parameter list:
echo "Data set name: $DATASET_NAME"
echo "Path ids     : $PATCHIDS"
echo "New Width    : $NEW_WIDTH"
echo "Train iter num : $TrainIterNum"
echo "Restore from iter num: $ResSolverStateIterNum"  # <0 means new training
echo "Train test list id   : $TrainTestListID"  # default is 0
echo "Comput mode  : $COMPUT_MODE"
echo "GPU device id: $GPU_DevID"
echo "Current Expr. id str : $CurExprID"  # for error handling only
echo "Is do DNN training   : $DoDnnTrain"
echo

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
echo Now start to training DNN mean ...
OLD_PWD=`pwd`

# all python script should be called from python/vmmr path
cd $VMMR_PYTHON_PATH
echo Now current work dir is `pwd` . "(should be vmmr python dir)"

for pi in $PATCHIDS
do
    echo "Now training patch `GetPatchCode $pi` of $DATASET_NAME"
    if [ -z $CurExprID ] || [ "$CurExprID" == "" ]
    then 
        ./$PY_DNNTrain $DATASET_NAME $pi $NEW_WIDTH -tin $TrainIterNum -rst $ResSolverStateIterNum -ttlid $TrainTestListID -cm $COMPUT_MODE -gid $GPU_DevID -doDnn 1
    else
        ./$PY_DNNTrain $DATASET_NAME $pi $NEW_WIDTH -tin $TrainIterNum -rst $ResSolverStateIterNum -ttlid $TrainTestListID -cm $COMPUT_MODE -gid $GPU_DevID -expid $CurExprID -doDnn $DoDnnTrain
    fi
done

cd $OLD_PWD
echo Now current work dir is `pwd`

echo "Complete DNN training  :)"


#
# Restore system IFS
#
IFS=$IFS.OLD



