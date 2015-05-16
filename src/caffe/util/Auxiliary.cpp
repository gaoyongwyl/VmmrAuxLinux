/*"""
  TopFolder
  |
  <Make1_Model1_Model2>
  |
  <Make1_Model3>
  |
  ... ...
  |
  
  For make and model of a file, truct it folder name not its name !!!
  Two level folder structure.
  The make and model is decided by subfolder !!! not the filenames. because we may
  adjust file's make or model only by moving it to other folder when we double check
  samples.   
  """
*/


#include <stdlib.h>
//#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>
//b#include <io.h>

#include "caffe/util/Auxiliary.h"


namespace VMMR {

ofstream ofGlobalLogger;

long factorial(int num)  
{  
  long result=1;  
  for(int i=1;i<=num;i++){  
    result*=i;  
  }  
  return result;  
}  

long pnm(int num, int len)  
{  
  return factorial(num)/len*factorial(num-len);  
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

#ifndef WIN32

  static void _split_whole_name(const char *whole_name, char *fname, char *ext)
{
  char *p_ext;

  char _szWholeName[2048];
  strcpy( _szWholeName, whole_name );
  
  p_ext = rindex( _szWholeName, '.');
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

  char _szPath[2048];
  strcpy( _szPath, path );
  p_whole_name = rindex(_szPath, '/');
  if (NULL != p_whole_name)
    {
      p_whole_name++;
      _split_whole_name(p_whole_name, fname, ext);

      snprintf(dir, strlen(path) - strlen(p_whole_name), "%s", path);
    }
  else
    {
      _split_whole_name(path, fname, ext);
      dir[0] = '\0';
    }
}

#endif

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

// return 0 if exist
// return -1 if non-exist!
int FileOrFolderExist( string strFileOrFolder )
{
int status = access(strFileOrFolder.c_str(),0 );

if( status == 0 ) {
return  0;   // exist
} else{  
return -1;   // not existed!
}	
}

int InitGlobalLogger( string strLoggerFileName )
{	
if( ofGlobalLogger.is_open() ) {
cout << "Global logger already initialized !" << endl;
return -1;
}
ofGlobalLogger.open( strLoggerFileName.c_str(), ios::out );	
if( !ofGlobalLogger.is_open() ) {
cout << "ERROR: Open global logger file %s !" << endl << "File name: " << strLoggerFileName << endl; 
return -2;
}
return 0;
}

int CloseGlobalLogger()
{
if( ofGlobalLogger.is_open() ) {
ofGlobalLogger.close();
} else {
cout << "Global logger already closed !" << endl;
}
return 0;
}

string strToLower(const string &str)
{
string strTmp = str;
transform(strTmp.begin(),strTmp.end(),strTmp.begin(),::tolower);
return strTmp;
}

unsigned int CountFileLineNum( string& strFilePathname )
{
unsigned int uLineNum = 0;
ifstream ifFileRead;
ifFileRead.open( strFilePathname.c_str(), ios::in );

if( !ifFileRead.is_open() ) {
cout << endl << "Failed to open file : " << strFilePathname << endl;
return -1;
}

while( !ifFileRead.eof() ) {
char szLine[_MAX_LINE];
ifFileRead.getline( szLine, _MAX_LINE );
if( strlen( szLine ) > 0 ) {
uLineNum += 1;
}
}
ifFileRead.close();

return uLineNum;
}

// return ture : the same
// return false : differenct
bool compareNoCase(const string &strA,const string &strB)
{
string str1 = strToLower(strA);
string str2 = strToLower(strB);
return (str1 == str2);
}

/*
  A file's correct make and model is decied by its folder name.
  
  list file conent:
  大众_0012_0027_0029/大众_0012_00015637.jpg|大众_0012
  大众_0012_0027_0029/大众_0012_00028722.jpg|大众_0012
  大众_0007/大众_0007_00014383.jpg|大众_0007
  大众_0007/大众_0007_00023254.jpg|大众_0007
  大众_0007/大众_0007_00011061.jpg|大众_0007
  大众_0007/大众_0007_00012797.jpg|大众_0007
  大众_0007/大众_0007_00024926.jpg|大众_0007
*/
int LoadGroundTruth( string& strGroundTruthFile, map<string, VMMGrounTruth>& mapVmmGroundTruth )
{
ifstream ifGroundTruth( strGroundTruthFile.c_str() );
int count = 0;
while( !ifGroundTruth.eof() && ifGroundTruth.is_open() ) {
char szLine[_MAX_LINE ];
ifGroundTruth.getline( szLine, _MAX_LINE );
string strLine = szLine;
TrimSpace( strLine );
if( strLine.size() < 4 ) {
continue;
}
string strRelpathFilename = strLine.substr(0, strLine.find_first_of( "|" ) );
string strMakemodel = strLine.substr( strLine.find_first_of( "|" ) + 1 );
VMMGrounTruth stVmmGroundTruth;
stVmmGroundTruth.strFileName = strRelpathFilename;
stVmmGroundTruth.strMake = strMakemodel.substr( 0, strMakemodel.find_first_of( "_" ) );
stVmmGroundTruth.strModel = strMakemodel.substr( strMakemodel.find_first_of( "_" ) + 1 ); // there may be multi-"_" in model name 
mapVmmGroundTruth[strRelpathFilename] = stVmmGroundTruth ;

count ++;
printf( "Load [%d] : %s %s_%s\t\r", count, strRelpathFilename.c_str(), stVmmGroundTruth.strMake.c_str(), stVmmGroundTruth.strModel.c_str() );
}
ifGroundTruth.close();

cout << endl << "Totally load " << count << " relpathname and truth class" << endl << endl;

return 0;
}

/* 
   MG_0000 1
   MG_0001 2
   MG_0002 3
   MG_0003 4
   MG_0004 5
   三一重工_0000 6
   三环十通汽车_0000 7
*/
int LoadClassLabelDict( string& strClassLabelDictFile, VRClassType enVRClassType, map<int, VMMName>& mapLabelMakeModel )
{
ifstream ifClassLabelDict( strClassLabelDictFile.c_str() );

int count = 0;
while( !ifClassLabelDict.eof() && ifClassLabelDict.is_open() ) {
char szLine[_MAX_LINE];
ifClassLabelDict.getline( szLine, _MAX_LINE );
string strLine = szLine;
TrimSpace( strLine );
if( strLine.size() < 2 ) continue;

string strMakeModel = strLine.substr( 0, strLine.find_first_of( " " ) );
string strClassLabel = strLine.substr( strLine.find_first_of( " " ) + 1 );
int iLabel = atoi( strClassLabel.c_str() );

VMMName stVmmName;
if( enVRClassType == MAKE_MODEL ) { // for maekmodel label dict, <make>_<model(may have "_">
stVmmName.strMake = strMakeModel.substr( 0, strMakeModel.find_first_of( "_" ) );
stVmmName.strModel = strMakeModel.substr( strMakeModel.find_first_of( "_" ) + 1 );  // there maybe multi "_" in model name
} else if( enVRClassType == MAKE ) {
TrimSpace( strMakeModel );
stVmmName.strMake = strMakeModel; //for make label dict, only make name.
stVmmName.strModel = "";
} else {
std::cout << "Error: Unknown class type ! " << endl;
}
mapLabelMakeModel[iLabel] = stVmmName;
count ++;

 if( enVRClassType == MAKE_MODEL ) {
   printf( "[%d] : %s_%s\t\t\r", count, stVmmName.strMake.c_str(), stVmmName.strModel.c_str() );
 } else if( enVRClassType == MAKE ) {
   printf( "[%d] : %s\t\t\r", count, stVmmName.strMake.c_str() );
 }
 }
 
 std::cout << endl << "Total " << count << " classes loaded." << endl << endl;
 
 ifClassLabelDict.close();
 return 0;
}

int ReadRelpathFileNameFromListFile( string& strListFile, list<string>& lstRelpathFilenames )
{
  ifstream listFile( strListFile.c_str() );
  
  if( !listFile.is_open() ) {
    cout << endl << "Failed to open file : " << strListFile << endl;
    return -1;
  }
  
  int count = 0;
  while( !listFile.eof() ) {
    char szLine[ _MAX_LINE ];
    listFile.getline( szLine, _MAX_LINE );
    string strLine = szLine;
    TrimSpace( strLine );
    if( strLine.size() < 4 ) continue;
    size_t nSpacePos =  strLine.find_first_of( " " );
    string strRelpathFilename;
    if ( nSpacePos != string::npos ) {
      strRelpathFilename = strLine.substr( 0, nSpacePos);
      TrimSpace( strRelpathFilename );
    }else {
      strRelpathFilename =  strLine;
    }
    lstRelpathFilenames.push_back( strRelpathFilename );
    printf( "Read [%d]: %s \t\r", count, strRelpathFilename.c_str() );
    
    count ++;
  }
  listFile.close();
  
  std::cout << endl << "Totally read " << count << " relative path and file names" << endl << endl;
  
  return 0;
}

void TrimSpace( string& str )
{
  str.erase( 0, str.find_first_not_of("\r\t\n ")); 
  str.erase( str.find_last_not_of("\r\t\n ") + 1);
}

int LoadRandSelPairListFromFile( string strFileName, int nTotalNum, unsigned int nRandSelected, vector<ImagePair>& vecPairs )
{
  if( vecPairs.size() < nRandSelected ) {
    cout << endl << "Please allocate enough size for vecPairs before calling this function !" << endl;
  }
  vector<unsigned int> vecIndex( nTotalNum );
  vector<unsigned int> vecSelectedIndex( nRandSelected );
  
  for( int n = 0; n < nTotalNum; n ++ ) {
    vecIndex[n] = n;
  }
  cout << "First 50 : "  << endl;
  for( int n = 0; n  < 50; n ++ ) {
    cout << vecIndex[n] << " " ;
  }
  cout << endl;
  
  cout << "Random shuffle ... \n";
  random_shuffle( vecIndex.begin(), vecIndex.end() ); 
  cout << "Random shuffle over. \n";
  
  for( int m = 0; m < nRandSelected; m ++ ){
    vecSelectedIndex[m] = vecIndex[m];
  }
  
  //sort :
  sort( vecSelectedIndex.begin(), vecSelectedIndex.end() );
  
  cout << "First 50 : \n";
  for( int n = 0; n  < 50; n ++ )	{
    cout << vecSelectedIndex[n] << " ";
  }
  cout << endl;
  
  FILE *hExtraModelListFile = fopen( strFileName.c_str(), "r" );
  unsigned int nCount = 0; 
  unsigned int nSelected = 0;
  
  if( hExtraModelListFile  == NULL ){ 
    printf( "ERROR: open pair list file %s \n", strFileName.c_str() );
    return -1;
  }
  char szLine[_MAX_LINE]; 
  
  printf( "\nStart load pair list randomly from %s ... \n", strFileName.c_str() );
  if( ofGlobalLogger.is_open() ){
    ofGlobalLogger << "Start load pair list randomly from  : " << strFileName;
  }
  
  while( fgets( szLine, _MAX_LINE, hExtraModelListFile ) != NULL ) {
    if( nCount == vecSelectedIndex[nSelected] ) {
      string csLineText = szLine;
      int nColonPos = csLineText.find_last_of( ':' );
      string strRelpathFileNameA = csLineText.substr( 0, nColonPos );
      string strRelpathFileNameB = csLineText.substr( nColonPos + 1, csLineText.size() - nColonPos - 1 );
      TrimSpace( strRelpathFileNameA );
      TrimSpace( strRelpathFileNameB );			
      
      ImagePair imgPair;
      imgPair.csFileNameA = strRelpathFileNameA;
      imgPair.csFileNameB = strRelpathFileNameB;			
      vecPairs[nSelected] = imgPair;
      nSelected ++;
    }
    if( nSelected >= nRandSelected ) { 	
      break;
    }
    nCount ++;
    
    if( nCount % 10000 == 0 ) { printf ( "Load %d \r", nCount ); }
  }
  fclose( hExtraModelListFile ); 
  hExtraModelListFile = NULL;
  
  printf( "\nLoad %d pair list randomly .\n", nSelected );
  if( ofGlobalLogger.is_open() ) {
    ofGlobalLogger << "\nLoad " << nSelected << "extra pair list .\n";
  }
  
  if( nSelected != nRandSelected ){
    cout << endl << "Error: Hope randomly selected " << nRandSelected << " pairs, but only got " << nSelected;
    return -1;
  }
  
  cout << endl << "Over - Randomly air list load.\n";
  if( nSelected != nRandSelected ){
    cout << "Over - Randomly pair list load.\n";
  }
  
  return 0;
}

int LoadPairListFromFile( string strFileName, list<ImagePair>& lstPairs )
{
  FILE *hExtraModelListFile = fopen( strFileName.c_str(), "r" );
  unsigned int nCount = 0; 
  
  if( hExtraModelListFile  == NULL ) { 
    printf( "ERROR: Load extra file %s \n", strFileName.c_str() );
    return -1;
  }
  char szLine[_MAX_LINE]; 
  
  printf( "\nStart load extra pair list from %s ... \n", strFileName.c_str() );
  if( ofGlobalLogger.is_open() ) {
    ofGlobalLogger << "\nStart load extra pair list from : \n" << strFileName;
  }
  
  while( fgets( szLine, _MAX_LINE, hExtraModelListFile ) != NULL )
    {
      string csLineText = szLine;
      int nColonPos = csLineText.find_last_of( ':' );
      string strRelpathFileNameA = csLineText.substr( 0, nColonPos );
      string strRelpathFileNameB = csLineText.substr( nColonPos + 1, csLineText.size() - nColonPos - 1 );
      TrimSpace( strRelpathFileNameA );
      TrimSpace( strRelpathFileNameB );
      
      ImagePair imgPair;
      imgPair.csFileNameA = strRelpathFileNameA;
      imgPair.csFileNameB = strRelpathFileNameB;
      lstPairs.push_back( imgPair );
      nCount ++;
      
      if( nCount % 10000 == 0 ){
	printf ( "Load %d \r", nCount );
      }
    }
  fclose( hExtraModelListFile ); 
  hExtraModelListFile = NULL;
  
  printf( "\nLoad %d pair list .\n", lstPairs.size() );
  cout << endl << "Over - pair list load.\n\n";
  if( ofGlobalLogger.is_open() ){
    ofGlobalLogger << "\nLoad " << lstPairs.size() << " pair list.\n" ;
    ofGlobalLogger << "Over - pair list load.\n\n";
  }
  
  return 0;
}

// create ModelSet vector
// Add files to into make_model 
int ReadAllModelSetFromList( string strListFileName, 
			     VRClassType enVRClassType,   //make or make_model
			     vector<ModelSet>& vecAllModelSet, 
			     vector<string>& vecFileNames )
{
  FILE *hvectorFile = fopen( strListFileName.c_str(), "r" );
  char szLine[_MAX_LINE]; 
  int nNumModel = 0;
  int nNumOtherAdded = 0;
  
  if( hvectorFile != NULL )
    {
      ModelSet *pModelSet = NULL;		
      cout << endl << endl << "Read all make model info and create file name set : " << endl;
      if( ofGlobalLogger.is_open() ) {
	cout << endl << endl;
	cout << "Read all make model info and create file name set : " << endl;
      }
      
      while( fgets( szLine, _MAX_LINE, hvectorFile ) != NULL ) {
	string strLineText = szLine;
	string strRelpath, strFilename;
	int iLabel = -1; // label always >= 0
	ParseListFileLine( strLineText, strRelpath, strFilename, iLabel );
	
	vecFileNames.push_back( strRelpath + strFilename );
	string strMakemodelName = strRelpath.substr( 0, strRelpath.size() - 1 ); //remove slash
	size_t nSepPos = strMakemodelName.find_first_of( "_" );
	if( nSepPos == string::npos ) {
	  cout << endl << "Error: invalid relpath : " << strMakemodelName << endl;
	}
	string strMakeName = strMakemodelName.substr( 0, nSepPos );
	
	//check whether already add ?
	bool bAdded = false;
	for( int n = 0; n < vecAllModelSet.size(); n ++ )
	  {
	    string lowerCaseModelNew = "";
	    string lowerCaseModelExist =  vecAllModelSet[n].csModelName;
	    
	    if( enVRClassType == MAKE ) { 
	      lowerCaseModelNew = strMakeName;
	    } else if ( enVRClassType == MAKE_MODEL ) {
	      lowerCaseModelNew = strMakemodelName;
	    } else {
	      cout << "Unknown vehicel recognition type ! " << endl;
	    }
	    
	    if( lowerCaseModelExist.compare( lowerCaseModelNew ) == 0 )
	      {
		bAdded = true;
		vecAllModelSet[n].csFileNames.push_back(  strRelpath + strFilename );
		nNumOtherAdded ++;
		break;
	      }
	  }
	
	if( bAdded == false )
	  {
	    ModelSet newModelSet;
	    if( enVRClassType == MAKE ) { 
	      newModelSet.csModelName = strMakeName;
	    } else if ( enVRClassType == MAKE_MODEL ) {
	      newModelSet.csModelName = strMakemodelName;
	    } else {
	      cout << "Unknown vehicel recognition type ! " << endl;
	    }
	    newModelSet.csFileNames.push_back( strRelpath + strFilename  );
	    vecAllModelSet.push_back( newModelSet );
	    
	    nNumModel++;
	    
	    printf( "Add make model [%d]: %s \r", nNumModel, strMakemodelName.c_str()  );
	  }
      }
      
      fclose( hvectorFile ); hvectorFile = NULL;
    } else {
    printf( "ERROR: Read %s ! \n", strListFileName.c_str() );
  }
  
  cout << "Total model number is: " << nNumModel << "  " << endl << endl;
  if( ofGlobalLogger.is_open() ) {		
    ofGlobalLogger << "Total model number is: " << nNumModel << endl << endl;
  }
  
  return 0;
}

int ReadAllModelSetFromList( string strListFile, 
			     VRClassType enVRClassType,   //make or make_model
			     list<ModelSet>& lstAllModelSet, 
			     list<string>& lstRelPathFileNames )
{
  FILE *hvectorFile = fopen( strListFile.c_str(), "r" );
  char szLine[_MAX_LINE]; 
  int nNumModel = 0;
  int nNumOtherAdded = 0;
  
  if( hvectorFile != NULL ) {
    ModelSet *pModelSet = NULL;
    
    cout << endl <<  "Read all make model info and create file name set : \n" << endl;
    if( ofGlobalLogger.is_open() ) {			
      ofGlobalLogger << endl << "Read all make model info and create file name set : " << endl;
    }
    
    while( fgets( szLine, _MAX_LINE, hvectorFile ) != NULL ) {
      string strLineText = szLine;
      TrimSpace( strLineText );
      
      string strRelpath, strFilename;
      int iLabel = -1; // label always >= 0
      ParseListFileLine( strLineText, strRelpath, strFilename, iLabel );
      
      //push relpath filename into list:
      string strRelpathFilename = strRelpath + strFilename;
      lstRelPathFileNames.push_back( strRelpathFilename );
      
      string strMakemodelName = strRelpath.substr( 0, strRelpath.size() - 1 ); //remove slash			
      size_t nSepPos = strMakemodelName.find_first_of( "_" );
      if( nSepPos == string::npos ) {
	cout << endl << "Error: invalid relpath : " << strMakemodelName << endl;
      }
      string strMakeName = strMakemodelName.substr( 0, nSepPos );
      
      //check whether already add ?
      bool bAdded = false;
      list<ModelSet>::iterator modelSetItr = lstAllModelSet.begin();
      for( int n = 0; n < lstAllModelSet.size(); n ++, modelSetItr ++ )
	{
	  string lowerCaseModelNew = "";
	  string lowerCaseModelExist = modelSetItr->csModelName;
	  if( enVRClassType == MAKE ) { 
	    lowerCaseModelNew = strMakeName;
	  } else if ( enVRClassType == MAKE_MODEL ) {
	    lowerCaseModelNew = strMakemodelName;
	  } else {
	    cout << "Unknown vehicel recognition type ! " << endl;
	  }
	  transform( lowerCaseModelExist.begin(), lowerCaseModelExist.end(), lowerCaseModelExist.begin(), ::tolower );
	  transform( lowerCaseModelNew.begin(), lowerCaseModelNew.end(), lowerCaseModelNew.begin(), ::tolower );
	  
	  if( lowerCaseModelExist.compare( lowerCaseModelNew ) == 0 )
	    {
	      bAdded = true;
	      modelSetItr->csFileNames.push_back( strRelpathFilename );  //push relpath filename
	      nNumOtherAdded ++;
	      break;
	    }
	}
      
      if( bAdded == false ) {
	ModelSet newModelSet;
	if( enVRClassType == MAKE ) { 
	  newModelSet.csModelName = strMakeName;
	} else if ( enVRClassType == MAKE_MODEL ) {
	  newModelSet.csModelName = strMakemodelName;
	} else {
	  cout << "Unknown vehicel recognition type ! " << endl;
	}
	newModelSet.csFileNames.push_back( strRelpathFilename );
	lstAllModelSet.push_back( newModelSet );
	nNumModel++;
	printf( "Add model [%d]: %s \r", nNumModel, strMakemodelName.c_str()  );
      }
    }
    
    fclose( hvectorFile ); hvectorFile = NULL;
  } else {
    printf( "ERROR: Read %s ! \n", strListFile.c_str() );
  }
  
  cout << "Total model number is: " << nNumModel << endl;
  cout << "Total repath file name : " << nNumModel + nNumOtherAdded << endl << endl;
  if( ofGlobalLogger.is_open() ) {		
    ofGlobalLogger << "Total model number is: " << nNumModel <<" and total repath file name : " << nNumModel + nNumOtherAdded << endl << endl;
  }
  
  return nNumModel + nNumOtherAdded ;
}

//need save to file for check!!!
int CreateIntraExtraPair( vector<ModelSet>& vecAllModelSet, 
			  list<ImagePair>& lstIntraPairs, 
			  list<ImagePair>& lstExtraPairs,
			  bool bPushToList,
			  string csIntravectorFile, 
			  string csExtravectorFile )
{
  int nIntra = 0, nExtra = 0;
  FILE *hIntravector = NULL, *hExtravector = NULL;
  
  if( csIntravectorFile != "" ) {
    hIntravector = fopen( csIntravectorFile.c_str(), "w" );
    if( hIntravector == NULL  ) {
      printf( "ERROR: open intra or extra vector file for write !\n" );
      printf( "%s\n", csIntravectorFile.c_str() );
      return -1;
    }
  }
  if( csExtravectorFile != "" ) {
    hExtravector = fopen( csExtravectorFile.c_str(), "w" );
    if( hExtravector == NULL ) {
      printf( "ERROR: open intra or extra vector file for write !\n" );
      printf( "%s\n", csExtravectorFile.c_str() );
      return -1;
    }
  }	
  
  //create intra pair
  for( int k = 0; k < vecAllModelSet.size(); k ++ ) {
    ModelSet& modelSet = vecAllModelSet[k];
    int nNumFileNames = modelSet.csFileNames.size();
    int nShouldbe = ( nNumFileNames * ( nNumFileNames - 1 ) ) / 2;
    
    int nModelIntra = 0;
    for( int m = 0; m < modelSet.csFileNames.size(); m ++ )
      {
	for( int n = m+1; n < modelSet.csFileNames.size(); n ++ )
	  {
	    ImagePair imPair;
	    imPair.csFileNameA = modelSet.csFileNames[m];
	    imPair.csFileNameB = modelSet.csFileNames[n];
	    
	    if( bPushToList == true ) {						
	      lstIntraPairs.push_back( imPair );
	    }
	    nModelIntra ++;
	    
	    if( hIntravector != NULL )	{
	      fprintf( hIntravector, "%s : %s\n", imPair.csFileNameA.c_str(), imPair.csFileNameB.c_str() );	
	    }
	    printf( "Intra pair [%d]: %s vs %s \r", nIntra, imPair.csFileNameA.c_str(), imPair.csFileNameB.c_str() );
	  }
      }
    
    if( nModelIntra !=  nShouldbe )
      {
	printf( "ERROR: The number combination is not correct for %s \n", modelSet.csModelName.c_str() );
	return -1;
      }
    nIntra += nModelIntra;
  }
  if( hIntravector != NULL )
    {
      fprintf( hIntravector, "Total intra-class pair is %d \n", nIntra );
      fclose( hIntravector ); hIntravector = NULL;
    }
  
  cout << "Total intra-class pair is " << nIntra << endl;
  
  //create extra pair
  for( int k = 0; k < vecAllModelSet.size(); k ++ )
    {
      ModelSet& modelSetA = vecAllModelSet[k];
      for( int l = k+1; l < vecAllModelSet.size(); l ++ )
	{
	  ModelSet& modelSetB = vecAllModelSet[l];
	  
	  for( int m = 0; m < modelSetA.csFileNames.size(); m ++ )
	    {
	      for( int n = 0; n < modelSetB.csFileNames.size(); n ++ )
		{
		  ImagePair imPair;
		  imPair.csFileNameA = modelSetA.csFileNames[m];
		  imPair.csFileNameB = modelSetB.csFileNames[n];
		  if( bPushToList == true ) {					
		    lstExtraPairs.push_back( imPair );
		  }
		  
		  nExtra ++;
		  
		  if( hExtravector != NULL ) {
		    fprintf( hExtravector, "%s : %s\n", imPair.csFileNameA.c_str(), imPair.csFileNameB.c_str() );	
		  }
		  printf( "Extra pair: [%d] %s vs %s   \r", nExtra, imPair.csFileNameA.c_str(), imPair.csFileNameB.c_str() );
		}
	    }
	  
	}
    }	
  
  if( hExtravector != NULL )
    {
      fprintf( hExtravector, "Total extra-class pair is %d \n", nExtra );
      fclose( hExtravector ); hExtravector = NULL;
    }
  
  cout << endl << "Total extra-class pair is " << nExtra << endl;
  
  return 0;
}

//special purpose funcion
int LoadFileNameMap( string mapFile, map<string, string>& mapFileNames )
{
  FILE *hMapFile = fopen( mapFile.c_str(), "r" );
  
  if( hMapFile == NULL ) {
    printf( "ERROR: open map file %s !\n", mapFile.c_str() );
    return -1;
  }
  
  char szLine[_MAX_LINE];
  while( fgets( szLine, _MAX_LINE, hMapFile ) != NULL ){
    string strLine = szLine;
    TrimSpace( strLine );
    
    string strName1 = strLine.substr( strLine.find_last_of( "\\" )+1, strLine.size() -  strLine.find_last_of( "\\" ) -1);
    
    string strName2;
    if( fgets( szLine, _MAX_LINE, hMapFile ) != NULL ){
      strLine = szLine;
      TrimSpace( strLine );
      
      strName2 = strLine.substr( strLine.find_last_of( "\\" )+1, strLine.size() -  strLine.find_last_of( "\\" )-1 );
    } else {
      printf( "Error: read name 1 !\n" );
      return -1;
    }
    
    if( fgets( szLine, _MAX_LINE, hMapFile ) != NULL ) {
      strLine = szLine;
      TrimSpace( strLine );
      
      if( strLine != "0" ){
	printf( "Error: no zero flag found !\n" );
	return -1;
      } else {
	mapFileNames[strName1] = strName2;
      }
    } else {
      printf( "Error: read name 2 !\n" );
      return -1;
    }
  }
  
  fclose( hMapFile ); hMapFile = NULL;
  
  printf( "\nLoad file name map items : %d \n", mapFileNames.size() );
  
  return 0;
}

int LoadKeypointsFromFileLab( string strAnnotFile,  
			      vector<CvPoint2D32f>& vecKeyPoints )
{
  ifstream ifAnnotFile;
  ifAnnotFile.open( strAnnotFile.c_str(), ios::in );
  if( !ifAnnotFile.is_open() ) {
    cout << "Error: open file " << strAnnotFile << endl;
  }
  int nKeyPoints = -1;
  bool bStart = false, bEnd = false;
  int count = 0;

  while( ! ifAnnotFile.eof() ) {
    string strLine;
    getline( ifAnnotFile, strLine );
    
    if( strLine.find( "n_points" ) != string::npos ) {
      string strNum = strLine.substr( strLine.find_last_of( ":" ) + 1 );
      TrimSpace( strNum );
      nKeyPoints = atoi( strNum.c_str() );
      if( nKeyPoints < 0 ) {
	cout << "Key points number parse error in file: " << strAnnotFile << endl;
	return -1;
      }
      if( vecKeyPoints.size() < nKeyPoints ) {
	vecKeyPoints.resize( nKeyPoints );
      }
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
	  vecKeyPoints[count] = ptKeyPoint;
          count ++;	  
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

int LoadKeypointsFromFile( string strAnnotFile,  
			   vector<VFKeyPoint>& vecKeyPoints )
{
  FILE *hAnnotFile = fopen( strAnnotFile.c_str(), "r" );
  
  int index = 0;
  if( hAnnotFile != NULL ) {
    char szLine[_MAX_LINE];
    char szFileName[_MAX_LINE];
    
    if( fgets( szLine, _MAX_LINE, hAnnotFile ) != NULL ) {
      CvPoint2D32f ptKeyPt[_NUM_KEY_POINT_];
      
      sscanf( szLine, "%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", szFileName,
	      &(ptKeyPt[0].x), &(ptKeyPt[0].y), &(ptKeyPt[1].x), &(ptKeyPt[1].y), &(ptKeyPt[2].x), &(ptKeyPt[2].y), &(ptKeyPt[3].x), &(ptKeyPt[3].y), 
	      &(ptKeyPt[4].x), &(ptKeyPt[4].y), &(ptKeyPt[5].x), &(ptKeyPt[5].y), &(ptKeyPt[6].x), &(ptKeyPt[6].y), &(ptKeyPt[7].x), &(ptKeyPt[7].y), 
	      &(ptKeyPt[8].x), &(ptKeyPt[8].y), &(ptKeyPt[9].x), &(ptKeyPt[9].y), &(ptKeyPt[10].x), &(ptKeyPt[10].y) );
      
      for( int n = 0; n < _NUM_KEY_POINT_; n ++ )
	{
	  VFKeyPoint vfKeyPt;
	  vfKeyPt.index = n;
	  vfKeyPt.keyPoint = ptKeyPt[n];
	  vecKeyPoints.push_back( vfKeyPt );
	}
    }		
  } else {
    printf( "Error: load %s \n", strAnnotFile.c_str() );
    return -1;
  }
  
  
  fclose( hAnnotFile ); hAnnotFile = NULL;
  
  return 0;
}

//
//fFalsePosRate: Keep all intra correctly classified, then computer how many extra pair are classified as intra pair.
//
// In this function ,draw both hists in same figures and save them .... 
float CalcIntraExtraDistHistDist( vector<float>& vecIntraDists, 
				  vector<float>& vecExtraDists, 
				  float& fThreshold, 
				  float& fFalsePosRate, 
				  int method,
				  string strIntraExtraHistImageFile )
{
  float fHistDist = 0;
  Mat mIntra(vecIntraDists);
  Mat mExtra(vecExtraDists);
  
  //calc fFalsePosRate
  printf( "\n" );
  printf( "Sort vecIntraDists and vecExtraDists ... \n" );
  sort( vecIntraDists.begin(), vecIntraDists.end() );
  sort( vecExtraDists.begin(), vecExtraDists.end () );
  printf( "In vecIntraDists, Max = %f ; Min = %f \n", vecIntraDists[0], vecIntraDists[vecIntraDists.size()-1] );
  printf( "In vecExtraDists, Max = %f ; Min = %f \n", vecExtraDists[0], vecExtraDists[vecExtraDists.size()-1] );
  printf( "\n" );
  
  float fMaxDist = vecIntraDists[vecIntraDists.size()-1] > vecExtraDists[vecExtraDists.size()-1] ?
    vecIntraDists[vecIntraDists.size()-1] : vecExtraDists[vecExtraDists.size()-1];
  
  if( ofGlobalLogger.is_open() )	{
    ofGlobalLogger <<  "Sort vecIntraDists and vecExtraDists ... \n";
    ofGlobalLogger << "In vecIntraDists, Max = " << vecIntraDists[0] << "; Min = " << vecIntraDists[vecIntraDists.size()-1];
    ofGlobalLogger << "In vecExtraDists, Max = " << vecExtraDists[0] << "; Min = " << vecExtraDists[vecExtraDists.size()-1] << endl;		
  }
  
  unsigned int index98Percent = (unsigned int)(vecIntraDists.size() * 0.98);
  
  fThreshold = vecIntraDists[index98Percent-1];
  printf( "98 percent intra threshold is %f \n\n", fThreshold );
  if( ofGlobalLogger.is_open() ) {
    cout << "98 perc. intra threshold is " << fThreshold << endl << endl;
  }
  
  int nFalsePos = 0;
  //then count how many extra dist less or equal than the fThreshold hold
  for( int n = 0; n < vecExtraDists.size(); n ++ )
    {
      if( vecExtraDists[n] <= fThreshold ) {
	nFalsePos ++;
      } else {
	break;
      }
    }
  fFalsePosRate = ( nFalsePos * 1.0 ) / (float) vecExtraDists.size();
  printf( "When threshold is %f (98 perc. correct), false positive rate is %f  [%d | %d ]\n",
	  fThreshold, fFalsePosRate, nFalsePos,  vecExtraDists.size() );
  
  if( ofGlobalLogger.is_open() ) {
    ofGlobalLogger << "When threshold is " <<  fThreshold << "(98 perc. correct), false positive rate is " 
		   << fFalsePosRate << " " << nFalsePos << " " <<  vecExtraDists.size();
  }
  
  const int histSize = 100; // number of bin, the element number should be same as the hist dims
  int channels[] = {0}; // the element number should be same as the hist dims
  
  float hist_ranges[] = { 0, fMaxDist };
  const float* arr_intra_ranges[] = {hist_ranges }; // the element number should be same as the hist dims
  
  
  Mat intra_hist, extra_hist;
  
  calcHist( &mIntra, 1, channels, Mat(),
	    intra_hist,  1, &histSize, arr_intra_ranges, 
	    true, false );
  
  float extra_ranges[] = { 0, fMaxDist };
  const float* arr_extra_ranges[] = {hist_ranges };
  
  calcHist( &mExtra, 1, channels, Mat(),
	    extra_hist,  1, &histSize, arr_extra_ranges, 
	    true, false );
  
  fHistDist = compareHist( intra_hist, extra_hist, method ); //should pad zero to make them same length
  printf( "The distance between Intra dist hist and Extra dist hist is %f\n", fHistDist );
  printf( "The distance type is %d\n", method );
  
  if( ofGlobalLogger.is_open() ) {
    ofGlobalLogger << "The distance between Intra dist hist and Extra dist hist is " << fHistDist << endl;
    ofGlobalLogger << "The distance type is " << method << endl;
  }
  
  if( strIntraExtraHistImageFile != "" )
    {
      IplImage *pIntraExtraHistImage = cvCreateImage(cvSize( 900, 680),8,3); 
      cvSet( pIntraExtraHistImage, cvScalar( 200, 200, 200 ) );
      
      if( pIntraExtraHistImage != NULL )
	{
	  float intra_hist_min = 9999, intra_hist_max = 0;
	  float extra_hist_min = 9999, extra_hist_max = 0;
	  float hist_min_value = 9999, hist_max_value = 0;
	  for( int n = 0; n < histSize; n ++ )
	    {
	      float fIntraBinVal = intra_hist.at<float>(n);
	      float fExtraBinVal = extra_hist.at<float>(n);
	      
	      if( fIntraBinVal < 0 || fExtraBinVal < 0 ) {
		printf( "ERROR: distance histogram bin val less than zero !\n" );
		return -2;
	      }
	      
	      if( intra_hist_min > fIntraBinVal ) { intra_hist_min = fIntraBinVal; }
	      if( intra_hist_max < fIntraBinVal ) { intra_hist_max = fIntraBinVal; }
	      if( extra_hist_min > fExtraBinVal ) { extra_hist_min = fExtraBinVal; }
	      if( extra_hist_max < fExtraBinVal ) { extra_hist_max = fExtraBinVal; }
	    }
	  hist_min_value = intra_hist_min < extra_hist_min ? intra_hist_min : extra_hist_min;
	  hist_max_value = intra_hist_max > extra_hist_max ? intra_hist_max : extra_hist_max;
	  
	  double bin_img_width=(double)pIntraExtraHistImage->width/(histSize*1.0);    
	  double bin_img_H_unit=(double)pIntraExtraHistImage->height/hist_max_value;    
	  
	  for( int n = 0; n < histSize; n ++ )
	    {
	      float fIntraBinVal = intra_hist.at<float>(n);
	      float fExtraBinVal = extra_hist.at<float>(n);
	      
	      CvPoint p0=cvPoint( n*bin_img_width, pIntraExtraHistImage->height);    
	      CvPoint p1=cvPoint((n+1)*bin_img_width, 
				 pIntraExtraHistImage->height - fIntraBinVal*bin_img_H_unit);   
	      cvRectangle(pIntraExtraHistImage, p0, p1,cvScalar(0,0,255), -1, 8, 0);  //用红色显示直方图  
	      
	      CvPoint p2=cvPoint((n+1)*bin_img_width, 
				 pIntraExtraHistImage->height - fExtraBinVal*bin_img_H_unit);   
	      cvRectangle(pIntraExtraHistImage, p0, p2, cvScalar(255,0,0), 1, 8, 0);    //用绿色显示直方图  
	    }
	  //show histogram distance:
	  CvFont font; 
	  char szHistDist[_MAX_LINE];
	  sprintf( szHistDist, "DOH(t:%d) = %f", method, fHistDist );
	  cvInitFont( &font, CV_FONT_VECTOR0, 1, 1, 0, 2, 8);
	  cvPutText( pIntraExtraHistImage, szHistDist, cvPoint(20, 40), &font, CV_RGB( 20, 0, 20) );
	  sprintf( szHistDist, "FalsePosRate(98Perc.) = %f", fFalsePosRate );
	  cvPutText( pIntraExtraHistImage, szHistDist, cvPoint(20, 100), &font, CV_RGB( 200, 0, 20) );
	  
	  cvSaveImage( strIntraExtraHistImageFile.c_str(), pIntraExtraHistImage );
	  
	  //cvShowImage( "Intra(Red) Extra(Green) Distance Hist", pIntraExtraHistImage );
	  //cvWaitKey(0);
	  
	  cvReleaseImage( &pIntraExtraHistImage ); pIntraExtraHistImage = NULL;		
	}
      else
	{
	  printf( "ERROR: Create intra extra distance hist image .\n" );
	}
      
    }
  
  return fHistDist;
}

int SaveIntraExtraDists( vector<float>& vecIntraDists, 
			 vector<float>& vecExtraDists, 
			 string csIntraDistFile, 
			 string csExtraDistFile )
{
  FILE *hIntraFile = fopen( csIntraDistFile.c_str(), "w" );
  FILE *hExtraFile = fopen( csExtraDistFile.c_str(), "w" );
  
  if( (hIntraFile == NULL ) || ( hExtraFile == NULL ) ){
    cout << "ERROR: Create intra or extra file !\n";
    return -1;
  }
  else
    {
      cout << "Sort vecIntraDists and vecExtraDists ... \n";
      sort( vecIntraDists.begin(), vecIntraDists.end() );
      sort( vecExtraDists.begin(), vecExtraDists.end () );
      
      for( int n = 0; n < vecIntraDists.size(); n++ ) {
	fprintf( hIntraFile, "%f\n", vecIntraDists[n] );
      }
      
      for( int n = 0; n < vecExtraDists.size(); n++ ) {
	fprintf( hExtraFile, "%f\n", vecExtraDists[n] );
      }
    }
  
  fclose( hIntraFile ); hIntraFile = NULL;
  fclose( hExtraFile ); hExtraFile = NULL;
  
  return 0;
}

int SaveFloatVecFeature( string strFileName, vector<float>& floatVecFeature )
{
  FILE *hHOGFeatFile = fopen( strFileName.c_str(), "w" );
  
  if( hHOGFeatFile != NULL )
    {
      fprintf( hHOGFeatFile, "%d\n", floatVecFeature.size() );
      for( int n = 0; n < floatVecFeature.size(); n ++ )
	{
	  fprintf( hHOGFeatFile, "%f\n", floatVecFeature[n] );
	}
    }
  else
    {
      printf( "ERROR: save feature file %s !\n", strFileName.c_str() );
      return -1;
    }
  
  fclose( hHOGFeatFile ); hHOGFeatFile = NULL;
  
  return 0;
}

int LoadFloatVecFeature( string csFileName, 
			 vector<float>& floatVecFeature )
{
  FILE *hHOGFeatFile = fopen( csFileName.c_str(), "r" );
  
  if( hHOGFeatFile != NULL )
    {
      int nFeatDims = 0;
      int count = 0;
      
      if( fscanf( hHOGFeatFile, "%d", &nFeatDims ) != 1 )
	{
	  printf( "ERROR: Read feature dims !	\n; " );
	  return -1;
	}
      
      if( nFeatDims > 0 )
	{
	  float tmp = 0;
	  
	  while( fscanf( hHOGFeatFile, "%f", &tmp ) != 0 )
	    {
	      floatVecFeature[count] = tmp;
	      count ++;
	      
	      if( count >= nFeatDims )
		{
		  break;
		}
	    }
	}
      else
	{
	  printf( "ERROR: HOG Feat dims is %d in feature file %s. \n", csFileName.c_str() );
	}
      fclose( hHOGFeatFile ); hHOGFeatFile = NULL;
      
      if( count != nFeatDims )
	{
	  printf( "ERROR: the number of read value is not equal to the HOG dims \n" );
	  printf( "Clear feature vector ... \n" );
	  return -2;
	}
    }
  else
    {
      printf( "ERROR: load feature file %s !\n", csFileName.c_str() );
      return -1;
    }
  
  return 0;
}


int LoadFloatVecFeaturesFromFiles( vector<string> vecFileNames, 
				   string strFloatVecFeatSaveFolder, 
				   int nFeatDims,
				   map<string, vector<float> >& mapAllNamedFeats )
{
  printf( "\n\nStart load HOG features ... \n" );
  
  int numLoaded = 0;
  for( int n = 0; n < vecFileNames.size(); n ++ )
    {
      printf( "Load [%6d]%s \r", n, vecFileNames[n].c_str() );
      
      vector<float> hogFeature(nFeatDims, 0 );
      string strHogFileName = strFloatVecFeatSaveFolder + vecFileNames[n].c_str();
      strHogFileName = strHogFileName.replace( strHogFileName.find_last_of( '.' ), 4, ".hog" );
      
      int ret = LoadFloatVecFeature( strHogFileName, hogFeature );
      
      if( ret < 0 )
	{
	  printf( "ERROR: load hog feature %s \n", strHogFileName.c_str() );	
	  continue;
	}
      
      mapAllNamedFeats[ vecFileNames[n]] = hogFeature;
      numLoaded ++;
    }
  
  printf( "Total loaded %d vehicle face hog representations. \n", numLoaded );
  printf( "\nComplete HOG features loading \n\n" );
  
  return 0;
}

int CalcIntraExtraDistances( map<string, vector<float> >& mapAllNamedFeats,
			     vector<ImagePair>& vecIntraPairs, 
			     vector<ImagePair>& vecExtraPairs, 
			     vector<float>& vecIntraDists, 
			     vector<float>& vecExtraDists, 
			     CompareFeatDistFunc ptrCompareFunc,
			     VMMRFeatDistType method,
			     unsigned int& nValidIntraDist,
			     unsigned int& nValidExtraDist )
{
  unsigned int nIntraDistCount = 0;
  unsigned int nExtraDistCount = 0;
  
  map<string, vector<float> >::iterator iterA, iterB;
  
  vector<ImagePair>::iterator iterImagePairvector;
  //compute intra pair distance:
  for( iterImagePairvector = vecIntraPairs.begin(); iterImagePairvector != vecIntraPairs.end(); iterImagePairvector ++ )
    {
      iterA = mapAllNamedFeats.find( iterImagePairvector->csFileNameA );
      iterB = mapAllNamedFeats.find( iterImagePairvector->csFileNameB );
      
      if( iterA != mapAllNamedFeats.end() && iterB != mapAllNamedFeats.end() )
	{			
	  float fDistAB = -9999;
	  fDistAB = (*ptrCompareFunc)( iterA->second, iterB->second, method );			
	  vecIntraDists[nIntraDistCount] = fDistAB;
	  nIntraDistCount ++;
	  printf( "%d th intra distance = %f\r", nIntraDistCount, fDistAB );
	}
      else
	{
	  if( iterA == mapAllNamedFeats.end() ) printf( "Not found %s feature \n",  
							iterImagePairvector->csFileNameA.c_str() );
	  if( iterB == mapAllNamedFeats.end() ) printf( "Not found %s feature \n",  
							iterImagePairvector->csFileNameB.c_str() );
	}
    }
  
  //compute extra pair distance:
  for( iterImagePairvector =vecExtraPairs.begin(); iterImagePairvector != vecExtraPairs.end(); iterImagePairvector++ )
    {
      iterA = mapAllNamedFeats.find( iterImagePairvector->csFileNameA );
      iterB = mapAllNamedFeats.find( iterImagePairvector->csFileNameB );
      
      if( iterA != mapAllNamedFeats.end() && iterB != mapAllNamedFeats.end() )
	{			
	  float fDistAB = -9999;
	  fDistAB = (*ptrCompareFunc)( iterA->second, iterB->second, method );
	  vecExtraDists[nExtraDistCount] = fDistAB;
	  nExtraDistCount ++;
	  printf( "%d th extra distance = %f\r", nExtraDistCount, fDistAB );
	}
      else
	{
	  if( iterA == mapAllNamedFeats.end() ) printf( "Not found %s feature \n",  
							iterImagePairvector->csFileNameA.c_str() );
	  if( iterB == mapAllNamedFeats.end() ) printf( "Not found %s feature \n",  
							iterImagePairvector->csFileNameB.c_str() );
	}
    }
  
  printf( "\nComplete distance computation.\n" );
  printf( "The number of intra distance is %d \n", nIntraDistCount );
  printf( "The number of extra distance is %d \n", nExtraDistCount );
  printf( "\n\n" );
  if( ofGlobalLogger.is_open() )
    {
      ofGlobalLogger << "\nComplete distance computation.\n";
      ofGlobalLogger << "The number of intra distance is " << nIntraDistCount;
      ofGlobalLogger << "The number of extra distance is " << nExtraDistCount << endl << endl;		
    }
  
  nValidIntraDist = nIntraDistCount;
  nValidExtraDist = nExtraDistCount;
  
  return 0;
}

float CompareGeneralFeatDistance( vector<float>& vecFeaturesA, 
				  vector<float>& vecFeaturesB, VMMRFeatDistType dType )
{
  float fDist = -9999;
  if( vecFeaturesA.size() != vecFeaturesB.size() ) {
    cout <<  "eature size not equal ! Cannot compare.\n";
    return -99999;		
  }
  
  float s1 = 0, s2 = 0, s3 = 0;
  float temp = 0;
  for( int n = 0; n < vecFeaturesA.size(); n ++ ){
    switch( dType )
      {
      case L1:
	s1 += std::abs( vecFeaturesA[n] - vecFeaturesB[n] );
	break;
      case L2:
	temp = ( vecFeaturesA[n] - vecFeaturesB[n] );
	s1 += (temp*temp );
	break;
      case COSINE:
	s1 +=  ( vecFeaturesA[n] * vecFeaturesB[n] );
	s2 += ( vecFeaturesA[n] * vecFeaturesA[n] );
	s3 += ( vecFeaturesB[n] * vecFeaturesB[n] );
	break;
      case CHISQR:
	s2 =  vecFeaturesA[n] - vecFeaturesB[n];
	s3 = vecFeaturesA[n] + vecFeaturesB[n];
	if( s3 > DBL_EPSILON ) {
	  s1 += ( (s2*s2) / s3 );
	}
	break;
      case INTERSECT:
	s1 += std::min(  vecFeaturesA[n], vecFeaturesB[n] );
	break;
      }
  }
  
  switch( dType )
    {
    case VMMR::L1:
      fDist = s1;
      break;
    case VMMR::L2:
      fDist = std::sqrt( s1 );
      break;
    case COSINE:
      fDist = ( s1 / (std::sqrt(s2) * std::sqrt(s3) ) );
      fDist = 1.f / fDist;
      break;
    case CHISQR:
      fDist = s1;
      break;
    case INTERSECT:
      fDist = s1;
      break;
    }
  
  return fDist;
}

int NormalizeHist( vector<float>& vecHistFeat, HistNormType enuNormType )
{
  float fEpslon = FLT_MIN;
  float s1 = 0, s2  = 0;
  float temp = 0;
  
  //for L2-Hys
  float thresh  = 0.2f;
  float scale = 0;
  
  for( int n = 0; n < vecHistFeat.size(); n ++ ){
    switch( enuNormType ) {
    case L1_NORM:
    case L1_SQRT:
      s1 += vecHistFeat[n];
      break;
    case L2_NORM:
    case L2_HYS:
      s1 += ( vecHistFeat[n]* vecHistFeat[n] );
      break;
    default:
      cout << "ERROR: not supported hist norm type!\n" ;
    }
  }
  for( int n = 0; n < vecHistFeat.size(); n ++ ) {
    switch( enuNormType ){
    case L1_NORM:
      vecHistFeat[n] = vecHistFeat[n]/( s1 + fEpslon );
      break;
    case L1_SQRT:
      vecHistFeat[n] = sqrt( vecHistFeat[n]/( s1 + fEpslon ) );
      break;
    case L2_NORM:
      vecHistFeat[n] = vecHistFeat[n] / sqrt( s1 + fEpslon*fEpslon );
      break;
    case L2_HYS:
      scale = 1.f / (std::sqrt(s1) + vecHistFeat.size()*0.1f);			
      vecHistFeat[n] = std::min( vecHistFeat[n]*scale, thresh );
      s2 += vecHistFeat[n] * vecHistFeat[n];
      break;		
    default:
      cout <<  "ERROR: not supported hist norm type!\n" ;
    }
  }
  
  if( enuNormType == L2_HYS ) {
    float scale = 1.f /( std::sqrt(s2) + 1e-3f);
    for( int n = 0; n < vecHistFeat.size(); n ++ ) {
      vecHistFeat[n] *= scale;
    }
  }
  
  return 0;
}

int NormalizeFeat( vector<float>& vecFeature, FeatNormType enFeatNormType )
{
  vector<float>::iterator vecIterMaxpos;
  float fMaxFeatVal = FLT_MIN;
  float fEpslon = 0.001;
  float fSumElemSq = 0.0f;
  
  switch( enFeatNormType ) {
  case FN_Norm01:
    vecIterMaxpos = max_element( vecFeature.begin(), vecFeature.end() );
    fMaxFeatVal = *vecIterMaxpos;
    fMaxFeatVal = max( fMaxFeatVal, fEpslon );
    for( int n = 0; n < vecFeature.size(); n ++ ) {			
      vecFeature[n] = vecFeature[n] / fMaxFeatVal;
    }
    break;
  case FN_L2:
    for( int n = 0; n < vecFeature.size(); n ++ ) {
      fSumElemSq += vecFeature[n] * vecFeature[n];
    }
    for( int n = 0; n < vecFeature.size(); n ++ ) {			
      vecFeature[n] = vecFeature[n] / sqrt(fSumElemSq + FLT_MIN * FLT_MIN );
    }
    break;
  case FN_Norm01_L2:
    // Norm01:
    vecIterMaxpos = max_element( vecFeature.begin(), vecFeature.end() );
    fMaxFeatVal = *vecIterMaxpos;
    fMaxFeatVal = max( fMaxFeatVal, fEpslon );
    for( int n = 0; n < vecFeature.size(); n ++ ) {			
      vecFeature[n] = vecFeature[n] / fMaxFeatVal;
    }
    // L2
    for( int n = 0; n < vecFeature.size(); n ++ ) {
      fSumElemSq += vecFeature[n] * vecFeature[n];
    }
    for( int n = 0; n < vecFeature.size(); n ++ ) {			
      vecFeature[n] = vecFeature[n] / sqrt(fSumElemSq + FLT_MIN * FLT_MIN );
    }
    break;
  default:
    cout << endl << "Not support feature normalization type !" << endl;
    return -1;
    
  }
}

// A list file line may have the following two format:
/* relpath/<filename>
   relpath/<filename>  <label>
   only one level path contained
   there is no any space in  relpath/<filename>.
   relpath is only one level.
   
   string& strPath: has end slash!
*/
int ParseListFileLine( string& strLine, string& strPath, string& strFileName, int& iLabel )
{
  string strCurrLine = strLine;
  TrimSpace( strCurrLine );
  
  int nCurrLabel = -1;
  string strRelpathFilename = "";
  
  size_t nSpacePos = strCurrLine.find_first_of( " " );
  if( nSpacePos == string::npos ) {
    strRelpathFilename = strCurrLine;
  } else {
    strRelpathFilename = strCurrLine.substr( 0, nSpacePos ); // there is label
    string strLabel = strCurrLine.substr( nSpacePos + 1 );
    nCurrLabel = atoi( strLabel.c_str() );
  }
  
  iLabel = nCurrLabel;
  
  string _strDriver, _strDir, _strFileName, _strExt;
  SplitPathFileNameExt( strRelpathFilename, _strDriver, _strDir, _strFileName, _strExt );
  
  strPath = _strDir;
  strFileName = _strFileName + _strExt;
  
  //See whether relpath is valid: only one level by finding slash
  // for windows _splitpath, "c:\good\test/aaa/image.jpg"
  // will : c: , \good\test/aaa/, image, .jpg 
  string strPathNoEndSlash = strPath.substr( 0, strPath.size()-1 );
  int nSlashPos1 = strPathNoEndSlash.find_first_of( "\\" );
  int nSlashPos2 = strPathNoEndSlash.find_first_of( "/" );
  if( nSlashPos1 != string::npos || nSlashPos2 != string::npos ) {
    cout << endl << "Error: there are muli-level path in relative path part !" << endl;
    return -1;
  }
  
  
  return 0;
}

/*
  Keep consistent with python version !
*/
string PatchKPIDToStr( int kpID )
{
  switch( kpID ){
  case -2:
    return "LogoArea";
  case -1:
    return "vface";
  case 0:
    return "WinGlassLT";  //   KP_WinGlassLT = 0 # Windshield Glass Left-top		
  case 1:
    return "WinGlassRT";  //   #KP_WinGlassRT = 1 #: Windshield Glass Right-top
  case 2:
    return "WinGlassLB";  //    KP_WinGlassLB = 2 #: Windshield Glass Left-bottom
  case 3:
    return "WinGlassRB "; //    KP_WinGlassRB =     3 #: Windshield Glass Right-bottom
  case 4:
    return "LeftHLamp";   //    KP_LeftHLamp = 4 #: Left Head Lamp center
  case 5:
    return "RightHLamp";  //    KP_RightHLamp = 5 #: Right Head Lamp center
  case 6:
    return "FrontBumpLB"; //    KP_FrontBumpLB = 6 #: Front Bumper Left Bottom corner
  case 7:
    return "FrontBumpRB"; //    KP_FrontBumpRB = 7 #: Front Bumper Right Bottom corner
  case 8:
    return "VehicleLogo"; //    KP_VehicleLogo = 8 #: Vehicle Logo center
  case 9:
    return "LicensePC";   //    KP_LicensePC = 9 #: License Plate center
  case 10:
    return "MidLineBot";  //    KP_MidLineBot = 10 #: Middle line bottom
  default:
    printf( "Invalid patch or key point id !" );
    return "BAD";
  }
}

bool IsPathEndWithSlash( string& strPath )
{
  if( ( strPath[strPath.size()-1] == '\\' ) || (strPath[strPath.size()-1]== '/' ) ){
    return true;
  } else {
    return false;
  }
}

} //end VMMR namespace
