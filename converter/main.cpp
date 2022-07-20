// #include <iostream>
// #include <vector>
// #include <string>
#include "datatype.h"
#include "parse.h"
#include <vector>

vector<MACRO> macroInstance;
vector<NET> netInstance;
vector<TERM> termInstance;
vector<Node> nodeInstance;
int macroCNT;
int netCNT;
int main(int argc, char* argv[]) {
  macroInstance.clear();
  netInstance.clear();
  termInstance.clear();
  macroCNT = 0;
  netCNT = 0;
  string def_name ;
  vector<string> lef_name;
  string pb_file_name;
    
  for(int i = 1;i<argc;++i)
  {
      
      if( strcmp(argv[i],"-def") == 0)
      {
          
          i++;
          
          if(argv[i][0] != '-')
          {
              pb_file_name = argv[i];
          }
      }
      if( strcmp(argv[i],"-pb") == 0)
      {
          
          i++;
          
          if(argv[i][0] != '-')
          {
              def_name = argv[i];
          }
      }
      if(strcmp(argv[i],"-lef")==0){
          i++;
          while(i<argc && argv[i][0] != '-')
          {
              
              lef_name.push_back(argv[i]);
              ++i;
          }
      }
      
  }
  
  // def lef 文件名初始，demo 版本，需要改为 arg 输入
  

  

  parse parser(def_name, lef_name);

  parser.write_proto("/home/Regularity-Based-ASIC-Design-Flow/circuit/test.pb.txt");

//   parser.~parse();
  // parse p;
  
  return 0;
}