#ifndef datatype
#define datatype

#include <iostream>
#include <stdint.h>
#include <map>
#include <string.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <stdio.h>
using namespace std;
#define IS_FLOAT 0
#define IS_DOUBLE 1
#define PREC_MODE IS_FLOAT




typedef double prec;
#define PREC_MAX DBL_MAX
#define PREC_MIN DBL_MIN
#define PREC_EPSILON numeric_limits< double >::epsilon()


enum moduleType {module, term, macro};
struct POS;
struct FPOS {
  prec x;
  prec y;

  FPOS();
  FPOS(prec xloc, prec yloc);
  FPOS(prec loc);
  void Set(prec a);
  void Set(FPOS a);

  void Set(prec xloc, prec yloc);
  void Set(POS a);
  void SetZero();

  prec GetX();
  prec GetY(); 

  void Add(FPOS a);
  void Minus(FPOS a);
  void SetAdd(FPOS a, FPOS b);

  void Min(FPOS a);
  void SetMin(FPOS a, FPOS b);

  void Max(FPOS a);
  void SetMax(FPOS a, FPOS b);

  prec GetProduct();
  void Dump();
  void Dump(std::string a);

  
};
struct POS {
  int x;
  int y;

  POS();
  POS(int xloc, int yloc);

  void Set(int a);
  void Set(POS a);

  void Set(int xloc, int yloc);

  void Set(FPOS fp);
  void SetZero();
  void Add(POS a);
  

  void SetAdd(POS a, POS b);
  void Min(POS a);
  void SetMin(POS a, POS b);
  void Max(POS a); 
  void SetMax(POS a, POS b);
  int GetProduct();
  void SetXProjection(int a, int b);
  void SetYProjection(int a, int b);
  void SetProjection(POS a, POS b);
  void SetXYProjection(POS a, POS b);
  void Dump(); 
  void Dump(std::string a);
};


extern std::vector<std::string> moduleNameStor; 
extern std::vector<std::string> terminalNameStor;
extern std::vector<std::string> cellNameStor;
extern std::vector<std::string> netNameStor;


struct PIN {
  FPOS fp;
  FPOS e1;
  FPOS e2;
  POS flg1;
  POS flg2;
  moduleType type;
  int moduleID;
  int pinIDinModule;
  int netID;
  int pinIDinNet;
  int gid;   
  int IO;    
  int term;  
  int X_MIN;
  int Y_MIN;
  int X_MAX;
  int Y_MAX;
  int clk; 
  string name;
  PIN(); 
  
};

struct MODULE {
  FPOS pmin;
  FPOS pmax;
  FPOS size;
  FPOS half_size;
  FPOS center;
  FPOS *pof;
  PIN **pin;
  prec area;
  int idx;
  int netCNTinObject;
  int pinCNTinObject;
  int flg;
  int tier;
  int mac_idx;
  int ovlp_flg;
  POS pmin_lg;
  POS pmax_lg;

  const char* Name();
  MODULE();
  void Dump(std::string a);
};

struct TERM {
  
  FPOS pmin;
  FPOS pmax;
  prec area;
  FPOS size;
  FPOS center;
  vector<FPOS> pof;
  vector<PIN> pins;
  int idx;
  int netCNTinObject;
  int pinCNTinObject;
  int IO;  
  
  bool isTerminalNI;
  prec PL_area;
  int clk; 
  const char* Name();

  TERM();
  string orient;
  string name;
  void Dump();
};

struct MACRO {
  
  FPOS pmin;
  FPOS pmax;
  FPOS size;
  FPOS half_size;
  FPOS center;
  vector<FPOS> pof;
  vector<PIN> pins;
  prec area;
  int idx;
  int netCNTinObject;
  int pinCNTinObject;
  int flg;//don't use
  int tier;//don't use
  int mac_idx;
  int ovlp_flg;
  POS pmin_lg;
  POS pmax_lg;
  string orient;
  string name;
  string type;
  const char* Name();
  MACRO();
  void Dump(std::string a);
};


struct CELL {
  FPOS pmin;
  FPOS pmax;
  FPOS den_pmin;
  FPOS den_pmax;
  FPOS center;
  prec area;
  FPOS *pof;
  PIN **pin;
  prec x_grad;
  prec y_grad;
  int tier;
  FPOS size;
  prec den_scal;
  FPOS half_size;
  FPOS half_den_size;
  int idx;
  int pinCNTinObject;
  int netCNTinObject;
  int flg;
  prec area_org_befo_bloating;
  prec inflatedNewArea;
  prec inflatedNewAreaDelta;
  prec den_scal_org_befo_bloating;
  FPOS size_org_befo_bloating;
  FPOS half_size_org_befo_bloating;
  FPOS half_den_size_org_befo_bloating;
  FPOS *pof_tmp;
  PIN **pin_tmp;
  const char* Name();
  void Dump();
};

class TwoPinNets {
 public:
  bool selected;
  int start_modu;
  int end_modu;
  int rect_dist;
  int i;
  int j;

  TwoPinNets();
  TwoPinNets(int start_modu, int end_modu, 
      prec rect_dist, int i, int j);

};

struct NET {
  public:
 
  std::vector< TwoPinNets > two_pin_nets;

  prec min_x;
  prec min_y;
  prec max_x;
  prec max_y;
  FPOS sum_num1;
  FPOS sum_num2;
  FPOS sum_denom1;
  FPOS sum_denom2;
  
  vector<PIN*> pins;
 
  FPOS terminalMin;
  FPOS terminalMax;

  prec hpwl_x;
  prec hpwl_y;
  prec hpwl;
  int outPinIdx;           
  int pinCNTinObject;       
  
  int idx;
  int mod_idx;
  prec timingWeight;
  prec customWeight;
  prec wl_rsmt;             
  int clk; 
  string name;
  const char* Name();
  NET();
};

struct ROW {
  prec site_wid;  
  prec site_spa;  

  std::string ori;
  
  bool isXSymmetry;
  bool isYSymmetry;
  bool isR90Symmetry;

  int x_cnt;  
  FPOS pmin;
  FPOS pmax;
  FPOS size;

  ROW();
  void Dump(std::string a);
};

struct PLACE {
  FPOS org;
  FPOS end;
  FPOS center;
  FPOS stp;
  FPOS cnt;
  prec area;
  PLACE();
  void Dump(std::string a);
};

struct TIER {
  struct FPOS bin_org;
  struct FPOS inv_bin_stp;
  struct POS dim_bin;
  struct FPOS pmin;
  struct FPOS pmax;
  struct BIN *bin_mat;
  struct FPOS center;
  struct FPOS size;
  struct ROW *row_st;
  struct MODULE **modu_st;
  struct TERM **term_st;
  struct MODULE **mac_st;
  struct CELL **cell_st;
  struct FPOS bin_stp;
  prec area;
  prec modu_area;
  prec term_area;
  prec virt_area;
  prec filler_area;
  prec pl_area;
  prec ws_area;
  prec bin_area;
  prec tot_bin_area;
  prec inv_bin_area;
  prec sum_ovf;
  int row_cnt;
  int modu_cnt;
  int filler_cnt;
  int cell_cnt;
  int mac_cnt;
  int term_cnt;
  int tot_bin_cnt;
  prec temp_mac_area;
  struct FPOS bin_off;
  struct FPOS half_bin_stp;
  struct MODULE *max_mac;
  struct CELL **cell_st_tmp;

  // routability
  struct FPOS tile_stp;
  prec tile_area;
  struct POS dim_tile;
  int tot_tile_cnt;
  struct FPOS half_tile_stp;
  struct FPOS inv_tile_stp;
  struct FPOS tile_org;
  struct FPOS tile_off;
  struct TILE *tile_mat;
};

struct Attr{
  string key;
  bool isNum;
  string value;
  Attr();
  Attr(string _key,bool _isNum,string _value);
};
struct Node{
  string name;
  vector<PIN*> inputPins;
  vector<Attr> Attrs;
};

extern int terminalCNT;
extern int stdcellCNT;
extern int macroCNT;
extern int netCNT;


extern vector<MACRO> macroInstance;
// extern CELL *gcell_st;
extern vector<TERM> termInstance;
extern vector<NET> netInstance;
extern vector<Node> nodeInstance;
extern ROW *row_st;
extern PLACE *place_st;
#endif