#include "datatype.h"
#include <vector>

FPOS::FPOS(prec loc) { x = y = loc; }

void FPOS::Set(FPOS a) {
  x = a.x;
  y = a.y;
}

void FPOS::Minus(FPOS a) {
  x -= a.x;
  y -= a.y;
}

void FPOS::Add(FPOS a) {
  x += a.x;
  y += a.y;
}

prec FPOS::GetX(){
  return x;
}
prec FPOS::GetY(){
  return y;
}

void FPOS::SetZero() { x = y = (prec)0.0; }



void POS::SetZero() { x = y = 0; }

TERM::TERM() : pof(0), area(0.0), idx(0), netCNTinObject(0), clk(0) {
  size.SetZero();
  center.SetZero();
  pinCNTinObject = netCNTinObject = 0;
  pins = vector<PIN>(0);
}
MACRO::MACRO()
    : pof(0), area(0.0), idx(0), netCNTinObject(0), flg(0), tier(0),
      ovlp_flg(0) {
  size.SetZero();
  half_size.SetZero();
  center.SetZero();
  pinCNTinObject = netCNTinObject = 0;
  pins = vector<PIN>(0);
}

FPOS::FPOS() { x = y = 0.0; }

POS::POS() { x = y = 0; }

PIN::PIN() {
  clk = 0;
  fp.SetZero();
  e1.SetZero();
  e2.SetZero();
  flg1.SetZero();
  flg2.SetZero();
  netID = -1;
  term = 0;
}



PLACE::PLACE() {
  org.SetZero();
  end.SetZero();
  center.SetZero();
  stp.SetZero();
  cnt.SetZero();
  area = 0;
}

NET::NET() {
  pins = vector<PIN *>(0);
  pinCNTinObject = 0;
  clk = 0;
}

Attr::Attr(string _key,bool _isNum,string _value)
{
  key = _key;
  isNum = _isNum;
  value = _value;
}