#include "parse.h"
#include "datatype.h"

#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

parse::parse(string defname, vector<string> &lefname) {
  cout << "start lef parsing" << endl;
  parse_lef(lefname);
  cout << "start def parsing" << endl;
  parse_def(defname);
  cout << "end parsing" << endl;
}

parse::parse() { cout << "test parse 1" << endl; }

parse::~parse() {

}

void parse::parse_def(std::string defname) {
  vector<char> breaksigns;
  breaksigns.push_back(' ');
  string token;
  string line;
  ifstream _deffile;
  _deffile.open(defname);

  if (_deffile.is_open()) {
    
    do {
      getline(_deffile, line);
      strtok(line, breaksigns, token);
    } while (token != "DIEAREA");

    
    strtok(line, breaksigns, token);
    strtok(line, breaksigns, token);
    org.x = (double)stoi(token);
    strtok(line, breaksigns, token);
    org.y = (double)stoi(token);
    cout<<"org:"<<org.x<<" "<<org.y<<endl;
    
    strtok(line, breaksigns, token);
    strtok(line, breaksigns, token);
    strtok(line, breaksigns, token);
    end.x = (double)stoi(token);
    strtok(line, breaksigns, token);
    end.y = (double)stoi(token);
    cout<<"end:"<<end.x<<" "<<end.y<<endl;

    
    do {
      getline(_deffile, line);
      strtok(line, breaksigns, token);
    } while (token != "COMPONENTS");
    strtok(line, breaksigns, token);
    int component_number = stoi(token);
    
    for (int i = 0; i < component_number; ++i) {
      string tot_line = "";

      do {
        getline(_deffile, line);
        tot_line += line;
      } while (!isEndLine(line, ';'));

      strtok(tot_line, breaksigns, token);
      strtok(tot_line, breaksigns, token);
      string name = token;
      strtok(tot_line, breaksigns, token);
      string type = token;

      auto it = macro_lib.find(type);
      if (it != macro_lib.end()) {
        
        parseMacro(tot_line, name, it);
      } else {
        
      }
    }
    
    do {
      getline(_deffile, line);
      strtok(line, breaksigns, token);
    } while (token != "PINS");
    strtok(line, breaksigns, token);
    int PIN_number = stoi(token);
    // for loop read terminal pins
    for (int i = 0; i < PIN_number; ++i) {
      string tot_line = "";

      do {
        getline(_deffile, line);
        tot_line += line;
      } while (!isEndLine(line, ';'));

      strtok(tot_line, breaksigns, token);
      strtok(tot_line, breaksigns, token);
      string name = token;

      parseTERM(tot_line, name);
    }
    
    do {
      getline(_deffile, line);
      strtok(line, breaksigns, token);
    } while (token != "NETS");
    strtok(line, breaksigns, token);
    int NET_number = stoi(token);
    
    for (int i = 0; i < NET_number; ++i) {
      string tot_line = "";

      do {
        getline(_deffile, line);
        tot_line += line;
      } while (!isEndLine(line, ';'));
      strtok(tot_line, breaksigns, token);
      strtok(tot_line, breaksigns, token);
      string name = token;
      parseNET(tot_line, name);
    }
  }
}

void parse::parse_lef(std::vector<std::string> &lefname) {
  cout << "parse lef" << lefname.size() << endl;
  for (int i = 0; i < lefname.size(); ++i) 
  {
    
    cout << "parsing " << lefname.at(i) << endl;
    ifstream _leffile;
    _leffile.open(lefname[i]);
    string line;
    if (_leffile.is_open()) {
      MACRO cur_macro;

      while (line.substr(0, 5) != "MACRO") {
        getline(_leffile, line);
      }
      vector<char> breaksigns;
      breaksigns.push_back(' ');
     

      string token;
      strtok(line, breaksigns, token);
      strtok(line, breaksigns, token);
      cur_macro.name = token; 

      while (token != "SIZE") {
        getline(_leffile, line);
        while (line.length() > 0 and token != "SIZE") {
          strtok(line, breaksigns, token);
        }
      }
      strtok(line, breaksigns, token);
      cur_macro.size.x = (prec)stod(token); 
      cur_macro.half_size.x = cur_macro.size.x*0.5;
      strtok(line, breaksigns, token);
      strtok(line, breaksigns, token);
      cur_macro.size.y = (prec)stod(token); 
      cur_macro.half_size.y = cur_macro.size.y*0.5;

      // PIN info
      getline(_leffile, line);

      bool readPIN_flag;
      vector<PIN> pins;
      vector<FPOS> pos_v;
      do {
        PIN cur_pin;
        FPOS curpin_pos;
        readPIN_flag =
            readPIN(cur_macro, _leffile, line, breaksigns, cur_pin, curpin_pos);
        pins.push_back(cur_pin);
        pos_v.push_back(curpin_pos);
      } while (readPIN_flag);

      cur_macro.pof.swap(pos_v);
      cur_macro.pins.swap(pins);
      
      macro_lib.insert({cur_macro.name, cur_macro});
      cout << "pin reading finished" << endl;
    }
  }
}

void parse::write_proto(string filename)
{
  write_port();
  write_module();
  write_pb(filename);
}

void parse::write_pb(string filename)
{
  ofstream pbfile;
  pbfile.open(filename);
  pbfile<<"# proto-file: tensorflow/core/framework/graph.proto\n";
  pbfile<<"# proto-message: tensorflow.GraphDef\n";
  for(int i = 0;i < nodeInstance.size();++i)
  {
    Node node = nodeInstance[i];
    pbfile<<"node {\n";
   
    write_nodeName(pbfile, node.name);
    write_nodeInput(pbfile, node);
    write_Attr(pbfile, node);
    pbfile<<"}\n";
  }
  pbfile.close();
}
void parse::write_nodeName(ofstream& file,string name)
{
  file<<"  name: "<<"\""<<name<<"\"\n";
}
void parse::write_nodeInput(ofstream& file,Node &node){
  for(auto pin:node.inputPins)
  {
    file<<"  input: "<<"\""<<pin->name<<"\"\n";
  }
}
void parse::write_Attr(ofstream& file,Node &node){
  for(auto attr:node.Attrs)
  {
    file<<"  attr {\n";
    file<<"    key: "<<"\""<<attr.key<<"\"\n";
    file<<"    value {\n";
    if(attr.isNum)
    {
      file<<"      f: "<<attr.value<<"\n";
    }
    else{
      file<<"      placeholder: "<<"\""<<attr.value<<"\"\n";
    }
    file<<"    }\n";
    file<<"  }\n";
  }
}

void parse::write_port(){
  for(auto term:termInstance)
  {
    Node node;
    node.name = term.name;
    Attr side = Attr("side",false,determinePortSide(term));
    Attr type = Attr("type",false,"PORT");
    node.Attrs.push_back(side);
    node.Attrs.push_back(type);

    if(term.IO == 0){
      int netId = term.pins[0].netID;
      int curPinIdInNet = term.pins[0].pinIDinNet;
      NET* net = &netInstance[netId];
      for(auto pin : net->pins)
      {
        if(pin->pinIDinNet!=curPinIdInNet)
        {
          node.inputPins.push_back(pin);
        }
      }
    }

    Attr pos_x = Attr("x",true,to_string(term.center.x));
    Attr pos_y = Attr("y",true,to_string(term.center.y));
    node.Attrs.push_back(pos_x);
    node.Attrs.push_back(pos_y);
    nodeInstance.push_back(node);
  }
}
void parse::write_module(){
  for(auto macro:macroInstance)
  {
    Node node;
    node.name = macro.name;
    Attr height = Attr("height",true,to_string(macro.size.y));
    Attr width = Attr("width",true,to_string(macro.size.x));
    Attr type = Attr("type",false,"MACRO");
    Attr orient = Attr("orientation",false,macro.orient);
    Attr pos_x = Attr("x",true,to_string(macro.center.x));
    Attr pos_y = Attr("y",true,to_string(macro.center.y));

    node.Attrs.push_back(height);
    node.Attrs.push_back(width);
    node.Attrs.push_back(type);
    node.Attrs.push_back(orient);
    node.Attrs.push_back(pos_x);
    node.Attrs.push_back(pos_y);
    nodeInstance.push_back(node);
    write_pin_in_module(macro);
  }
}
void parse::write_pin_in_module(MACRO macro){
  for(int i =0;i < macro.pinCNTinObject;++i)
  {
    PIN pin = macro.pins[i];
    Node node;
    node.name = pin.name;
    if(pin.IO == 0)
    {
      NET* net = &netInstance[pin.netID];
      for(auto pinInNet:net->pins)
      {
        if(pinInNet->pinIDinModule != pin.pinIDinModule)
        {
          node.inputPins.push_back(pinInNet);
        }
      }
    }
    int macroId = pin.moduleID;
    int pinIdInMacro = pin.pinIDinModule;
    Attr macroName = Attr("macro_name",false,macroInstance[pin.moduleID].name);
    Attr type = Attr("type",false,"macro_pin");
    Attr x_offset = Attr("x_offset",true,to_string(macroInstance[macroId].pof[pinIdInMacro].GetX()));
    Attr y_offset = Attr("y_offset",true,to_string(macroInstance[macroId].pof[pinIdInMacro].GetY()));
    FPOS pos;
    pos.x = macroInstance[macroId].center.GetX()+macroInstance[macroId].pof[pinIdInMacro].GetX();
    pos.y = macroInstance[macroId].center.GetY()+macroInstance[macroId].pof[pinIdInMacro].GetY();
    Attr x = Attr("x",true,to_string(pos.GetX()));
    Attr y = Attr("y",true,to_string(pos.GetY()));

    node.Attrs.push_back(macroName);
    node.Attrs.push_back(type);
    node.Attrs.push_back(x_offset);
    node.Attrs.push_back(y_offset);
    node.Attrs.push_back(x);
    node.Attrs.push_back(y);
  }
}
void parse::write_pin(){

}

string parse::determinePortSide(TERM term)
{
  string side = "wrong";
  prec k = (end.y-org.y)/(end.x-org.x);
  prec x,y,h;
  x = term.center.x;
  y = term.center.y;
  h = end.y-org.y;
  if(y >= k*x && y <= (-k*x+h))
  {
    return "LEFT";
  }
  if(y <= k*x && y <= (-k*x+h))
  {
    return "BOTTOM";
  }
  if(y <= k*x && y >= (-k*x+h))
  {
    return "RIGHT";
  }
  if(y >= k*x && y >= (-k*x+h))
  {
    return "TOP";
  }

  
  return side;
}

void parse::strtok(string &str, vector<char> &signlist,
                   string &token) 
{

  int pos = str.length();
  int temp;
  int start_pos = 0;
  while (find(signlist.begin(), signlist.end(), (char)str[start_pos]) !=
         signlist.end()) {
    ++start_pos;
  }
  for (auto s : signlist) {
    temp = str.find(s, start_pos);
    if (temp < 0)
      continue;
    pos = temp < pos ? temp : pos;
  }

  token = str.substr(start_pos, pos - start_pos);

  
  while (find(signlist.begin(), signlist.end(), (char)str[pos]) !=
         signlist.end()) {
    ++pos;
  }
  str = str.substr(pos);
}

bool parse::readPIN(MACRO &curMacro, ifstream &leffile, string &nameline,
                    vector<char> &signlist, PIN &cur_pin, FPOS &curpin_pos) {
  string token;
  strtok(nameline, signlist, token);
  while (token != "PIN") {
    getline(leffile, nameline);
    strtok(nameline, signlist, token);
  }
  

  strtok(nameline, signlist, token);
  
  int _pinNum = curMacro.pinCNTinObject;
  if (token == "VSS" or token == "VDD" or token == "GND") {
    return false;
  }
  cur_pin.name = token;
  
  string line;
  getline(leffile, line);
  strtok(line, signlist, token);
  while (token != "END") {
    if (token == "DIRECTION") {
      strtok(line, signlist, token);
      if (token == "INPUT") {
        cur_pin.IO = 0;
      } else if (token == "OUTPUT") {
        cur_pin.IO = 1;
      }
    } else if (token == "USE") {
      strtok(line, signlist, token);
      if (token == "CLOCK") {
        cur_pin.clk = 1;
      } else {
        cur_pin.clk = 0;
      }
    } else if (token == "PORT") {
      getline(leffile, line);
      strtok(line, signlist, token);
      while (token != "RECT") {
        getline(leffile, line);
        strtok(line, signlist, token);
      }
     
      
      strtok(line, signlist, token);
      curpin_pos.x = (prec)stod(token);
      strtok(line, signlist, token);
      curpin_pos.y = (prec)stod(token);
      strtok(line, signlist, token);
      curpin_pos.x += (prec)stod(token);
      strtok(line, signlist, token);
      curpin_pos.y += (prec)stod(token);

      curpin_pos.x /=2;
      curpin_pos.y /=2;
      curpin_pos.x -= curMacro.half_size.x;
      curpin_pos.y -= curMacro.half_size.y;
      

    }
    getline(leffile, line);
    strtok(line, signlist, token);
  }
  ++curMacro.pinCNTinObject;
  getline(leffile, line);
  return true; 
}

bool parse::isEndLine(string line, char endSign) {
  return line.find(endSign) != string::npos;
}

void parse::parseMacro(string line, string name,
                       unordered_map<string, MACRO>::const_iterator it) {
  MACRO cur;

  cur.size.Set(it->second.size);
  cur.half_size.x = cur.size.x/2;
  cur.half_size.y = cur.size.y/2;
  cur.pof = it->second.pof;
  cur.pins = it->second.pins;
  cur.name = name;
  cur.type = it->second.name;
  cur.pinCNTinObject = it->second.pinCNTinObject;
  cur.idx = macroInstance.size();
  string token;
  vector<char> breaksigns;
  breaksigns.push_back(' ');
  breaksigns.push_back(';');
  strtok(line, breaksigns, token);
  if (token == "+") {
    strtok(line, breaksigns, token);
    if (token == "PLACED") {
      
      strtok(line, breaksigns, token);
      strtok(line, breaksigns, token);
      cur.center.x = (double)stoi(token);
      strtok(line, breaksigns, token);
      cur.center.y = (double)stoi(token);
      strtok(line, breaksigns, token);
      strtok(line, breaksigns, token);
      cur.orient = token;
    }
  }
  PIN *cur_pin;
  for (int i = 0; i < cur.pinCNTinObject; ++i) {
    cur_pin = &cur.pins[i];
    cur_pin->moduleID = cur.idx;
    cur_pin->pinIDinModule = i;
    cur_pin->type = macro;
    cur_pin->name = cur.name + it->second.pins[i].name;
  }
  ++macroCNT;
  macroInstance.push_back(cur);
  macro_namemap.insert({name, cur.idx});
}

void parse::parseTERM(string line, string name) {
  TERM cur;
  string token;
  vector<char> breaksigns;
  breaksigns.push_back(' ');
  breaksigns.push_back(';');
  cur.idx = termInstance.size();
  cur.name = name;
  term_namemap.insert({name,cur.idx});
  while (!line.empty()) {
    strtok(line, breaksigns, token);
    if (token == "DIRECTION") {
      strtok(line, breaksigns, token);
      if (token == "INPUT") {
        cur.IO = 0;
      } else if (token == "OUTPUT") {
        cur.IO = 1;
      }
    } else if (token == "FIXED") {
      strtok(line, breaksigns, token);
      strtok(line, breaksigns, token);
      cur.center.x = (double)stoi(token);
      strtok(line, breaksigns, token);
      cur.center.y = (double)stoi(token);
    } else if (token == "USE") {
      strtok(line, breaksigns, token);
      if (token == "CLOCK") {
        cur.clk = 1;
      }
    }
  }

  
  PIN p;
  p.term = 1;
  p.name = name;
  p.IO = cur.IO;
  p.type = term;
  p.pinIDinModule = 0;
  p.moduleID = cur.idx;
  p.clk = cur.clk;

  cur.pinCNTinObject = 1;
  FPOS pos = FPOS(0);
  cur.pof.push_back(pos);
  cur.pins.push_back(p);

  termInstance.push_back(cur);
}

void parse::parseNET(string line, string name) {
  string token;
  vector<char> breaksigns;
  breaksigns.push_back(' ');
  breaksigns.push_back(';');
  NET cur;
  cur.idx = netInstance.size();

  cur.name = name;

  
  auto it = term_namemap.find(name);
  if(it != term_namemap.end())
  {
    TERM *curTerm = &termInstance[it->second];
    PIN *curPIN = &curTerm->pins[0];
    cur.pins.push_back(curPIN);
    ++cur.pinCNTinObject;
  }
  
  ++netCNT;
  

  while (!line.empty()) {
    strtok(line, breaksigns, token);
    if (token == "(") {
      strtok(line, breaksigns, token);
      auto it = macro_namemap.find(token);
      int idx = it->second;
      if (it != macro_namemap.end()) {
        strtok(line, breaksigns, token);
        MACRO *curMacro = &macroInstance[idx];
        PIN *curPIN;
        for (int i = 0; i < curMacro->pinCNTinObject; ++i) {
          curPIN = &curMacro->pins[i];
          if (curPIN->name == token) {
            curPIN->netID = cur.idx;
            curPIN->pinIDinNet = cur.pinCNTinObject;
            ++cur.pinCNTinObject;
            cur.pins.push_back(curPIN);
            break;
          }
        }
      }
    }
  }
  netInstance.push_back(cur);
}
