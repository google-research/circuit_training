#ifndef __parse__
#define __parse__


#include "datatype.h"
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
class parse{
    public:
    parse();
    parse(string defname,vector<string> &lefname);
    void parse_def(string defname);
    void parse_lef(vector<string> &lefname);
    void parse_netlist(string filename);
    void write_proto(string filename);
    FPOS org,end;
    ~parse();

    private:

    unordered_map<string,int> macro_namemap,term_namemap,module_namemap,net_namemap;
    unordered_map<string, MODULE> module_lib;
    unordered_map<string, TERM> term_lib;
    unordered_map<string, MACRO> macro_lib;

    MACRO* parse_macro_lef(string macroname);
    char* str2tok(string &str);//string 转换为 tokens
    void strtok(string &str,vector<char> &signlist,string &token);
    bool readPIN(MACRO &curMacro, ifstream &leffile,string &nameline,vector<char> &signlist,PIN &curPin,FPOS &curpin_pos);
    bool isEndLine(string line,char endSign);
    void parseMacro(string line,string name,unordered_map<string, MACRO>::const_iterator it);
    void parseTERM(string line,string name);
    void parseNET(string line,string name);

    void write_port();
    string determinePortSide(TERM term);
    void write_module();
    void write_pin_in_module(MACRO macro);
    void write_pin();
    void write_pb(string filename);
    void write_nodeName(ofstream& file,string name);
    void write_nodeInput(ofstream& file,Node &node);
    void write_Attr(ofstream& file,Node &node);
};

#endif