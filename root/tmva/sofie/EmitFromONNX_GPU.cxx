// Author: Federico Sossai
// Last modified: 2021/07/30
// Description:
//    SOFIE command line compiler.
//    This program is automatically run when the corresponding test target is built.
//    Usage example: $./EmitFromONNX indir/mymodel.onnx outdir/myname.hxx

#include <iostream>

#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

int main(int argc, char *argv[]){
   if (argc < 2) {
      std::cerr << "ERROR: missing input file\n";
      return -1;
   }

   std::string outname = argv[1];
   int last_slash = outname.find_last_of('/');
   std::string file = outname.substr(last_slash+1);
   file.erase(file.end()-5, file.end());
   
   RModelParser_ONNX parser;
   std::cout << "Parsing file " << argv[1] << std::endl;
   RModel model = parser.Parse(argv[1]);
   model.GenerateGPU();
   model.OutputGeneratedGPU(file + "GPU_FromONNX.hxx");

   return 0;
}
