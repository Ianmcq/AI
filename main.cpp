#include<iostream>
#include<fstream>
#include<math.h>
#include<random>
#include<array>
#include<vector>
#include"FeedForward.h"

int highestoutput(FeedForward f){
 int max = 0;
 for(int i = 0; i < f.getoutputsize(); i++){
  if(f.getoutput(i) > f.getoutput(max)){
   max = i;
  }
 }
 return max;
}

int main(char argv[], int args){
 if(args != 0){
  std::cout << "No arguments please." << endl;
  return 0;
 }
 std::cout << "Deep learning with ANN, an evolutionary approach. Compiled on " __DATE__ << endl;
 return 0;
}
