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

int main(int argc, char **argv){
 if(argc != 1){
  std::cout << "No arguments please." << std::endl;
  return 0;
 }
 std::cout << "Interactive deep learning with ANN, first with an evolutionary approach. Compiled on " __DATE__ " at " __TIME__ "." << std::endl;
 std::cout << "To begin we must construct the neural network. Input and output node count are 784 and 10 respectively, fixed by implementation" << std::endl;
 std::cout << "However, you must set the dimensions of the hidden portion of the network. How many hidden layers should it have?" << std::endl;
 unsigned int layers, nperl;//Number of hidden layers, and number of nodes per layer
 int t;//temp for data validation
 while(std::cout << "Enter a positive integer." && !(std::cin >> t) || t <= 0 ){
  std::cin.clear();
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::cout << "Caught invalid data. Try again." << std::endl;
 }
 layers = t;
 std::cout << "Now how many nodes in each hidden layer?" << std::endl;
 while(std::cout << "Enter a positive integer." && !(std::cin >> t) || t <= 0 ){
  std::cin.clear();
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::cout << "Caught invalid data. Try again." << std::endl;
 }
 nperl = t;
 std::cout << "Constructing network with the given specifications." << std::endl << layers << " is the number of hidden layers and " << nperl << " is the number of nodes in each hidden layer." << std::endl;
 FeedForward trainee(784, layers, nperl, 10);
 std::cout << "Trainee network constructed!" << std::endl;
 return 0;
}
