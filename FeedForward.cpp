#include"./FeedForward.h"
#include<iostream>

FeedForward::Node::Node(){
 activation = 0;
 bias = 0;
}

FeedForward::Node::Node(double b, unsigned int w){
 activation = 0;
 bias = b;
 for(int i = 0; i < w; i++){
  weights.push_back(1);
 }
}

double FeedForward::Node::getweight(unsigned int i){
 return weights[i];
}

void FeedForward::Node::setweight(unsigned int i, double w){
 weights[i] = w;
}

double FeedForward::Node::getact(){
 return activation;
}

void FeedForward::Node::setact(double a){
 activation = a;
}

double FeedForward::Node::getbias(){
 return bias;
}

void FeedForward::Node::setbias(double b){
 bias = b;
}

void FeedForward::init(){
 for(int i = 0; i < insize; i++){
  input.push_back(FeedForward::Node(0,0));
 }
 for(int i = 0; i < layers; i++){
  hidden.push_back(std::vector<FeedForward::Node>());
 }
 for(int i = 0; i < width; i++){
  hidden[0].push_back(FeedForward::Node(0.0, insize));
 }
 if(layers > 1){
  for(int i = 1; i < layers; i++){
   for(int j = 0; j < width; j++){
    hidden[i].push_back(FeedForward::Node(0.0, width));
   }
  }
 }
 for(int i = 0; i < outsize; i++){
  output.push_back(FeedForward::Node(0.0, width));
 }
}

FeedForward::FeedForward(){
 insize = 2;
 outsize = 1;
 layers = 1;
 width = 2;
 init();
}

FeedForward::FeedForward(unsigned int i, unsigned int l, unsigned int w, unsigned int o){
 insize = i;
 layers = l;
 width = w;
 outsize = o;
 init();
}

FeedForward::FeedForward(std::string path){
 std::ifstream infile(path, std::ios::binary | std::ios::in);
 int readint;
 double readdouble;
 if(infile.is_open()){
  infile.read(reinterpret_cast<char *>(&readint),sizeof(readint));
  insize = readint;
  infile.read(reinterpret_cast<char *>(&readint),sizeof(readint));
  layers = readint;
  infile.read(reinterpret_cast<char *>(&readint),sizeof(readint));
  width = readint;
  infile.read(reinterpret_cast<char *>(&readint),sizeof(readint));
  outsize = readint;
  init();
 }
 for(int i = 0; i < width; i++){
  infile.read(reinterpret_cast<char *>(&readdouble),sizeof(readdouble));
  hidden[0][i].setbias(readdouble);
  for(int j = 0; j < insize; j++){
   infile.read(reinterpret_cast<char *>(&readdouble),sizeof(readdouble));
   hidden[0][i].setweight(j, readdouble);
  }
 }
 for(int i = 1; i < layers; i++){
  for(int j = 0; j < width; j++){
   infile.read(reinterpret_cast<char *>(&readdouble),sizeof(readdouble));
   hidden[i][j].setbias(readdouble);
   for(int k = 0; k < width; k++){
    infile.read(reinterpret_cast<char *>(&readdouble),sizeof(readdouble));
    hidden[i][j].setweight(k, readdouble);
   }
  }
 }
 for(int i = 0; i < outsize; i++){
  infile.read(reinterpret_cast<char *>(&readdouble),sizeof(readdouble));
  output[i].setbias(readdouble);
  for(int j = 0; j < width; j++){
   infile.read(reinterpret_cast<char *>(&readdouble),sizeof(readdouble));
   output[i].setweight(j, readdouble);
  }
 }
 infile.close();
}

unsigned int FeedForward::getinputsize(){
 return insize;
}

unsigned int FeedForward::getoutputsize(){
 return outsize;
}

unsigned int FeedForward::getlayers(){
 return hidden.size();
}

unsigned int FeedForward::getwidth(){
 return hidden[0].size();
}

double FeedForward::getoutput(unsigned int i){
 return output[i].getact();
}

void FeedForward::setinput(unsigned int i, double a){
 input[i].setact(a);
}

void FeedForward::setweight(bool end, unsigned int i, unsigned int j, unsigned int k, double w){
 if(end){
  output[i].setweight(j,w);
  return;
 }
 hidden[i][j].setweight(k,w);
}

double FeedForward::getweight(bool end, unsigned int i, unsigned int j, unsigned int k){
 if(end){
  return output[i].getweight(j);
 }
 return hidden[i][j].getweight(k);
}

void FeedForward::setbias(bool end, unsigned int i, unsigned int j, double b){
 if(end){
  output[i].setbias(b);
  return;
 }
 hidden[i][j].setbias(b);
}

double FeedForward::getbias(bool end, unsigned int i, unsigned int j){
 if(end){
  return output[i].getbias();
 }
 return hidden[i][j].getbias();
}
//(e^x/e^x + 1) = sigmoid function
void FeedForward::feed(){
 double sum;

 for(int i = 0; i < width; i++){
  sum = 0;
  for(int j = 0; j < insize; j++){
   sum += hidden[0][i].getweight(j) * input[j].getact();
  }
  sum += hidden[0][i].getbias();
  hidden[0][i].setact(exp(sum)/(exp(sum) + 1));
 }

 if(layers > 1){
  for(int i = 1; i < layers; i++){
   for(int j = 0; j < width; j++){
    sum = 0;
    for(int k = 0; k < width; k++){
     sum += hidden[i][j].getweight(k) * hidden[i-1][k].getact();
    }
    sum += hidden[i][j].getbias();
    hidden[i][j].setact(exp(sum)/(exp(sum) + 1));
   }
  }
 }

 for(int i = 0; i < outsize; i++){
  sum = 0;
  for(int j = 0; j < width; j++){
   sum += output[i].getweight(j) * hidden[layers - 1][j].getact();
  }
  sum += output[i].getbias();
  output[i].setact(exp(sum)/(exp(sum) + 1));
 }

}

void FeedForward::tofile(std::string path){
 std::ofstream file;
 file.open(path, std::ios::out | std::ios::binary);
 int writeint;
 double writedouble;
 writeint = insize;
 file.write(reinterpret_cast<char *>(&writeint), sizeof(writeint));
 writeint = layers;
 file.write(reinterpret_cast<char *>(&writeint), sizeof(writeint));
 writeint = width;
 file.write(reinterpret_cast<char *>(&writeint), sizeof(writeint));
 writeint = outsize;
 file.write(reinterpret_cast<char *>(&writeint), sizeof(writeint));
 for(int i = 0; i < width; i++){
  writedouble = hidden[0][i].getbias();
  file.write(reinterpret_cast<char *>(&writedouble), sizeof(writedouble));
  for(int j = 0; j < insize; j++){
   writedouble = hidden[0][i].getweight(j);
   file.write(reinterpret_cast<char *>(&writedouble), sizeof(writedouble));
  }
 }
 
 for(int i = 1; i < layers; i++){
  for(int j = 0; j < width; j++){
   writedouble = hidden[i][j].getbias();
   file.write(reinterpret_cast<char *>(&writedouble), sizeof(writedouble));
   for(int k = 0; k < width; k++){
    writedouble = hidden[i][j].getweight(k);
    file.write(reinterpret_cast<char *>(&writedouble), sizeof(writedouble));
   }
  }
 }
 
 for(int i = 0; i < outsize; i++){
  writedouble = output[i].getbias();
  file.write(reinterpret_cast<char *>(&writedouble), sizeof(writedouble));
  for(int j = 0; j < width; j++){
   writedouble = output[i].getweight(j);
   file.write(reinterpret_cast<char *>(&writedouble), sizeof(writedouble));
  }
 }
 file.close();
}
