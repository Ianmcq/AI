#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H
#include<vector>
#include<math.h>
#include<string>
#include<fstream>
class FeedForward{
 private:
  class Node{
   private:
    double activation;
    double bias;
    std::vector<double> weights;
   public:
    Node();
    Node(double b, unsigned int w);
    double getweight(unsigned int i);
    void setweight(unsigned int i, double w);
    double getact();
    void setact(double a);
    double getbias();
    void setbias(double b);
  };
  std::vector<Node> input;
  std::vector< std::vector<Node> > hidden;
  std::vector<Node> output;
  unsigned int insize, outsize, layers, width;
  void init();
 public:
  FeedForward();
  FeedForward(unsigned int i, unsigned int l, unsigned int w, unsigned int o);
  FeedForward(FeedForward *f);
  FeedForward(std::string path);
  unsigned int getinputsize();
  unsigned int getoutputsize();
  unsigned int getlayers();
  unsigned int getwidth();
  double getoutput(unsigned int i);
  void setinput(unsigned int i, double a);
  void setweight(bool end, unsigned int i, unsigned int j, unsigned int k, double w);
  double getweight(bool end, unsigned int i, unsigned int j, unsigned int k);
  void setbias(bool end, unsigned int i, unsigned int j, double b);
  double getbias(bool end, unsigned int i, unsigned int j);
  void feed();
  void tofile(std::string path);
};
#endif
