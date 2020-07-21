#include<iostream>
#include<fstream>
#include<math.h>
#include<random>
#include<array>
#include<vector>
#include"FeedForward.h"

/*
Finds and returns the index of the maximum output of
a FeedForward given a pointer to it.
*/
int highestoutput(FeedForward *f){
 int max = 0;
 for(int i = 0; i < f->getoutputsize(); i++){
  if(f->getoutput(i) > f->getoutput(max)){
   max = i;
  }
 }
 return max;
}

int main(int argc, char **argv){
 //Detect command line arguments to the program. Exit if too many are found.
 if(argc > 2){
  std::cout << "No third argument please." << std::endl;
  std::cout << "Usage: AIbin [FILE]" << std::endl << "Tests the network saved to FILE by this program." << std::endl << "If no FILE provided, create and train new network." << std::endl;
  return 0;
 }
 //If only 2 arguments, interpret the second as the name of a neural network file to test.
 if(argc == 2){
  std::cout << "Testing a saved network's performance." << std::endl;
  std::vector<int> testlabels;
  std::vector<std::array<unsigned char, 784> > testpics;
  std::array<unsigned char, 784> nowpic;
  uint32_t nextint;
  unsigned char nextbyte;
  std::ifstream fstestlabel("./Test/t10k-labels-idx1-ubyte", std::ios::binary | std::ios::in);
  if(fstestlabel.is_open()){
   std::cout << "Opening labels." << std::endl;
   for(int i = 0; i < 2; i++){
    fstestlabel.read(reinterpret_cast<char *>(&nextint),sizeof(nextint));
   }
   for(int i = 0; i < 10000; i++){
    fstestlabel.read(reinterpret_cast<char *>(&nextbyte), sizeof(nextbyte));
    testlabels.push_back((int)nextbyte);
    //std::cout << (int)nextbyte << std::endl;
   }
  }
  fstestlabel.close();
  std::cout << "Closed labels." << std::endl;
 
  std::ifstream fstestpics("./Test/t10k-images-idx3-ubyte", std::ios::binary | std::ios::in);
  if(fstestpics.is_open()){
   std::cout << "Opening images." << std::endl;
   for(int i = 0; i < 4; i++){
    fstestpics.read(reinterpret_cast<char *>(&nextint),sizeof(nextint));
   }
   for(int picnum = 0; picnum < 10000; picnum++){
    int piccount = 0;
    testpics.push_back(std::array<unsigned char, 784>());
    for(int i = 0; i < 28; i++){
     for(int j = 0; j < 28; j++){
      fstestpics.read(reinterpret_cast<char *>(&nextbyte), sizeof(nextbyte));
      testpics[picnum][piccount] = nextbyte;
      piccount++;
     }
    } 
   }
  }
  fstestpics.close();
  std::cout << std::endl << "Closed images." << std::endl << std::endl;
  FeedForward evaluee(argv[1]);
  std::cout << "Evaluee loaded, it's dimensions are as follows." << std::endl;
  std::cout << "Evaluee ins: " << evaluee.getinputsize() << std::endl;
  std::cout << "Evaluee layers: " << evaluee.getlayers() << std::endl;
  std::cout << "Evaluee width: " << evaluee.getwidth() << std::endl;
  std::cout << "Evaluee outs: " << evaluee.getoutputsize() << std::endl;
  std::cout << "Test data assesment ongoing..." << std::endl;
  int right = 0;
  int wrong = right;
  double input;
  for(int j = 0; j < 10000; j++){
   for(int k = 0; k < 784; k++){
    input = (double)testpics[j][k]/((double)255.0);
    evaluee.setinput(k, input);
   }
   evaluee.feed();
   if(highestoutput(&evaluee) == testlabels[j]){
    right++;
   }else{
    wrong++;
   }
  }
  double result = ((double)right/(double)(right+wrong));
  std::cout << "Result is " << result << " accuracy over test data." << std::endl;
  return 0;
 }
 //If only 1 argument, output opening prompts for creation and training of a new network.
 std::cout << "Interactive deep learning with ANN, first with an evolutionary approach. Compiled on " __DATE__ " at " __TIME__ "." << std::endl;
 std::cout << "To begin we must construct the neural network. Input and output node count are 784 and 10 respectively, fixed by implementation" << std::endl;
 std::cout << "However, you must set the dimensions of the hidden portion of the network. How many hidden layers should it have?" << std::endl;
 unsigned int layers, nperl;//Number of hidden layers, and number of nodes per layer.
 int t;//temp for data validation

 //Prompt for and accept only a positive whole number twice, assigning them to layers and nperl.
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

 //Data acceptance prompt, then construct the trainee network using layers and nperl.
 std::cout << "Constructing network with the given specifications." << std::endl << layers << " is the number of hidden layers and " << nperl << " is the number of nodes in each hidden layer." << std::endl;
 FeedForward trainee(784, layers, nperl, 10);//The network we will be training to recognize handwritten digits.
 std::cout << "Trainee network constructed!" << std::endl;

 //initialize pseudorandom number generator system for evolutionary algorithm.
 std::random_device rd;
 std::mt19937 mt(rd());
 double range = 10.0;//the range of values for generating random networks
 std::uniform_real_distribution<double> dist(-1 * range, range);//initialize distribution for generating new network
 std::uniform_real_distribution<double> smalldist(-0.05, 0.05);//initialize narrower distribution for mutating existing network

 //Report to user and initialize trainee randomly
 std::cout << "At first the trainee will be completely random, and will predictably perform very poorly." << std::endl;
 for(int i = 0; i < trainee.getwidth(); i++){
  for(int j = 0; j < trainee.getinputsize(); j++){
   trainee.setweight(false, 0, i, j, dist(mt));
  }
  trainee.setbias(false,0,i,dist(mt));
 }
 
 for(int i = 1; i < trainee.getlayers(); i++){
  for(int j = 0; j < trainee.getwidth(); j++){
   for(int k = 0; k < trainee.getwidth(); k++){
    trainee.setweight(false, i, j, k, dist(mt));
   }
   trainee.setbias(false, i, j, dist(mt));
  }
 }

 for(int i = 0; i < trainee.getoutputsize(); i++){
  for(int j = 0; j < trainee.getwidth(); j++){
   trainee.setweight(true, i, j, 0, dist(mt));
  }
  trainee.setbias(true, i, 0, dist(mt));
 }

 //Prepare MNIST data from filesystem. NOTE setup.sh must have already run successfully.
 std::cout << "Now we need to load the training data to guage our trainee's performance and improve it." << std::endl;
 std::vector<int> labels;//Vector of 60,000 integer labels. Numbers 0-9 representing the digit a picture represents.
 std::vector<std::array<unsigned char, 784> > pics;//Vector of 60,000 grayscale 28x28 images of handwritten digits.
 std::array<unsigned char, 784> nowpic;//One 28x28 image for reading buffer.
 uint32_t nextint;//int buffer.
 unsigned char nextbyte;//single byte buffer.
 std::ifstream fstrainlabel("./Training/train-labels-idx1-ubyte", std::ios::binary | std::ios::in);//Training label input file stream
 if(fstrainlabel.is_open()){
  std::cout << "Opened training labels." << std::endl;
  for(int i = 0; i < 2; i++){//Read first two ints from file, these are magic numbers and should exist
   fstrainlabel.read(reinterpret_cast<char *>(&nextint),sizeof(nextint));
  }
  std::cout << "Reading labels into RAM." << std::endl;
  for(int i = 0; i < 60000; i++){
   fstrainlabel.read(reinterpret_cast<char *>(&nextbyte), sizeof(nextbyte));
   labels.push_back((int)nextbyte);
  }
 }
 fstrainlabel.close();
 std::cout << "Closed labels." << std::endl;
 std::ifstream fstrainpics("./Training/train-images-idx3-ubyte", std::ios::binary | std::ios::in);
 if(fstrainpics.is_open()){
  std::cout << "Opening training images." << std::endl;
  for(int i = 0; i < 4; i++){
   fstrainpics.read(reinterpret_cast<char *>(&nextint),sizeof(nextint));
  }
  int piccount;
  std::cout << "Loading images into RAM." << std::endl;
  for(int picnum = 0; picnum < 60000; picnum++){
   piccount = 0;
   pics.push_back(std::array<unsigned char, 784>());
   for(int i = 0; i < 28; i++){
    for(int j = 0; j < 28; j++){
     fstrainpics.read(reinterpret_cast<char *>(&nextbyte), sizeof(nextbyte));
     pics[picnum][piccount] = nextbyte;
     piccount++;
    }
   } 
  }
 }
 fstrainpics.close();
 std::cout << std::endl << "Closed Images." << std::endl << std::endl;

 //check that the 7th picture is of a 3, and that the 7th label is three. Also output the 3 as a sample to be sure the images are valid.
 int which = 7;
 std::cout << "Data integrity check." << std::endl;
 std::cout << "An image of a handwritten 3 constructed from numbers should appear in the terminal below." << std::endl;
 int count = 0;
 for(int i = 0; i < 28; i++){
  std::cout << std::endl;
  for(int j = 0; j < 28; j++){
   std::cout << (int)pics[which][count]%10;
   count++;
  }
 }
 std::cout << std::endl;
 if(labels[which] =! 3){
  std::cout << "Data is invalid. Delete Training and Test directories, rerun setup.sh, and try again.";
  return EXIT_FAILURE;
 }

 //Start preliminary assessment of the trainee network's accuracy. Should be around 0.1, indicating 1 out of 10 guesses are correct.
 std::cout << "Now we will use our training dataset to measure how accurately the trainee recognizes digits." << std::endl;
 std::cout << "The scale is from 0 (the trainee was wrong every time) to 1 (the trainee was correct every time)." << std::endl;
 std::cout << "Starting assesment..." << std::endl;
 double ta;//trainee accuracy
 int tr = 0, tw = 0;//trainee right answers and trainee wrong answers respectively
 for(int j = 0; j < 60000; j++){
  for(int k = 0; k < 784; k++){
   trainee.setinput(k, ((double)pics[j][k])/((double)255.0));//Set the trainee's inputs to each byte of the image.
  }
  trainee.feed();//Trainee calculates outputs.
  if(highestoutput(&trainee) == labels[j]){//Check which output of trainee is the highest, compare to image j's label, and count the right and wrong answers
   tr++;
  }else{
   tw++;
  }
 }
 ta = ((double)tr/(double)(tr+tw));//Calculate accuracy
 std::cout << "Done with trainee accuracy assesment!" << std::endl;
 std::cout << "The trainee's accuracy was only " << ta << "." << std::endl;

 //Initialize population of random networks for evolution.
 std::cout << "Now we will make a population of 10 more random networks." << std::endl;
 std::vector<FeedForward> population;//Population of neural networks to evolve.
 std::vector<FeedForward> newpop;//Next generation buffer population.
 std::vector<double> accuracy;//Accuracy of each network in population.
 FeedForward mangle(784, layers, nperl, 10);//Temporary network for creating and mutating networks in the population.
 for(int i = 0; i < 10; i++){//To start randomize temp 10 times and push copies to population.

  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, dist(mt));
   }
   mangle.setbias(false,0,i,dist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, dist(mt));
    }
    mangle.setbias(false, i, j, dist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, dist(mt));
   }
   mangle.setbias(true, i, 0, dist(mt));
  }
  population.push_back(mangle);
  std::cout << "Done adding net " << i + 1 << " to population." << std::endl;
 }

 //Population assessment. Essentially the same as above trainee assessment, but with array of FeedForward objects
 std::cout << "Next we will assess this population in a like manner to the trainee." << std::endl;
 std::cout << "Starting accuracy calculation..." << std::endl;
 int right, wrong;
 for(int i = 0; i < 10; i++){
  right = 0;
  wrong = right;
  for(int j = 0; j < 60000; j++){
   for(int k = 0; k < 784; k++){
    population[i].setinput(k, ((double)pics[j][k])/((double)255.0));
   }
   population[i].feed();
   if(highestoutput(&(population[i])) == labels[j]){
    right++;
   }else{
    wrong++;
   }
  }
  accuracy.push_back((double)right/(double)(right+wrong));
  std::cout << "Done with net accuracy " << i + 1 << "." << std::endl;
 }
 double max = -1;//Find index of best accuracy network.
 int maxi = 0;
 for(int i = 0; i < accuracy.size(); i++){
  if(accuracy[i] > max){
   max = accuracy[i];
   maxi = i;
  }
 }
 double average, sum = 0;//Calculate average performance of population to show general improvement over time.
 for(int i = 0; i < accuracy.size(); i++){
  sum = sum + accuracy[i];
 }
 average = sum/accuracy.size();
 std::cout << "Average accuracy: " << average << std::endl;
 std::cout << "Highest accuracy: " << max << std::endl;

 //Get from user how long to train.
 //For each iteration perform population assessment like above, then create new population from best performers for next iteration.
 std::cout << "Finally we will create a new population of 10 new networks based on the old 10." << std::endl;
 std::cout << "To do this we copy the network with the highest accuracy to the next generation." << std::endl;
 std::cout << "Then for the 4 most accurate networks we will copy them with slight variations." << std::endl;
 std::cout << "Last we will create a brand new random network to add to the new generation, to admit fresh tactics." << std::endl;
 std::cout << "We will repeat this process some number of times, then copy the best performer to our trainee." << std::endl;
 std::cout << "How many generations should we use to improve our trainee?" << std::endl;
 t = 0;
 int stop;//How many iterations to evolve.
 while(std::cout << "Enter a positive integer." && !(std::cin >> t) || t <= 0 ){
  std::cin.clear();
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::cout << "Caught invalid data. Try again." << std::endl;
 }
 stop = t;
 int gen = 0;//Generation counter.
 int first = 0, second = 0, third = 0, fourth = 0;//Top 4 network indices.
 double best;//Value of the best accuracy in population.
 std::vector<FeedForward> top4;//Vector of top 4 networks.

 //Training loop will repeat the number of times supplied by the user.
 do{
  std::cout << "Generation " << gen << " complete." << std::endl;
  std::cout << "Producing new population..." << std::endl;
  newpop.clear();
  top4.clear();
  //best copied
  newpop.push_back(population[maxi]);
  //one totally random
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, dist(mt));
   }
   mangle.setbias(false,0,i,dist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, dist(mt));
    }
    mangle.setbias(false, i, j, dist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, dist(mt));
   }
   mangle.setbias(true, i, 0, dist(mt));
  }
  newpop.push_back(mangle);
  //top 4 make 2
  best = 0;
  for(int i = 0; i < population.size(); i++){//Calculate top 4 indices.
   if(accuracy[i] > best){
    best = accuracy[i];
    first = i;
   }
  }
  best = 0;
  for(int i = 0; i < population.size(); i++){
   if(i != first){
    if(accuracy[i] > best){
     best = accuracy[i];
     second = i;
    }
   }
  }
  best = 0;
  for(int i = 0; i < population.size(); i++){
   if(i != first && i != second){
    if(accuracy[i] > best){
     best = accuracy[i];
     third = i;
    }
   }
  }
  best = 0;
  for(int i = 0; i < population.size(); i++){
   if(i != first && i != second && i != third){
    if(accuracy[i] > best){
     best = accuracy[i];
     fourth = i;
    }
   }
  }

  //First makes 2
  //TODO shorten these. More nested Loops? Macros?
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[first].getweight(false, 0, i, j) + smalldist(mt));//Uses smalldist to randomize about a networks current values.
   }
   mangle.setbias(false,0,i,population[first].getbias(false,0,i) + smalldist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[first].getweight(false, i, j, k) + smalldist(mt));
    }
    mangle.setbias(false, i, j, population[first].getbias(false, i , j) + smalldist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[first].getweight(true, i, j, 0) + smalldist(mt));
   }
   mangle.setbias(true, i, 0, population[first].getbias(true, i, 0) + smalldist(mt));
  }
  newpop.push_back(mangle);//1
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[first].getweight(false, 0, i, j) + smalldist(mt));
   }
   mangle.setbias(false,0,i,population[first].getbias(false,0,i) + smalldist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[first].getweight(false, i, j, k) + smalldist(mt));
    }
    mangle.setbias(false, i, j, population[first].getbias(false, i , j) + smalldist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[first].getweight(true, i, j, 0) + smalldist(mt));
   }
   mangle.setbias(true, i, 0, population[first].getbias(true, i, 0) + smalldist(mt));
  }
  newpop.push_back(mangle);//2

  //Second makes 2

  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[second].getweight(false, 0, i, j) + smalldist(mt));
   }
   mangle.setbias(false,0,i,population[second].getbias(false,0,i) + smalldist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[second].getweight(false, i, j, k) + smalldist(mt));
    }
    mangle.setbias(false, i, j, population[second].getbias(false, i , j) + smalldist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[second].getweight(true, i, j, 0) + smalldist(mt));
   }
   mangle.setbias(true, i, 0, population[second].getbias(true, i, 0) + smalldist(mt));
  }
  newpop.push_back(mangle);//1
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[second].getweight(false, 0, i, j) + smalldist(mt));
   }
   mangle.setbias(false,0,i,population[second].getbias(false,0,i) + smalldist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[second].getweight(false, i, j, k) + smalldist(mt));
    }
    mangle.setbias(false, i, j, population[second].getbias(false, i , j) + smalldist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[second].getweight(true, i, j, 0) + smalldist(mt));
   }
   mangle.setbias(true, i, 0, population[second].getbias(true, i, 0) + smalldist(mt));
  }
  newpop.push_back(mangle);//2

  //Third makes 2

  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[third].getweight(false, 0, i, j) + smalldist(mt));
   }
   mangle.setbias(false,0,i,population[third].getbias(false,0,i) + smalldist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[third].getweight(false, i, j, k) + smalldist(mt));
    }
    mangle.setbias(false, i, j, population[third].getbias(false, i , j) + smalldist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[third].getweight(true, i, j, 0) + smalldist(mt));
   }
   mangle.setbias(true, i, 0, population[third].getbias(true, i, 0) + smalldist(mt));
  }
  newpop.push_back(mangle);//1
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[third].getweight(false, 0, i, j) + smalldist(mt));
   }
   mangle.setbias(false,0,i,population[third].getbias(false,0,i) + smalldist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[third].getweight(false, i, j, k) + smalldist(mt));
    }
    mangle.setbias(false, i, j, population[third].getbias(false, i , j) + smalldist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[third].getweight(true, i, j, 0) + smalldist(mt));
   }
   mangle.setbias(true, i, 0, population[third].getbias(true, i, 0) + smalldist(mt));
  }
  newpop.push_back(mangle);//2

  //Fourth makes 2

  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[fourth].getweight(false, 0, i, j) + smalldist(mt));
   }
   mangle.setbias(false,0,i,population[fourth].getbias(false,0,i) + smalldist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[fourth].getweight(false, i, j, k) + smalldist(mt));
    }
    mangle.setbias(false, i, j, population[fourth].getbias(false, i , j) + smalldist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[fourth].getweight(true, i, j, 0) + smalldist(mt));
   }
   mangle.setbias(true, i, 0, population[fourth].getbias(true, i, 0) + smalldist(mt));
  }
  newpop.push_back(mangle);//1
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[fourth].getweight(false, 0, i, j) + smalldist(mt));
   }
   mangle.setbias(false,0,i,population[fourth].getbias(false,0,i) + smalldist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[fourth].getweight(false, i, j, k) + smalldist(mt));
    }
    mangle.setbias(false, i, j, population[fourth].getbias(false, i , j) + smalldist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[fourth].getweight(true, i, j, 0) + smalldist(mt));
   }
   mangle.setbias(true, i, 0, population[fourth].getbias(true, i, 0) + smalldist(mt));
  }
  newpop.push_back(mangle);//2

  //Clear population and refill with new population.
  population.clear();
  for(int i = 0; i < newpop.size(); i++){
   population.push_back(newpop[i]);
  }

  //Assess new Population.
  std::cout << "Starting accuracy calculation..." << std::endl;
  accuracy.clear();
  for(int i = 0; i < 10; i++){
   right = 0;
   wrong = right;
   for(int j = 0; j < 60000; j++){
    for(int k = 0; k < 784; k++){
     population[i].setinput(k, ((double)pics[j][k])/((double)255.0));
    }
    population[i].feed();
    if(highestoutput(&(population[i])) == labels[j]){
     right++;
    }else{
     wrong++;
    }
   }
   accuracy.push_back((double)right/(double)(right+wrong));
   std::cout << "Done with net accuracy " << i + 1 << "." << std::endl;
  }
  max = -1;
  maxi = 0;
  for(int i = 0; i < accuracy.size(); i++){
   if(accuracy[i] > max){
    max = accuracy[i];
    maxi = i;
   }
  }
  double average, sum = 0;
  for(int i = 0; i < accuracy.size(); i++){
   sum = sum + accuracy[i];
  }
  average = sum/accuracy.size();
  std::cout << "Average accuracy: " << average << std::endl;
  std::cout << "Highest accuracy: " << max << std::endl;
 }while(++gen < stop);
 //Update and save trainee with name prompted from user.
 trainee = new FeedForward(population[maxi]);//Create network that's a copy of the best performer, and assign to trainee.
 std::cout << "Training finished. Now let's save our graduated trainee for later. What would you like to name it?" << std::endl;
 std::string tname;
 std::cin >> tname;
 trainee.tofile(tname);
 std::cout << "Now your trainee is saved. That completes this evolutionary AI training program. Thank you for trying it out." << std::endl;
 return 0;
}
