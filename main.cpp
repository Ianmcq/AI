#include<iostream>
#include<fstream>
#include<math.h>
#include<random>
#include<array>
#include<vector>
#include"FeedForward.h"

//TODO add rest of documentation

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
 std::random_device rd;
 std::mt19937 mt(rd());
 double range = 10.0;
 std::uniform_real_distribution<double> dist(-1 * range, range);// use by "dist(mt)"
 std::uniform_real_distribution<double> lildist(-0.05, 0.05);// use by "dist(mt)"
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
 std::cout << "Now we need to load the training data to guage our trainee's performance and improve it." << std::endl;
 std::vector<int> labels;
 std::vector<std::array<unsigned char, 784> > pics;
 std::array<unsigned char, 784> nowpic;
 uint32_t nextint;
 unsigned char nextbyte;
 std::ifstream fstrainlabel("./Training/train-labels-idx1-ubyte", std::ios::binary | std::ios::in);
 if(fstrainlabel.is_open()){
  std::cout << "Opened training labels." << std::endl;
  for(int i = 0; i < 2; i++){
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
 std::cout << "Now we will use our training dataset to measure how accurately the trainee recognizes digits." << std::endl;
 std::cout << "The scale is from 0 (the trainee was wrong every time) to 1 (the trainee was correct every time)." << std::endl;
 std::cout << "Starting assesment..." << std::endl;
 double ta;
 int tr = 0, tw = 0;

 for(int j = 0; j < 60000; j++){
  for(int k = 0; k < 784; k++){
   trainee.setinput(k, ((double)pics[j][k])/((double)255.0));
  }
  trainee.feed();
  if(highestoutput(trainee) == labels[j]){
   tr++;
  }else{
   tw++;
  }
 }
 ta = ((double)tr/(double)(tr+tw));
 std::cout << "Done with trainee accuracy assesment!" << std::endl;
 std::cout << "The trainee's accuracy was only " << ta << "." << std::endl;
 std::cout << "Now we will make a population of 10 more random networks." << std::endl;
 std::vector<FeedForward> population;
 std::vector<FeedForward> newpop;
 std::vector<double> accuracy;
 FeedForward mangle(784, layers, nperl, 10);
 for(int i = 0; i < 10; i++){

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
   if(highestoutput(population[i]) == labels[j]){
    right++;
   }else{
    wrong++;
   }
  }
  accuracy.push_back((double)right/(double)(right+wrong));
  std::cout << "Done with net accuracy " << i + 1 << "." << std::endl;
 }
 
 double max = -1;
 int maxi = 0;
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
 std::cout << "Finally we will create a new population of 10 new networks based on the old 10." << std::endl;
 std::cout << "To do this we copy the network with the highest accuracy to the next generation." << std::endl;
 std::cout << "Then for the 4 most accurate remaining networks we will copy them with slight variations." << std::endl;
 std::cout << "Last we will create a brand new random network to add to the new generation, to admit fresh tactics." << std::endl;
 std::cout << "We will repeat this process some number of times, then copy the best performer to our trainee." << std::endl;
 std::cout << "How many generations should we use to improve our trainee?" << std::endl;
 t = 0;
 int stop;
 while(std::cout << "Enter a positive integer." && !(std::cin >> t) || t <= 0 ){
  std::cin.clear();
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::cout << "Caught invalid data. Try again." << std::endl;
 }
 t = stop;
 int gen = 0, first = 0, second = 0, third = 0, fourth = 0;
 double best;
 std::vector<FeedForward> top4;
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
  for(int i = 0; i < population.size(); i++){
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

  //FIRST MAKE 2

  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[first].getweight(false, 0, i, j) + lildist(mt));
   }
   mangle.setbias(false,0,i,population[first].getbias(false,0,i) + lildist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[first].getweight(false, i, j, k) + lildist(mt));
    }
    mangle.setbias(false, i, j, population[first].getbias(false, i , j) + lildist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[first].getweight(true, i, j, 0) + lildist(mt));
   }
   mangle.setbias(true, i, 0, population[first].getbias(true, i, 0) + lildist(mt));
  }
  newpop.push_back(mangle);//1
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[first].getweight(false, 0, i, j) + lildist(mt));
   }
   mangle.setbias(false,0,i,population[first].getbias(false,0,i) + lildist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[first].getweight(false, i, j, k) + lildist(mt));
    }
    mangle.setbias(false, i, j, population[first].getbias(false, i , j) + lildist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[first].getweight(true, i, j, 0) + lildist(mt));
   }
   mangle.setbias(true, i, 0, population[first].getbias(true, i, 0) + lildist(mt));
  }
  newpop.push_back(mangle);//2

  //SECOND MAKE 2

  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[second].getweight(false, 0, i, j) + lildist(mt));
   }
   mangle.setbias(false,0,i,population[second].getbias(false,0,i) + lildist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[second].getweight(false, i, j, k) + lildist(mt));
    }
    mangle.setbias(false, i, j, population[second].getbias(false, i , j) + lildist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[second].getweight(true, i, j, 0) + lildist(mt));
   }
   mangle.setbias(true, i, 0, population[second].getbias(true, i, 0) + lildist(mt));
  }
  newpop.push_back(mangle);//1
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[second].getweight(false, 0, i, j) + lildist(mt));
   }
   mangle.setbias(false,0,i,population[second].getbias(false,0,i) + lildist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[second].getweight(false, i, j, k) + lildist(mt));
    }
    mangle.setbias(false, i, j, population[second].getbias(false, i , j) + lildist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[second].getweight(true, i, j, 0) + lildist(mt));
   }
   mangle.setbias(true, i, 0, population[second].getbias(true, i, 0) + lildist(mt));
  }
  newpop.push_back(mangle);//2

  //THIRD MAKE 2

  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[third].getweight(false, 0, i, j) + lildist(mt));
   }
   mangle.setbias(false,0,i,population[third].getbias(false,0,i) + lildist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[third].getweight(false, i, j, k) + lildist(mt));
    }
    mangle.setbias(false, i, j, population[third].getbias(false, i , j) + lildist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[third].getweight(true, i, j, 0) + lildist(mt));
   }
   mangle.setbias(true, i, 0, population[third].getbias(true, i, 0) + lildist(mt));
  }
  newpop.push_back(mangle);//1
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[third].getweight(false, 0, i, j) + lildist(mt));
   }
   mangle.setbias(false,0,i,population[third].getbias(false,0,i) + lildist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[third].getweight(false, i, j, k) + lildist(mt));
    }
    mangle.setbias(false, i, j, population[third].getbias(false, i , j) + lildist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[third].getweight(true, i, j, 0) + lildist(mt));
   }
   mangle.setbias(true, i, 0, population[third].getbias(true, i, 0) + lildist(mt));
  }
  newpop.push_back(mangle);//2

  //FOURTH MAKE 2

  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[fourth].getweight(false, 0, i, j) + lildist(mt));
   }
   mangle.setbias(false,0,i,population[fourth].getbias(false,0,i) + lildist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[fourth].getweight(false, i, j, k) + lildist(mt));
    }
    mangle.setbias(false, i, j, population[fourth].getbias(false, i , j) + lildist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[fourth].getweight(true, i, j, 0) + lildist(mt));
   }
   mangle.setbias(true, i, 0, population[fourth].getbias(true, i, 0) + lildist(mt));
  }
  newpop.push_back(mangle);//1
  for(int i = 0; i < mangle.getwidth(); i++){
   for(int j = 0; j < mangle.getinputsize(); j++){
    mangle.setweight(false, 0, i, j, population[fourth].getweight(false, 0, i, j) + lildist(mt));
   }
   mangle.setbias(false,0,i,population[fourth].getbias(false,0,i) + lildist(mt));
  }
  
  for(int i = 1; i < mangle.getlayers(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    for(int k = 0; k < mangle.getwidth(); k++){
     mangle.setweight(false, i, j, k, population[fourth].getweight(false, i, j, k) + lildist(mt));
    }
    mangle.setbias(false, i, j, population[fourth].getbias(false, i , j) + lildist(mt));
   }
  }

  for(int i = 0; i < mangle.getoutputsize(); i++){
   for(int j = 0; j < mangle.getwidth(); j++){
    mangle.setweight(true, i, j, 0, population[fourth].getweight(true, i, j, 0) + lildist(mt));
   }
   mangle.setbias(true, i, 0, population[fourth].getbias(true, i, 0) + lildist(mt));
  }
  newpop.push_back(mangle);//2
  std::cout << "NPS" << newpop.size() << std::endl;
  population.clear();
  for(int i = 0; i < newpop.size(); i++){
   population.push_back(newpop[i]);
  }

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
   if(highestoutput(population[i]) == labels[j]){
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
 trainee = new FeedForward(population[maxi]);
 std::cout << "Training finished. Now let's save our graduated trainee for later. What would you like to name it?" << std::endl;
 std::string tname;
 std::cin >> tname;
 trainee.tofile(tname);
 std::cout << "Now your trainee is saved. That completes this evolutionary AI training program. Thank you for trying it out." << std::endl;
 return 0;
}
