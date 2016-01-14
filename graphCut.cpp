//Comentario inutil
#include <exception>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <queue>
#include <utility>
#include <limits>

#include "maxflow/graph.h"

using namespace std;
using namespace cv;

typedef struct{
  int k;
  int l;
  int n;
} triple;

/*
  Calculate the intersection between two patches
  Mat Aaux: 1 if pixl already belongs to image, 0 if not
  Mat I: Original image which forms the patch
  Mat C: contains information concerning the pixel enumeration for the graph creation
  int count: count the number of pixels on the intersection. Important to respect a bijection between pixel and graph node
  int k,l: original's patch displacement
 */
void intersection(const Mat& Aaux, const Mat& I, Mat& C, int& count, int k, int l);

/*
  Calculate the gradient
  On inner pixels, we use the two closest neighbors; on the borders, only the closest pixel inside the image
  We decided to keep it an int function, due to the nature of our graphs (Graph<int, int, int>), for the gradient was implemented as
    an optimization. A float gradient function would demand changes all over the code
 */
int gradientVertical(int i, int j, Mat A);
int gradientHorizontal(int i, int j, Mat A);

/*
  
 */
Graph<int, int, int> createGraph(const Mat& A, const Mat& Aaux, const Mat& I, const Mat& C, triple** cuts, int index, int count, int k, int l, int& outputCount);


int main(){
  //testGCuts();
  Mat aux=imread("../strawberries.jpg");
  //imshow("aux", aux);

  srand(time(NULL));
  //waitKey();

  Mat I;
  triple** cuts;
  cvtColor(aux,I,CV_BGR2GRAY);

  int n = 2; //Output image size
  
  int rows = n*I.rows;
  int cols = n*I.cols;
  cout << rows << "   " << cols << "\n";
  Mat A(rows, cols, CV_8U);
  Mat Aaux(rows, cols, CV_8U);
  Mat C = Mat::zeros(rows, cols, CV_32F);
  cuts = new triple*[rows];
  for (int i = 0; i < rows; i++) cuts[i] = new triple[cols];

  //Codes the node number + 1
  int count = 1;
  int countOutput = 0;
  
  //Initialize the variables
  //Copy image I into A(0,0)
  for(int i=0; i < I.rows; i++){
    for(int j=0; j < I.cols; j++){
      A.at<uchar>(i, j) = I.at<uchar>(i,j);
      Aaux.at<uchar>(i, j) = 1;
      cuts[i][j].k = 0;
      cuts[i][j].l = 0;
      cuts[i][j].n = 0;

    }
  }

  //Test
  int index = 0;
  int max_it = 100;
  while(index < max_it){
    index++;
    //std::cout << index<< std::endl;
    count = 1;
    int k, l;

    if(index < 0){
      k = 0;
      l = 0;
    }
    
    else if(index < (n*n)){
      k = (index % n)*(I.rows);
      l = (index/n)*(I.cols);
    }
    
    else{
      //k = I.rows/2;
      //l = I.cols/2;
      k = rand() % rows;
      l = rand() % cols;
      if(k==0) k++;
      if(l==0) l++;

      if(k==rows-1) k--;
      if(l==cols-1) l--;
    }

    if(index % 25 == 0)
      cout << index << " iterations of " << max_it << endl;
    
    //imshow("A1", A);
    intersection(Aaux, I, C, count, k, l);
  
  
    Graph<int, int, int> g = createGraph(A, Aaux, I, C, cuts, index, count, k, l, countOutput);
    //cout << "teste " << count << endl;
    int flow = g.maxflow();
    //cout << "Flow = " << flow << endl;

    //Copy all of I into A
    int cosmo = count;
    for(int i=0; i < I.rows; i++){
      for(int j=0; j < I.cols; j++){
        if(i + k < A.rows && j+l < A.cols && int(C.at<float>(i+k, j+l)) > 0){ //meaning this was in the intersection
          //find if it enters the cut part
          if(i+k+1 < C.rows && cuts[i+k][j+l].n != cuts[i+k+1][j+l].n && int(C.at<float>(i+k+1, j+l)) > 0){
            if(g.what_segment(cosmo) == Graph<int,int,int>::SINK){
              A.at<uchar>(i+k+1, j+l) = I.at<uchar>(i+1,j);
              cuts[i+k+1][j+l].k = k;
              cuts[i+k+1][j+l].l = l;
              cuts[i+k+1][j+l].n = index;
              cosmo++;
            }            
          }

          if(j+l+1 < C.cols && cuts[i+k][j+l].n != cuts[i+k][j+l+1].n && int(C.at<float>(i+k, j+l+1)) > 0){
            if(g.what_segment(cosmo) == Graph<int,int,int>::SINK){
              A.at<uchar>(i+k, j+l+1) = I.at<uchar>(i,j+1);
              cuts[i+k][j+l+1].k = k;
              cuts[i+k][j+l+1].l = l;
              cuts[i+k][j+l+1].n = index;
              cosmo++;
            }            
          }
          if(g.what_segment(int(C.at<float>(i+k, j+l)) - 1) == Graph<int,int,int>::SINK){
            A.at<uchar>(i+k, j+l) = I.at<uchar>(i,j);
            cuts[i+k][j+l].k = k;
            cuts[i+k][j+l].l = l;
            cuts[i+k][j+l].n = index;
          }
          C.at<float>(i+k, j+l) = 0;
        }
        else if(i + k < A.rows && j+l < A.cols && int(C.at<float>(i+k, j+l)) == 0){
          A.at<uchar>(i+k, j+l) = I.at<uchar>(i,j);
          cuts[i+k][j+l].k = k;
          cuts[i+k][j+l].l = l;
          cuts[i+k][j+l].n = index;
        }
        if(i + k < Aaux.rows && j+l < Aaux.cols)
          Aaux.at<uchar>(i+k, j+l) = 1; //Update where the image is
      }
    }

    /*for (int i = count; i < countOutput; i++) {
      if(g.what_segment(i) == Graph<int,int,int>::SINK){
      A.at<uchar>(k, l) = I.at<uchar>(0,0);
      cuts[k][l].k = k;
      cuts[k][l].l = l;
      cuts[k][l].n = index;
      }
      }*/

  }
  imshow("A2", A);

  waitKey();

  int kPrime;
  int lPrime;

  Mat Acolor(rows, cols, CV_8UC3, Scalar(0, 0, 0));
  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < A.cols; j++) {
      if(Aaux.at<uchar>(i,j) == 1){
        kPrime = cuts[i][j].k;
        lPrime = cuts[i][j].l;
        //cout << aux.at<Vec3b>(i-kPrime, j-lPrime) << endl;;
        Acolor.at<Vec3b>(i,j) = aux.at<Vec3b>(i-kPrime, j-lPrime);
      }
    }

  }
  imshow("final", Acolor);
  imwrite("montage.jpg", Acolor);
  
  waitKey();

  
}


void intersection(const Mat& Aaux, const Mat& I, Mat& C, int& count, int k, int l){

  for(int i=0; i < I.rows; i++)
    for(int j=0; j < I.cols; j++)
      //See on matrix A if the point (k+i, l+j) has 1 or 0
      if( k+i < Aaux.rows and l+j < Aaux.cols)
	if(Aaux.at<uchar>(k+i, l+j) == 1){
	  C.at<float>(k+i, l+j) = count;
	  count ++;
	}
  
  return;

}

int gradientVertical(int i, int j, Mat A){
  int output;
  if(i == 0 && i < A.rows)
    output = (A.at<uchar>(i+1,j) - A.at<uchar>(i,j))/2;
  else if(i > 0 && i == A.rows)
    output = (A.at<uchar>(i,j) - A.at<uchar>(i-1,j))/2;
  else
    output = (A.at<uchar>(i+1,j) - A.at<uchar>(i-1,j))/2;
  return abs(output);
}

int gradientHorizontal(int i, int j, Mat A){
  int output;
  if(j == 0 && j < A.cols)
    output = (A.at<uchar>(i,j+1) - A.at<uchar>(i,j))/2;
  else if(j > 0 && j == A.cols)
    output = (A.at<uchar>(i,j) - A.at<uchar>(i,j-1))/2;
  else
    output = (A.at<uchar>(i,j+1) - A.at<uchar>(i,j-1))/2;
  return abs(output);
}


/*
  A : matrix which will contain the patches
  I : one patch
  C : matrix which contains which position in the graph is correspondent to each pixel
  count : number of intersecting pixels
  (k, l) : upper left position in which the patch I will be added
*/

Graph<int, int, int> createGraph(const Mat& A, const Mat& Aaux, const Mat& I, const Mat& C, triple** cuts, int index, int count, int k, int l, int& outputCount){
  
  int countAux = count;
  int kPrime, lPrime;
  int k2Prime, l2Prime;
  Graph<int, int, int> g(count, count);
  g.add_node(count);
  //k = l = 0;
  double norm = 0;
  int coefVertical = 1;
  int coefHorizontal = 1;
  int imax = std::numeric_limits<int>::max();
  //cout <<  "asdasd"<< "\n";
  //Percorrer a matriz C e it colocando as edges
  for(int i=0; i<I.rows; i++){
    for(int j=0; j<I.cols; j++){
      if(i + k < A.rows && j+l < A.cols && C.at<float>(i+k, j+l) > 0) {
        //Add edges from source and sink
        //	cout << int(C.at<float>(i+k, j+l)) - 1 << endl;
        
        if(index == 6 && i == 0 && j == 0) cout << "Erro " << "\n";


	
	if(j+l-1 >= 0 && int(C.at<float>(i+k, j+l-1)) == 0 && Aaux.at<uchar>(i+k, j+l-1) != 0)
          g.add_tweights( int(C.at<float>(i+k, j+l)) - 1, imax, 0 );
        else if(j+l-1 >= 0 && int(C.at<float>(i+k, j+l-1)) == 0 && Aaux.at<uchar>(i+k, j+l-1) == 0)
          g.add_tweights( int(C.at<float>(i+k, j+l)) - 1, 0, imax);

        if(j+l+1 < C.cols && int(C.at<float>(i+k, j+l+1)) == 0 && Aaux.at<uchar>(i+k, j+l+1) != 0)
          g.add_tweights( int(C.at<float>(i+k, j+l)) - 1,  imax, 0 );
        else if(j+l+1 < C.cols && int(C.at<float>(i+k, j+l+1)) == 0 && Aaux.at<uchar>(i+k, j+l+1) == 0)
          g.add_tweights( int(C.at<float>(i+k, j+l)) - 1,  0, imax );
	
        if(i+k+1 < C.rows && int(C.at<float>(i+k+1, j+l)) == 0 && Aaux.at<uchar>(i+k+1, j+l) != 0)
          g.add_tweights( int(C.at<float>(i+k, j+l)) - 1,  imax, 0 );
        else if(i+k+1 < C.rows && int(C.at<float>(i+k+1, j+l)) == 0 && Aaux.at<uchar>(i+k+1, j+l) == 0)
          {
          g.add_tweights( int(C.at<float>(i+k, j+l)) - 1,  0, imax );
          }
	
        if(i+k > 0 && int(C.at<float>(i+k-1, j+l)) == 0 && Aaux.at<uchar>(i+k-1, j+l) != 0)
          g.add_tweights( int(C.at<float>(i+k, j+l)) - 1,  imax, 0 );
        else if(i+k > 0 && int(C.at<float>(i+k-1, j+l)) == 0 && Aaux.at<uchar>(i+k-1, j+l) == 0){
          g.add_tweights( int(C.at<float>(i+k, j+l)) - 1,  0, imax );
        }
        //if(index == 6 && i == 0 && j == 0) cout << "Erro 111" << "\n";
       
	//cout << "1" << endl;
	/*if(int(C.at<float>(i+k, j+l-1)) == 0 || int(C.at<float>(i+k, j+l+1)) == 0 || int(C.at<float>(i+k-1, j+l)) == 0 || int(C.at<float>(i+k+1, j+l)) == 0){ 
	  if(Aaux.at<uchar>(i+k, j+l) == 1) //Already belongs to the image
	    g.add_tweights( int(C.at<float>(i+k, j+l)) - 1, 50000, 0 );
            }*/


	
        kPrime = cuts[i+k][j+l].k;
        lPrime = cuts[i+k][j+l].l;
        //Look at the neighbours and see if they belong to the intersection
        //if(index == 6 && i == 0 && j == 0) cout << "Erro 222" << "\n";

	//cout << "2" << endl;
        if(j+l+1 < C.cols && cuts[i+k][j+l].n != cuts[i+k][j+l+1].n && int(C.at<float>(i+k, j+l+1)) > 0){          

          k2Prime = cuts[i+k][j+l+1].k;
          l2Prime = cuts[i+k][j+l+1].l;
          if(i+k - kPrime >= 0 && j+l-lPrime >= 0 && i+k - k2Prime >= 0 && j+l-l2Prime >= 0){
	  //if(i+k - kPrime < 0 || j+l-lPrime < 0) cout << "Erro" << "\n";
	  //if(i+k - k2Prime < 0 || j+l-l2Prime < 0) cout << "Erro" << "\n";

            g.add_node(1);

	    coefHorizontal = gradientHorizontal(i+k-kPrime, j+l-lPrime, I) + gradientHorizontal(i+k-kPrime, j+l+1-lPrime, I) + gradientHorizontal(i, j, I) + gradientHorizontal(i, j+1, I);
	    if(coefHorizontal == 0) coefHorizontal = 1;
            g.add_edge(int(C.at<float>(i+k,j+l)) - 1 , countAux - 1, (abs(I.at<uchar>(i+k - kPrime,j+l-lPrime) - I.at<uchar>(i,j)) + abs(I.at<uchar>(i+k - kPrime,j+l+1 - lPrime) - I.at<uchar>(i,j+1)))/coefHorizontal+1, (abs(I.at<uchar>(i+k - kPrime,j+l - lPrime) - I.at<uchar>(i,j)) + abs(I.at<uchar>(i+k - kPrime,j+l+1 - lPrime) - I.at<uchar>(i,j+1)))/coefHorizontal + 1); 

	    coefHorizontal = gradientHorizontal(i+k-k2Prime, j+l-l2Prime, I) + gradientHorizontal(i+k-k2Prime, j+l+1-l2Prime, I) + gradientHorizontal(i, j, I) + gradientHorizontal(i, j+1, I);
	    if(coefHorizontal == 0) coefHorizontal = 1;
            g.add_edge(countAux - 1 , int(C.at<float>(i+k,j+l+1)) - 1, (abs(I.at<uchar>(i+k - k2Prime,j+l - l2Prime) - I.at<uchar>(i,j)) + abs(I.at<uchar>(i+k - k2Prime,j+l+1 - l2Prime) - I.at<uchar>(i,j+1)))/coefHorizontal+1, (abs(I.at<uchar>(i+k - k2Prime,j+l - l2Prime) - I.at<uchar>(i,j)) + abs(I.at<uchar>(i+k - k2Prime,j+l+1 - l2Prime) - I.at<uchar>(i,j+1)))/coefHorizontal+1 );
            if(index == 6 && i == 0 && j == 0) cout << j+l-lPrime << "    Erro 333" << "\n";

	    coefHorizontal = gradientHorizontal(i+k-kPrime, j+l-lPrime, I) + gradientHorizontal(i+k-k2Prime, j+l-l2Prime, I) + gradientHorizontal(i+k-kPrime, j+l+1 - lPrime, I) + gradientHorizontal(i+k - k2Prime, j+l+1 - l2Prime, I);
	    if(coefHorizontal == 0) coefHorizontal = 1;
            g.add_tweights(countAux - 1,   /* capacities */  0, (abs(I.at<uchar>(i+k - kPrime, j+l - lPrime) - I.at<uchar>(i+k - k2Prime, j+l - l2Prime)) + abs(I.at<uchar>(i+k - kPrime,j+l+1 - lPrime) - I.at<uchar>(i+k - k2Prime,j+l+1 - l2Prime)))/coefHorizontal+1 );

            countAux++;
	    }
        }


	//cout << "3" << endl;
        if(i+k+1 < C.rows && cuts[i+k][j+l].n != cuts[i+k+1][j+l].n && int(C.at<float>(i+k+1, j+l)) > 0){
          k2Prime = cuts[i+k+1][j+l].k;
          l2Prime = cuts[i+k+1][j+l].l;
          
          if(i+k - kPrime >= 0 && j+l-lPrime >= 0 && i+k - k2Prime >= 0 && j+l-l2Prime >= 0){
            g.add_node(1);

	    coefVertical = gradientVertical(i+k-kPrime, j+l-lPrime, I) + gradientVertical(i+k+1-kPrime, j+l-lPrime, I) + gradientVertical(i, j, I) + gradientVertical(i+1, j, I);
	    if(coefVertical == 0) coefVertical = 1;
            g.add_edge(int(C.at<float>(i+k,j+l)) - 1 , countAux - 1, (abs(I.at<uchar>(i+k - kPrime,j+l - lPrime) - I.at<uchar>(i,j)) + abs(I.at<uchar>(i+k+1 - kPrime,j+l - lPrime) - I.at<uchar>(i+1,j)))/coefVertical+1, (abs(I.at<uchar>(i+k - kPrime,j+l -lPrime) - I.at<uchar>(i,j)) + abs(I.at<uchar>(i+k+1 - kPrime,j+l - lPrime) - I.at<uchar>(i+1,j)))/coefVertical+1 );

	    coefVertical = gradientVertical(i+k-k2Prime, j+l-l2Prime, I) + gradientVertical(i+k+1-k2Prime, j+l-l2Prime, I) + gradientVertical(i, j, I) + gradientVertical(i+1, j, I);
	    if(coefVertical == 0) coefVertical = 1;
            g.add_edge(countAux - 1 , int(C.at<float>(i+k+1,j+l)) - 1, (abs(I.at<uchar>(i+k - k2Prime,j+l - l2Prime) - I.at<uchar>(i,j)) + abs(I.at<uchar>(i+k+1 - k2Prime,j+l - l2Prime) - I.at<uchar>(i+1,j)))/coefVertical+1, (abs(I.at<uchar>(i+k - k2Prime,j+l - l2Prime) - I.at<uchar>(i,j)) + abs(I.at<uchar>(i+k+1 - k2Prime,j+l - l2Prime) - I.at<uchar>(i+1,j)))/coefVertical+1 );
		       
            if(index == 6 && i == 0 && j == 0) cout << j+l - k2Prime << "Bug\n";
            if(index == 6 && i == 0 && j == 0) cout << abs(I.at<uchar>(i+k+1 - k2Prime,j+l - k2Prime))  << "    Erro 444" << "\n";

	    
	    coefVertical = gradientVertical(i+k-kPrime, j+l-lPrime, I) + gradientVertical(i+k - k2Prime, j+l-l2Prime, I) + gradientVertical(i+k+1 - kPrime, j+l - lPrime, I) + gradientVertical(i+k+1 - k2Prime, j+l - l2Prime, I);
	    if(coefVertical == 0) coefVertical = 1;
            g.add_tweights(countAux - 1,   /* capacities */  0, (abs(I.at<uchar>(i+k - kPrime, j+l - lPrime) - I.at<uchar>(i+k - k2Prime, j+l - l2Prime)) + abs(I.at<uchar>(i+k+1 - kPrime,j+l - lPrime) - I.at<uchar>(i+k+1 - k2Prime,j+l - l2Prime)))/coefVertical+1 );
            countAux++;                             

	    }
        }


	//cout << "4" << endl;
        if(i + k - 1 >= 0 && int(C.at<float>(i+k-1, j+l)) > 0 && cuts[i+k][j+l].n == cuts[i+k-1][j+l].n){
          coefVertical = gradientVertical(i+k, j+l, A) + gradientVertical(i+k-1, j+l, A) + gradientVertical(i, j, I) + gradientVertical(i-1, j, I);
          if(coefVertical == 0) coefVertical = 1;
          g.add_edge(int(C.at<float>(i+k,j+l)) - 1 , int(C.at<float>(i+k-1,j+l)) - 1, (int( abs(A.at<uchar>(i+k,j+l) - I.at<uchar>(i,j)) + abs(A.at<uchar>(i+k-1,j+l) - I.at<uchar>(i-1,j)) ))/coefVertical + 1, 0); 
        }
        if(i+k+1 < C.rows && int(C.at<float>(i+k+1, j+l)) > 0 && cuts[i+k][j+l].n == cuts[i+k+1][j+l].n){
          coefVertical = gradientVertical(i+k, j+l, A) + gradientVertical(i+k+1, j+l, A) + gradientVertical(i, j, I) + gradientVertical(i+1, j, I);
          if(coefVertical == 0) coefVertical = 1;
          g.add_edge(int(C.at<float>(i+k,j+l)) - 1 , int(C.at<float>(i+k+1,j+l)) - 1, (int( abs(A.at<uchar>(i+k,j+l) - I.at<uchar>(i,j)) + abs(A.at<uchar>(i+k+1,j+l) - I.at<uchar>(i+1,j)) ))/coefVertical + 1, 0);
        }

        if(j+l-1 >= 0 && int(C.at<float>(i+k, j +l - 1)) > 0 && cuts[i+k][j+l].n == cuts[i+k][j+l-1].n){
          coefHorizontal = gradientHorizontal(i+k, j+l, A) + gradientHorizontal(i+k, j+l-1, A) + gradientHorizontal(i, j, I) + gradientHorizontal(i, j-1, I);
          if(coefHorizontal == 0) coefHorizontal = 1;
          g.add_edge(int(C.at<float>(i+k,j+l)) - 1 , int(C.at<float>(i+k,j+l - 1)) - 1, (int(abs(A.at<uchar>(i+k,j+l) - I.at<uchar>(i,j)) + abs(A.at<uchar>(i+k,j+l-1) - I.at<uchar>(i,j-1)) ))/coefHorizontal + 1, 0);
        }
        if(j+l+1 < C.cols && int(C.at<float>(i+k, j+l+1)) > 0 && cuts[i+k][j+l].n == cuts[i+k][j+l+1].n){
          coefHorizontal = gradientHorizontal(i+k, j+l, A) + gradientHorizontal(i+k, j+l+1, A) + gradientHorizontal(i, j, I) + gradientHorizontal(i, j+1, I);
          if(coefHorizontal == 0) coefHorizontal = 1;          
          g.add_edge(int(C.at<float>(i+k,j+l)) - 1 , int(C.at<float>(i+k,j+l+1)) - 1, (int( abs(A.at<uchar>(i+k,j+l) - I.at<uchar>(i,j)) + abs(A.at<uchar>(i+k,j+l+1) - I.at<uchar>(i,j+1)) ))/coefHorizontal + 1, 0);
        }
      }
    }
  }
  outputCount = countAux;
  return g;
}


