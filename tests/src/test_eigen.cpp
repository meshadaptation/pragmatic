#include <Eigen/Core>
#include <Eigen/Dense>

int main(){
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(6,6);
  
  A[0] = 0.959;   A[1] = 0.4595;  A[2] = 0.4882;  A[3] = 1.478;   A[4] = 0.8364;  A[5] = 2.405;
  A[6] = 0.4595;  A[7] = 0.7646;  A[8] = 0.559;   A[9] = 0.8651;  A[10] = 1.098;  A[11] = 1.708;
  A[12] = 0.4882; A[13] = 0.559;  A[14] = 0.4595; A[15] = 0.8364; A[16] = 0.8651; A[17] = 1.523;
  A[18] = 1.478;  A[19] = 0.8651; A[20] = 0.8364; A[21] = 2.405;  A[22] = 1.523;  A[23] = 4.2;
  A[24] = 0.8364; A[25] = 1.098;  A[26] = 0.8651; A[27] = 1.523;  A[28] = 1.708;  A[29] = 2.95;
  A[30] = 2.405;  A[31] = 1.708;  A[32] = 1.523;  A[33] = 4.2;    A[34] = 2.95;   A[35] = 8;

  Eigen::Matrix<double, Eigen::Dynamic, 1> b = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(6);
  b[0] = 1.419;
  b[1] = 1.224;
  b[2] = 1.047;
  b[3] = 2.343;
  b[4] = 1.934;
  b[5] = 4.113;
  
  Eigen::Matrix<double, Eigen::Dynamic, 1> a = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(6);
  
  // Reference solution from MATLAB
  Eigen::Matrix<double, Eigen::Dynamic, 1> ref_a = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(6);
  ref_a[0] = 1.2767;
  ref_a[1] = 1.0376;
  ref_a[2] = -0.0949;
  ref_a[3] = -0.2713;
  ref_a[4] = 0.0287;
  ref_a[5] = 0.0588;
  
  bool pass = true;
  A.ldlt().solve(b, &a);
  for(int i=0;i<6;i++)
    if(fabs(a[i] - ref_a[i])>1.0e-4)
      pass = false;
  
  A.lu().solve(b, &a);
  for(int i=0;i<6;i++)
    if(fabs(a[i] - ref_a[i])>1.0e-4)
      pass = false;
  
  if(pass)
    std::cout<<"pass\n";
  else
    std::cout<<"fail\n";
  
  return 0;
}
