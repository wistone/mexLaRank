// -*- C++ -*-
// Main functions for testing a Multiclass SVM Classifier learned with LaRank
// Copyright (C) 2008- Antoine Bordes

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#include <iostream>
#include <vector>
#include <algorithm>

#include <cstdlib>
//#include <sys/time.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cassert>

#include "vectors.h"
#include "LaRank.h"

// GLOBAL VARIALBLES
#define CMD_LEN 2048

#define NUM_OF_RETURN_FIELD 2
static const char *field_names[] = {
  "label",
  "output_instance",
};


// Kernel
int kernel_type;
double degree;
double kgamma;
double coef0;
bool hasOutput;

// Testing data
Exampler ex;
int nb_train;
long nb_kernel;

// Kernel function
double
kernel (int xi_id, int xj_id, void *kparam)
{
  SVector xi = ex.data[xi_id].inpt;
  SVector xj = ex.data[xj_id].inpt;
  double vdot = dot (xi, xj);
  nb_kernel++;
  switch (kernel_type)
    {
    case 0:
      return vdot;
    case 1:
      return pow (kgamma * vdot + coef0, degree);
    case 2:
      return exp (-kgamma * (ex.data[xi_id].norm + ex.data[xj_id].norm - 2 * vdot));
    }
  return 0;
}


void
exit_with_help ()
{
    mexPrintf(
    "Usage: [label, score] = mexLaRankClassify(testing_label_vector, testing_instance_matrix, "
            "training_label_vector, training_instance_matrix, model, 'libsvm_options')\n"
     "\nLA_RANK_CLASSIFY: uses models learned with the 'LaRank algorithm' for multiclass classification to make prediction.\n"
	   "options:\n"
	   "-t kernel function (default 0):\n"
	   "\t0 linear : K(X,X')=X*X'\n"
	   "\t1 polynomial : K(X,X')=(g*X*X'+c0)^d\n"
	   "\t2 rbf : K(X,X')=exp(-g*||X-X'||^2)\n"
	   "-g gamma : coefficent for polynomial and rbf kernels (default 1)\n"
	   "-d degree of polynomial kernel (default 2)\n"
     "-q log output (default open log)\n"
	   "-b c0 coefficient for polynomial kernel (default 0)\n");
}

// TESTING here
void
testing (Machine * svm, Exampler test, double* label_pred, double* score_pred)
{
  ex = test;
  double err = 0;
  if (hasOutput)
  {
    std::cout << "\n--> Testing on " << test.nb_ex -
      nb_train << "ex" << std::endl;
  }
  for (int i = nb_train; i < test.nb_ex; i++)
  {
    int ypred = svm->predict_with_score (i, score_pred + i - nb_train);	// call the predict function
    label_pred[i - nb_train] = (double)ypred;
    if (ypred != test.data[i].cls)
  	err++;
  }
  
  if(hasOutput)
  {
    std::cout << "Test Error:" << 100 * (err/(test.nb_ex - nb_train)) << "%" << std::endl;
  }
}


Machine *
load_model_mex (const mxArray *matlab_struct, int cs)
{
  if(hasOutput)
    std::cout << "\nLoading Model " << std::endl;
  Exampler model;

  int i, num_of_fields;
  mxArray **rhs;

  num_of_fields = mxGetNumberOfFields(matlab_struct);
  if(num_of_fields != NUM_OF_RETURN_FIELD) 
  {
    mexPrintf("number of return field is not correct\n");
    return NULL;
  }
  rhs = (mxArray **) mxMalloc(sizeof(mxArray *)*num_of_fields);

  for(i=0;i<num_of_fields;i++)
    rhs[i] = mxGetFieldByNumber(matlab_struct, 0, i);

  model.read_problem_dense(rhs[0], rhs[1]);

  Machine *svm = create_larank ();

  svm->kfunc = kernel;
  svm->cache = cs;
  svm->add_outputs (model);

  if(hasOutput)
  {
    switch (kernel_type)
    {
      case 0:
        std::cout << "Linear Kernel" << std::endl;
        break;
      case 1:
        std::cout << "Polynomial Kernel with g=" << kgamma << " ,d=" << degree <<" ,c0=" << coef0 << std::endl;
        break;
      case 2:
        std::cout << "RBF Kernel with g=" << kgamma << std::endl;
        break;
      default:
        std::cout << "Linear Kernel" << std::endl;
    }
  }
  return svm;
}

static void fake_answer(int nlhs, mxArray *plhs[])
{
  int i;
  for(i=0;i<nlhs;i++)
    plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[] )
{
  Exampler test;
  Exampler tmp;
  int i;
  kernel_type = 0;
  kgamma = 1.;
  coef0 = 0;
  degree = 1.;
  hasOutput = true;

  if(nlhs != 2 || nrhs > 6 || nrhs < 5)
  {
    exit_with_help();
    fake_answer(nlhs, plhs);
    return;
  }

  if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3])) {
    mexPrintf("Error: label vector and instance matrix must be double\n");
    fake_answer(nlhs, plhs);
    return;
  }

  if(mxIsStruct(prhs[4]))
  {
    const char *error_msg;

    // parse options
    if(nrhs==6)
    {
      int i, argc = 1;
      char cmd[CMD_LEN], *argv[CMD_LEN/2];

      // put options in argv[]
      mxGetString(prhs[5], cmd,  mxGetN(prhs[5]) + 1);
      if((argv[argc] = strtok(cmd, " ")) != NULL)
        while((argv[++argc] = strtok(NULL, " ")) != NULL)
          ;
      for (i = 1; i < argc; i++)
      {
        if (argv[i][0] != '-')
          break;
        ++i;
        switch (argv[i - 1][1])
        {
        case 't':
          kernel_type = atoi (argv[i]);
          break;
        case 'b':
          coef0 = atof (argv[i]);
          break;
        case 'd':
          degree = atof (argv[i]);
          break;
        case 'g':
          kgamma = atof (argv[i]);
          break;
        case 'q':
          hasOutput = false;
          break;
        default:
          mexPrintf("Unknown option: -%c\n", argv[i-1][1]);
          exit_with_help();
          fake_answer(nlhs, plhs);
          return;
        }
      }
    }

    // Load data  
    if(hasOutput)
      std::cout << "Loading Train Data " << std::endl;
    if(mxIsSparse(prhs[1]))
    {
      test.read_problem_sparse(prhs[0], prhs[1]);
    }
    else
    {
      mexPrintf("Error: training label vector should not be in sparse format\n");
      fake_answer(nlhs, plhs);
      return;
    }
    if(hasOutput)
      std::cout << "\nLoading Test Data" << std::endl;
    if(mxIsSparse(prhs[3]))
    {
      tmp.read_problem_sparse(prhs[2], prhs[3]);
    }
    else
    {
      mexPrintf("Error: testing label vector should not be in sparse format\n");
      fake_answer(nlhs, plhs);
      return;
    }

    nb_train = test.nb_ex;
    for (int ex = 0; ex < tmp.nb_ex; ex++)  // Aggregate train and test data in a big sparse kernel matrix
      test.data.push_back (tmp.data[ex]);
    test.nb_ex += tmp.nb_ex;
    test.max_index = jmax (test.max_index, tmp.max_index);
    test.nb_labels = jmax (test.nb_labels, tmp.nb_labels);

    // load the model
    Machine *svm = load_model_mex (prhs[4], test.nb_labels);
    
    // test the model
    plhs[0] = mxCreateDoubleMatrix(tmp.nb_ex, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(tmp.nb_ex, 1, mxREAL);
    double* label_pred = mxGetPr(plhs[0]);
    double* score_pred = mxGetPr(plhs[1]);
    testing (svm, test, label_pred, score_pred);
    // destroy model
    svm->destroy ();
  }
  else
  {
    mexPrintf("model file should be a struct array\n");
    fake_answer(nlhs, plhs);
  }

}
