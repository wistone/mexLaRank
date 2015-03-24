// -*- C++ -*-
// Main functions for learning a Multiclass SVM Classifier with LaRank
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
#include "mex.h"

// GLOBAL VARIABLES

// kernel function
int kernel_type;
double degree;
double kgamma;
double coef0;
int mode;
double C;
double verb;
double tau;
long cache;
bool hasOutput;

// Training data
Exampler ex;
int nb_train;
long nb_kernel;

#define CMD_LEN 2048

void print_string_matlab(const char *s) {mexPrintf(s);}

// Types of kernel functions LaRank can deal with
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
     "Usage: [model] = mexLaRankLearn(training_label_vector, training_instance_matrix, model, 'libsvm_options')\n"
	   "options:\n"
	   "-c cost : set the parameter C (default 1)\n"
	   "-e tau : threshold determining tau-violating pairs of coordinates (default 1e-4) \n"
	   "-t kernel function (default 0):\n"
	   "\t0 linear : K(X,X')=X*X'\n"
	   "\t1 polynomial : K(X,X')=(g*X*X'+c0)^d\n"
	   "\t2 rbf : K(X,X')=exp(-g*||X-X'||^2)\n"
	   "-g gamma : coefficient for polynomial and rbf kernels (default 1)\n"
	   "-d degree of polynomial kernel (default 2)\n"
	   "-b c0 coefficient for polynomial kernel (default 0)\n"
	   "-k cache size : in MB (default 64)\n"
	   "-m mode : set the learning mode (default 0)\n"
     "-q log output (default open log)\n"
	   "\t 0: online learning\n"
	   "\t 1: batch learning (stopping criteria: duality gap < C)\n"
	   "-v verbosity degree : display informations every v %% of the training set size (default 10)\n");
}

// TRAINING here
void
training (Machine * svm, Exampler train, int mode, int step)
{
  ex = train;
  int n_it = 1;
  double initime = getTime (), gap = DBL_MAX;

  if (hasOutput)
  {
    std::cout << "\n--> Training on " << train.nb_ex << "ex" << std::endl;
  }
  while (gap > svm->C)		// stopping criteria
  {
    double tr_err = 0;
    int ind = step;
    for (int i = 0; i < nb_train; i++)
    {
      if (svm->add (i, ex.data[i].cls) != ex.data[i].cls)	// call the add function
      tr_err++;
      if (i / ind)
      {
        if (hasOutput)
        {
          std::cout << "Done: " << (int) (((double) i) / ex.nb_ex * 100) << "%, Train error (online): " << (tr_err / ((double) i + 1)) * 100 << "%" << std::endl;
          svm->printStuff (initime, false);
        }
        ind += step;
      }
    }

    if (hasOutput)
    {
      std::cout << "End of iteration " << n_it++ << std::endl;
      std::cout << "Train error (online): " << (tr_err / nb_train) * 100 << "%" << std::endl;
    }
    gap = svm->computeGap ();
    if (hasOutput)
    {
      std::cout << "Duality gap: " << gap << std::endl;
      svm->printStuff (initime, true);
    }
    if (mode == 0)		// skip stopping criteria if online mode
    gap = 0;
  }

  if (hasOutput)
  {
    std::cout << "---- End of training ---- (Computed kernels " << nb_kernel << ")"<< std::endl;
  }
}


void
save_model (Machine * svm, const char *file)
{
  if (hasOutput)
  {
    std::cout << "\n--> Saving Model in \"" << file << " \" " << std::endl;
  }
  std::ofstream ostr (file);
  svm->save_outputs (ostr);
}


// nrhs should be 3
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
  int i, argc = 1;
  char cmd[CMD_LEN];
  char *argv[CMD_LEN/2];
  void (*print_func)(const char *) = print_string_matlab; // default printing to matlab display

  mode = 0;
  C = 1; verb = 10; tau = 0.0001;
  cache = 64;
  kernel_type = 0;
  kgamma = 1.;
  coef0 = 0;
  degree = 2.;
  hasOutput = true;

  if(nrhs <= 1)
    return 1;

  if(nrhs > 2)
  {
    // put options in argv[]
    mxGetString(prhs[2], cmd, mxGetN(prhs[2]) + 1);
    if((argv[argc] = strtok(cmd, " ")) != NULL)
      while((argv[++argc] = strtok(NULL, " ")) != NULL)
        ;
  }

  // parse options
  for (i = 1; i < argc; i++)
  {
    if (argv[i][0] != '-') break;
    ++i;
    switch (argv[i - 1][1])
    {
      case 'c':
        C = atof (argv[i]);
        break;
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
      case 'm':
        mode = atoi (argv[i]);
        break;
      case 'e':
        tau = atof (argv[i]);
        break;
      case 'v':
        verb = atof (argv[i]);
        break;
      case 'k':
        cache = atol (argv[i]);
      case 'q':
        hasOutput = false;
        break;
      default:
        mexPrintf("Unknown option -%c\n", argv[i-1][1]);
        return 1;
    }
  }

  return 0;
}

static void fake_answer(int nlhs, mxArray *plhs[])
{
  int i;
  for(i=0;i<nlhs;i++)
    plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[] )
{

  Exampler train;
  Exampler model;

  if(nlhs > 1)
  {
    exit_with_help();
    fake_answer(nlhs, plhs);
    return;
  }

  // Transform the input Matrix to libsvm format
  if(nrhs > 1 && nrhs < 4)
  {

    if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
    {
      mexPrintf("Error: label vector and instance matrix must be double\n");
      fake_answer(nlhs, plhs);
      return;
    }

    if(mxIsSparse(prhs[0]))
    {
      mexPrintf("Error: label vector should not be in sparse format\n");
      fake_answer(nlhs, plhs);
      return;
    }

    if(parse_command_line(nrhs, prhs, NULL))
    {
      exit_with_help();
      fake_answer(nlhs, plhs);
      return;
    }

    if(mxIsSparse(prhs[1]))
    {
      train.read_problem_sparse(prhs[0], prhs[1]);
    }
    else
    {
      mexPrintf("Error: data matrix should be in sparse format\n");
      fake_answer(nlhs, plhs);
      return;
    }

    Machine *svm = create_larank ();
    int step = (int) ((double) train.nb_ex / (100 / verb));
    svm->tau = tau;
    svm->C = C;
    svm->degree = degree;
    svm->kfunc = kernel;
    svm->cache = (int) (cache / train.nb_labels);
    svm->kernel_type = kernel_type;
    svm->kgamma = kgamma;
    svm->coef0 = coef0;
    svm->mode = mode;

    nb_train = train.nb_ex;
    nb_kernel = 0;

    // train LaRank
    training (svm, train, mode, step);

    int nr_feat = (int)mxGetN(prhs[1]);
    const char *error_msg;
    svm->model_to_matlab_structure(plhs);
    svm->destroy ();
  }
  else
  {
    exit_with_help();
    fake_answer(nlhs, plhs);
    return;
  }
}

