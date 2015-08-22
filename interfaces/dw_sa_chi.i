%module dw_sa_chi

%{
#include "annealer.h"
#include "ch_annealer.h"
%}

%include "std_vector.i"


namespace std {
    %template(IntVector)  vector < int >;
    %template(DoubleVector) vector < double >;
    %template(IntArray) vector< vector < int > >;
    %template(DoubleArray) vector< vector < double > >;
}

%include "annealer.h"
%include "ch_annealer.h"

