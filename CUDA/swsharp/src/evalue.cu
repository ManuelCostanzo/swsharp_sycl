/*
This code represents a SYCL-compatible, DPC++-based version of SW#.
Copyright (C) 2022 Manuel Costanzo, contributor Enzo Rucci.

swsharp - CUDA parallelized Smith Waterman with applying Hirschberg's and
Ukkonen's algorithm and dynamic cell pruning.
Copyright (C) 2013 Matija Korpar, contributor Mile Šikić

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Contact SW# author by mkorpar@gmail.com.

Contact SW#-SYCL authors by mcostanzo@lidi.info.unlp.edu.ar, erucci@lidi.info.unlp.edu.ar
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "chain.h"
#include "constants.h"
#include "cuda_utils.h"
#include "error.h"
#include "scorer.h"

#include "evalue.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SCORER_CONSTANTS_LEN (sizeof(scorerConstants) / sizeof(ScorerConstants))

struct EValueParams
{
    double lambda;
    double K;
    double logK;
    double H;
    double a;
    double C;
    double alpha;
    double sigma;
    double b;
    double beta;
    double tau;
    double G;
    double aUn;
    double alphaUn;
    long long length;
    int isDna;
};

typedef struct ScorerConstants
{
    const char *matrix;
    int gapOpen;
    int gapExtend;
    double lambda;
    double K;
    double H;
    double a;
    double C;
    double alpha;
    double sigma;
    int isDna;
} ScorerConstants;

// lambda, k, H, a, C, Alpha, Sigma
static ScorerConstants scorerConstants[] = {
    {"BLOSUM_62", -1, -1, 0.3176, 0.134, 0.4012, 0.7916, 0.623757, 4.964660, 4.964660, 0},
    {"BLOSUM_62", 11, 2, 0.297, 0.082, 0.27, 1.1, 0.641766, 12.673800, 12.757600, 0},
    {"BLOSUM_62", 10, 2, 0.291, 0.075, 0.23, 1.3, 0.649362, 16.474000, 16.602600, 0},
    {"BLOSUM_62", 9, 2, 0.279, 0.058, 0.19, 1.5, 0.659245, 22.751900, 22.950000, 0},
    {"BLOSUM_62", 8, 2, 0.264, 0.045, 0.15, 1.8, 0.672692, 35.483800, 35.821300, 0},
    {"BLOSUM_62", 7, 2, 0.239, 0.027, 0.10, 2.5, 0.702056, 61.238300, 61.886000, 0},
    {"BLOSUM_62", 6, 2, 0.201, 0.012, 0.061, 3.3, 0.740802, 140.417000, 141.882000, 0},
    {"BLOSUM_62", 13, 1, 0.292, 0.071, 0.23, 1.2, 0.647715, 19.506300, 19.893100, 0},
    {"BLOSUM_62", 12, 1, 0.283, 0.059, 0.19, 1.5, 0.656391, 27.856200, 28.469900, 0},
    {"BLOSUM_62", 11, 1, 0.267, 0.041, 0.14, 1.9, 0.669720, 42.602800, 43.636200, 0},
    {"BLOSUM_62", 10, 1, 0.243, 0.024, 0.10, 2.5, 0.693267, 83.178700, 85.065600, 0},
    {"BLOSUM_62", 9, 1, 0.206, 0.010, 0.052, 4.0, 0.731887, 210.333000, 214.842000, 0},
    {"EDNA_FULL", 10, 6, 0.163, 0.068, 0.16, 0, 0, 0, 0, 1},
    {"EDNA_FULL", 8, 6, 0.146, 0.039, 0.11, 0, 0, 0, 0, 1}};

#ifdef __CUDACC__
static __constant__ int length_;
static __constant__ int queryLen_;

static __constant__ double paramsLength_;
static __constant__ double paramsLambda_;
static __constant__ double paramsK_;
static __constant__ double paramsLogK_;
static __constant__ double paramsA_;
static __constant__ double paramsB_;
static __constant__ double paramsAlpha_;
static __constant__ double paramsBeta_;
static __constant__ double paramsSigma_;
static __constant__ double paramsTau_;
#endif // __CUDACC__

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

static void eValuesCpu(double *values, int *scores, Chain *query,
                       Chain **database, int databaseLen, EValueParams *eValueParams);

#ifdef __CUDACC__
static void eValuesGpu(double *values, int *scores, Chain *query,
                       Chain **database, int databaseLen, int *cards, int cardsLen,
                       EValueParams *eValueParams);
#endif // __CUDACC__

static double calculateEValueProt(int score, int queryLen, int targetLen,
                                  EValueParams *params);

static double calculateEValueDna(int score, int queryLen, int targetLen,
                                 EValueParams *params);

#ifdef _WIN32
double erf(double x);
#endif

#ifdef __CUDACC__
// With visual c++ compiler and prototypes declared cuda global memory variables
// do not work. No questions asked.
#ifndef _WIN32
__global__ static void kernelProt(double *values, int2 *data);

__global__ static void kernelDna(double *values, int2 *data);
#endif
#endif // __CUDACC__

//******************************************************************************

//******************************************************************************
// PUBLIC

extern EValueParams *createEValueParams(long long length, Scorer *scorer)
{

    const char *matrix = scorerGetName(scorer);
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int index = -1;
    int indexUn = -1;

    for (int i = 0; i < (int)SCORER_CONSTANTS_LEN; ++i)
    {

        ScorerConstants *entry = &(scorerConstants[i]);

        int sameName = strcmp(entry->matrix, matrix) == 0;

        if (sameName && indexUn == -1)
        {
            indexUn = i;
        }

        if (sameName && entry->gapOpen == gapOpen && entry->gapExtend == gapExtend)
        {
            index = i;
            break;
        }
    }

    // ignore miss of default matrix parameters
    if (indexUn == -1 && index != -1)
    {
        indexUn = index;
    }

    if (index == -1 || indexUn == -1)
    {

        if (indexUn == -1)
        {
            index = 0;
            indexUn = 0;
        }
        else
        {
            index = indexUn;
        }

        ScorerConstants *entry = &(scorerConstants[indexUn]);

        WARNING(1, "no e-value params found, using %s %d %d", entry->matrix,
                entry->gapOpen, entry->gapExtend);
    }

    double alphaUn = scorerConstants[indexUn].alpha;
    double aUn = scorerConstants[indexUn].a;
    double G = gapOpen + gapExtend;

    EValueParams *params = (EValueParams *)malloc(sizeof(struct EValueParams));

    params->G = G;
    params->aUn = aUn;
    params->alphaUn = alphaUn;
    params->lambda = scorerConstants[index].lambda;
    params->K = scorerConstants[index].K;
    params->logK = log(params->K);
    params->H = scorerConstants[index].H;
    params->a = scorerConstants[index].a;
    params->C = scorerConstants[index].C;
    params->alpha = scorerConstants[index].alpha;
    params->sigma = scorerConstants[index].sigma;
    params->b = 2.0 * G * (params->aUn - params->a);
    params->beta = 2.0 * G * (params->alphaUn - params->alpha);
    params->tau = 2.0 * G * (params->alphaUn - params->sigma);
    params->length = length;
    params->isDna = scorerConstants[index].isDna;

    printf("Using: lambda = %.3lf, K = %.3lf, H = %.3lf\n",
           params->lambda, params->K, params->H);

    return params;
}

extern void deleteEValueParams(EValueParams *eValueParams)
{
    free(eValueParams);
    eValueParams = NULL;
}

extern void eValues(double *values, int *scores, Chain *query,
                    Chain **database, int databaseLen, int *cards, int cardsLen,
                    EValueParams *eValueParams)
{

#ifdef __CUDACC__
    if (cardsLen == 0)
    {
#endif // __CUDACC__
        eValuesCpu(values, scores, query, database, databaseLen, eValueParams);
#ifdef __CUDACC__
    }
    else
    {
        eValuesGpu(values, scores, query, database, databaseLen, cards,
                   cardsLen, eValueParams);
    }
#endif // __CUDACC__
}

//******************************************************************************

//******************************************************************************
// PRIVATE

static void eValuesCpu(double *values, int *scores, Chain *query,
                       Chain **database, int databaseLen, EValueParams *eValueParams)
{

    double (*function)(int, int, int, EValueParams *);

    if (eValueParams->isDna)
    {
        function = calculateEValueDna;
    }
    else
    {
        function = calculateEValueProt;
    }

    int queryLen = chainGetLength(query);

    for (int i = 0; i < databaseLen; ++i)
    {

        int score = scores[i];
        int targetLen = chainGetLength(database[i]);

        if (score == NO_SCORE)
        {
            values[i] = INFINITY;
            continue;
        }

        values[i] = function(score, queryLen, targetLen, eValueParams);
    }
}

#ifdef __CUDACC__
static void eValuesGpu(double *values, int *scores, Chain *query,
                       Chain **database, int databaseLen, int *cards, int cardsLen,
                       EValueParams *params)
{

    int threads;
    int blocks;

    maxWorkGroups(cards[0], 120, 128, 0, &blocks, &threads);

    // init cpu
    size_t dataSize = databaseLen * sizeof(int2);
    int2 *dataCpu = (int2 *)malloc(dataSize);
    for (int i = 0; i < databaseLen; ++i)
    {
        dataCpu[i].x = scores[i];
        dataCpu[i].y = chainGetLength(database[i]);
    }

    // init global memory
    size_t valuesSize = databaseLen * sizeof(double);
    double *valuesGpu;
    CUDA_SAFE_CALL(cudaMalloc(&valuesGpu, valuesSize));

    int2 *dataGpu;
    CUDA_SAFE_CALL(cudaMalloc(&dataGpu, dataSize));
    CUDA_SAFE_CALL(cudaMemcpy(dataGpu, dataCpu, dataSize, TO_GPU));

    // init constants
    int queryLen = chainGetLength(query);
    double length = params->length;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(length_, &databaseLen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(queryLen_, &queryLen, sizeof(int)));

    if (params->isDna)
    {

        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsLength_, &length, sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsLambda_, &(params->lambda), sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsLogK_, &(params->logK), sizeof(double)));

        kernelDna<<<blocks, threads>>>(valuesGpu, dataGpu);
    }
    else
    {

        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsLength_, &length, sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsLambda_, &(params->lambda), sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsK_, &(params->K), sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsA_, &(params->a), sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsB_, &(params->b), sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsAlpha_, &(params->alpha), sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsBeta_, &(params->beta), sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsSigma_, &(params->sigma), sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(paramsTau_, &(params->tau), sizeof(double)));

        // solve
        kernelProt<<<blocks, threads>>>(valuesGpu, dataGpu);
    }

    // save results
    CUDA_SAFE_CALL(cudaMemcpy(values, valuesGpu, valuesSize, FROM_GPU));

    // clear memory
    CUDA_SAFE_CALL(cudaFree(valuesGpu));
    CUDA_SAFE_CALL(cudaFree(dataGpu));

    free(dataCpu);
}
#endif // __CUDACC__

//------------------------------------------------------------------------------
// GPU MODULES

#ifdef __CUDACC__
__global__ static void kernelProt(double *values, int2 *data)
{

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int m_ = queryLen_;

    double lambda_ = paramsLambda_;
    double k_ = paramsK_;
    double ai_hat_ = paramsA_;
    double bi_hat_ = paramsB_;
    double alphai_hat_ = paramsAlpha_;
    double betai_hat_ = paramsBeta_;
    double sigma_hat_ = paramsSigma_;
    double tau_hat_ = paramsTau_;

    // here we consider symmetric matrix only
    double aj_hat_ = ai_hat_;
    double bj_hat_ = bi_hat_;
    double alphaj_hat_ = alphai_hat_;
    double betaj_hat_ = betai_hat_;

    // this is 1/sqrt(2.0*PI)
    double const_val = 0.39894228040143267793994605993438;

    while (idx < length_)
    {

        int y_ = data[idx].x;
        int n_ = data[idx].y;

        if (y_ == NO_SCORE)
        {
            values[idx] = INFINITY;
        }
        else
        {

            double db_scale_factor = (double)paramsLength_ / (double)n_;

            double m_li_y, vi_y, sqrt_vi_y, m_F, P_m_F;
            double n_lj_y, vj_y, sqrt_vj_y, n_F, P_n_F;
            double c_y, p1, p2;
            double area;

            m_li_y = m_ - (ai_hat_ * y_ + bi_hat_);
            vi_y = MAX(2.0 * alphai_hat_ / lambda_, alphai_hat_ * y_ + betai_hat_);
            sqrt_vi_y = sqrt(vi_y);
            m_F = m_li_y / sqrt_vi_y;
            P_m_F = 0.5 + 0.5 * erf(m_F);
            p1 = m_li_y * P_m_F + sqrt_vi_y * const_val * exp(-0.5 * m_F * m_F);

            n_lj_y = n_ - (aj_hat_ * y_ + bj_hat_);
            vj_y = MAX(2.0 * alphaj_hat_ / lambda_, alphaj_hat_ * y_ + betaj_hat_);
            sqrt_vj_y = sqrt(vj_y);
            n_F = n_lj_y / sqrt_vj_y;
            P_n_F = 0.5 + 0.5 * erf(n_F);
            p2 = n_lj_y * P_n_F + sqrt_vj_y * const_val * exp(-0.5 * n_F * n_F);

            c_y = MAX(2.0 * sigma_hat_ / lambda_, sigma_hat_ * y_ + tau_hat_);
            area = p1 * p2 + c_y * P_m_F * P_n_F;

            values[idx] = area * k_ * exp(-lambda_ * y_) * db_scale_factor;
        }

        idx += gridDim.x * blockDim.x;
    }
}

__global__ static void kernelDna(double *values, int2 *data)
{

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double lambda = paramsLambda_;
    double logK = paramsLogK_;

    while (idx < length_)
    {

        int score = data[idx].x;
        int targetLen = data[idx].y;

        if (score == NO_SCORE)
        {
            values[idx] = INFINITY;
        }
        else
        {
            values[idx] = queryLen_ * targetLen * exp(-lambda * score + logK);
        }

        idx += gridDim.x * blockDim.x;
    }
}
#endif // __CUDACC__

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU MODULES

static double calculateEValueProt(int score, int queryLen, int targetLen,
                                  EValueParams *params)
{

    // code taken from blast
    // pile of statistical crap

    int y_ = score;
    int m_ = queryLen;
    int n_ = targetLen;

    // the pair-wise e-value must be scaled back to db-wise e-value
    double db_scale_factor = (double)params->length / (double)n_;

    double lambda_ = params->lambda;
    double k_ = params->K;
    double ai_hat_ = params->a;
    double bi_hat_ = params->b;
    double alphai_hat_ = params->alpha;
    double betai_hat_ = params->beta;
    double sigma_hat_ = params->sigma;
    double tau_hat_ = params->tau;

    // here we consider symmetric matrix only
    double aj_hat_ = ai_hat_;
    double bj_hat_ = bi_hat_;
    double alphaj_hat_ = alphai_hat_;
    double betaj_hat_ = betai_hat_;

    // this is 1/sqrt(2.0*PI)
    static double const_val = 0.39894228040143267793994605993438;
    double m_li_y, vi_y, sqrt_vi_y, m_F, P_m_F;
    double n_lj_y, vj_y, sqrt_vj_y, n_F, P_n_F;
    double c_y, p1, p2;
    double area;

    m_li_y = m_ - (ai_hat_ * y_ + bi_hat_);
    vi_y = MAX(2.0 * alphai_hat_ / lambda_, alphai_hat_ * y_ + betai_hat_);
    sqrt_vi_y = sqrt(vi_y);
    m_F = m_li_y / sqrt_vi_y;
    P_m_F = 0.5 + 0.5 * erf(m_F);
    p1 = m_li_y * P_m_F + sqrt_vi_y * const_val * exp(-0.5 * m_F * m_F);

    n_lj_y = n_ - (aj_hat_ * y_ + bj_hat_);
    vj_y = MAX(2.0 * alphaj_hat_ / lambda_, alphaj_hat_ * y_ + betaj_hat_);
    sqrt_vj_y = sqrt(vj_y);
    n_F = n_lj_y / sqrt_vj_y;
    P_n_F = 0.5 + 0.5 * erf(n_F);
    p2 = n_lj_y * P_n_F + sqrt_vj_y * const_val * exp(-0.5 * n_F * n_F);

    c_y = MAX(2.0 * sigma_hat_ / lambda_, sigma_hat_ * y_ + tau_hat_);
    area = p1 * p2 + c_y * P_m_F * P_n_F;

    return area * k_ * exp(-lambda_ * y_) * db_scale_factor;
}

static double calculateEValueDna(int score, int queryLen, int targetLen,
                                 EValueParams *params)
{

    double lambda = params->lambda;
    double logK = params->logK;

    return (double)queryLen * targetLen * exp(-lambda * score + logK);
}

#ifdef _WIN32
double erf(double x)
{

    // constants
    double a1 = 0.254829592;
    double a2 = -0.284496736;
    double a3 = 1.421413741;
    double a4 = -1.453152027;
    double a5 = 1.061405429;
    double p = 0.3275911;

    // Save the sign of x
    int sign = x < 0 ? -1 : 1;
    x = fabs(x);

    // A&S formula 7.1.26
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return sign * y;
}
#endif
//------------------------------------------------------------------------------
//******************************************************************************
