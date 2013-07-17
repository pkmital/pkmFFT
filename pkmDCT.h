/*
 *  pkmDCT.h
 *
 *  DCT wraper for Apple's Accelerate Framework
 *
 *  Created by Parag K. Mital - http://pkmital.com 
 *  Contact: parag@pkmital.com
 *
 
 Copyright (C) 2011 Parag K. Mital
 
 The Software is and remains the property of Parag K Mital
 ("pkmital") The Licensee will ensure that the Copyright Notice set
 out above appears prominently wherever the Software is used.
 
 The Software is distributed under this Licence: 
 
 - on a non-exclusive basis, 
 
 - solely for non-commercial use in the hope that it will be useful, 
 
 - "AS-IS" and in order for the benefit of its educational and research
 purposes, pkmital makes clear that no condition is made or to be
 implied, nor is any representation or warranty given or to be
 implied, as to (i) the quality, accuracy or reliability of the
 Software; (ii) the suitability of the Software for any particular
 use or for use under any specific conditions; and (iii) whether use
 of the Software will infringe third-party rights.
 
 pkmital disclaims: 
 
 - all responsibility for the use which is made of the Software; and
 
 - any liability for the outcomes arising from using the Software.
 
 The Licensee may make public, results or data obtained from, dependent
 on or arising out of the use of the Software provided that any such
 publication includes a prominent statement identifying the Software as
 the source of the results or the data, including the Copyright Notice
 and stating that the Software has been made available for use by the
 Licensee under licence from pkmital and the Licensee provides a copy of
 any such publication to pkmital.
 
 The Licensee agrees to indemnify pkmital and hold them
 harmless from and against any and all claims, damages and liabilities
 asserted by third parties (including claims for negligence) which
 arise directly or indirectly from the use of the Software or any
 derivative of it or the sale of any products based on the
 Software. The Licensee undertakes to make no liability claim against
 any employee, student, agent or appointee of pkmital, in connection 
 with this Licence or the Software.
 
 
 No part of the Software may be reproduced, modified, transmitted or
 transferred in any form or by any means, electronic or mechanical,
 without the express permission of pkmital. pkmital's permission is not
 required if the said reproduction, modification, transmission or
 transference is done without financial return, the conditions of this
 Licence are imposed upon the receiver of the product, and all original
 and amended source code is included in any transmitted product. You
 may be held legally responsible for any copyright infringement that is
 caused or encouraged by your failure to abide by these terms and
 conditions.
 
 You are not permitted under this Licence to use this Software
 commercially. Use for which any financial return is received shall be
 defined as commercial use, and includes (1) integration of all or part
 of the source code or the Software into a product for sale or license
 by or on behalf of Licensee to third parties or (2) use of the
 Software or any derivative of it for research with the final aim of
 developing software products for sale or license to a third party or
 (3) use of the Software or any derivative of it for research with the
 final aim of developing non-software products for sale or license to a
 third party, or (4) use of the Software to provide any service to an
 external organisation for which payment is received. If you are
 interested in using the Software commercially, please contact pkmital to
 negotiate a licence. Contact details are: parag@pkmital.com
 
 */
 
 
#ifndef pkmMatrix_pkmDCT_h
#define pkmMatrix_pkmDCT_h

#import <Accelerate/Accelerate.h>
#import <math.h>

class pkmDCT 
{
public:
    pkmDCT()
    {
        bAllocated = false;
    }
    
    ~pkmDCT()
    {
        if (bAllocated) {
            free(dctCorrectionFactors);
            vDSP_destroy_fftsetup(fftSetup);
            free(mirroredData);
            free(complexData->realp);
            free(complexData->imagp);
            free(complexData);
            bAllocated = false;
        }
    }
    
    // zero-pad to power of two...
    void setup(int size = 4096) 
    {     
        fftSize = size;
        fftSizeLog2n = log2f(fftSize);
        dctSizeLog2n = fftSizeLog2n - 1;
        dctSize = 1 << dctSizeLog2n;
        
        fftSetup = vDSP_create_fftsetup( fftSizeLog2n, FFT_RADIX2 );
        
        dctCorrectionFactors = (float*) malloc(sizeof(float) * fftSize);
        
        for (int i = 0; i < dctSize; i++) 
        {
            dctCorrectionFactors[2*i  ] = cosf( ( M_PI * (dctSize - 0.5) * i ) / dctSize ) / sqrtf(dctSize * 8.0);
            dctCorrectionFactors[2*i+1] = sinf( ( M_PI * (dctSize - 0.5) * i ) / dctSize ) / sqrtf(dctSize * 8.0);
        }
        dctCorrectionFactors[0] = dctCorrectionFactors[0] / sqrtf(2.0);
        dctCorrectionFactors[1] = dctCorrectionFactors[1] / sqrtf(2.0); 
        
        mirroredData = (float *)malloc(sizeof(float) * fftSize);
        complexData = (DSPSplitComplex*)calloc(1, sizeof(DSPSplitComplex));
        complexData->realp = (float *)malloc(sizeof(float) * dctSize);
        complexData->imagp = (float *)malloc(sizeof(float) * dctSize);
        
        bAllocated = true;
    }
    
    // result is half the size of the input
    void dctII_1D(float *input, float *result, int numCoefficients = -1) 
    {   
        if (!bAllocated) {
            cerr << "[ERROR]::pkmDCT::dctII_1D(...):: Not allocated! Call setup(int size); first!" << endl;
            return;
        }
        
        for (int i = 0; i < dctSize; i++) {
            mirroredData[i] = input[dctSize-i-1];
        }
        for (int i = dctSize; i < fftSize; i++) {
            mirroredData[i] = input[i-dctSize];
        }
        
        vDSP_ctoz((DSPComplex*) mirroredData, (vDSP_Stride) 2, complexData, (vDSP_Stride) 1, dctSize);
        vDSP_fft_zrip( fftSetup, complexData, (vDSP_Stride) 1, fftSizeLog2n, kFFTDirection_Forward);
        vDSP_ztoc(complexData, (vDSP_Stride) 1, (DSPComplex *) mirroredData, (vDSP_Stride) 2, dctSize);
        vDSP_vmul(mirroredData, (vDSP_Stride) 1, dctCorrectionFactors, (vDSP_Stride) 1, mirroredData, (vDSP_Stride) 1, fftSize);
        
        if(numCoefficients == -1)
        {
            for (int i = 0; i < dctSize; i++) {
                result[i] = mirroredData[2*i] - mirroredData[2*i+1];
            }
        }
        else
        {
            for (int i = 0; i < numCoefficients; i++) {
                result[i] = mirroredData[2*i] - mirroredData[2*i+1];
            }
        }
    }
    
private:
    bool bAllocated;
    
    FFTSetup fftSetup;
    float* dctCorrectionFactors;
    int fftSize, fftSizeLog2n, dctSize, dctSizeLog2n;
    
    float *mirroredData;
    DSPSplitComplex* complexData;
};


#endif
