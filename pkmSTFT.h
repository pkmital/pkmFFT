/*
 *  pkmSTFT.h
 *
 *  STFT implementation making use of Apple's Accelerate Framework (pkmFFT)
 *
 *  Created by Parag K. Mital - http://pkmital.com 
 *  Contact: parag@pkmital.com
 *
 *  Copyright 2011 Parag K. Mital. All rights reserved.
 
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
 
 *
 *
 *  Usage:
 *
 *  // be sure to either use malloc or __attribute__ ((aligned (16))
 *  int buffer_size = 4096;
 *  float *sample_data = (float *) malloc (sizeof(float) * buffer_size);
 *  pkm::Mat magnitude_matrix, phase_matrix;
 *
 *  pkmSTFT *stft;
 *  stft = new pkmSTFT(512);
 *  stft.STFT(sample_data, buffer_size, magnitude_matrix, phase_matrix);
 *  stft.ISTFT(sample_data, buffer_size, magnitude_matrix, phase_matrix);
 *  delete stft;
 *
 */
#pragma once

#include <Accelerate/Accelerate.h>
#include "pkmFFT.h"
#include "pkmMatrix.h"

class pkmSTFT
{
public:

	pkmSTFT(int size, int hop = 0)
	{
		fftSize = size;
		numFFTs = 0;
		fftBins = fftSize/2;
        if (hop == 0) {
            hopSize = fftSize/4;
        }
        else
            hopSize = hop;
		windowSize = fftSize;
		bufferSize = 0;
		
		initializeFFTParameters(fftSize, windowSize, hopSize);
	}
	~pkmSTFT()
	{
		free(FFT);
	}
	
	void initializeFFTParameters(int _fftSize, int _windowSize, int _hopSize)
	{
		fftSize = _fftSize;
		hopSize = _hopSize;
		windowSize = _windowSize;
		
		// fft constructor
		FFT = new pkmFFT(fftSize);
		
		numWindows = fftSize / hopSize + 1;
	}
	
	int getNumWindows(int bufSize)
	{
		int padBufferSize;
		int padding = ceilf((float)bufSize/(float)fftSize) * fftSize - bufSize;
		if (padding) {
			padBufferSize = bufSize + padding;
		}
		else {
			padBufferSize = bufSize;
		}
		int numWindows = (padBufferSize - fftSize)/hopSize + 1;
		return numWindows;
	}
		
	
	void STFT(float *buf, int bufSize, pkm::Mat &M_magnitudes, pkm::Mat &M_phases)
	{	
		// pad input buffer
		int padding = ceilf((float)bufSize/(float)fftSize) * fftSize - bufSize;
		int shift = padding / 2;
		float *padBuf;
		if (padding) {
			//printf("Padding %d sample buffer with %d samples\n", bufSize, padding);
			padBufferSize = bufSize + padding;
			padBuf = (float *)malloc(sizeof(float)*padBufferSize);
			// set padding to 0
			//memset(&(padBuf[bufSize]), 0, sizeof(float)*padding);
			vDSP_vclr(padBuf, 1, shift);
			vDSP_vclr(padBuf + bufSize + shift, 1, shift);
			// copy original buffer into padded one
			//memcpy(padBuf, buf, sizeof(float)*bufSize);	
		
			cblas_scopy(bufSize, buf, 1, padBuf + shift, 1);
		}
		else {
			padBuf = buf;
			padBufferSize = bufSize;
		}
		
		// create output fft matrix
		numWindows = (padBufferSize - fftSize)/hopSize + 1;
		
		if (M_magnitudes.rows != numWindows && M_magnitudes.cols != fftBins) {
			M_magnitudes.reset(numWindows, fftBins, true);
			M_phases.reset(numWindows, fftBins, true);
		}
		
		// stft
		for (int i = 0; i < numWindows; i++) {
			
			// get current col of freq mat
			float *magnitudes = M_magnitudes.row(i);
			float *phases = M_phases.row(i);
			float *buffer = padBuf + i*hopSize;
			
			FFT->forward(0, buffer, magnitudes, phases);	
			
			
		}
		// release padded buffer
		if (padding) {
			free(padBuf);
		}
	}
	
	int getBins()
	{
		return fftBins;
	}
	
	int getWindows()
	{
		return numWindows;
	}
	
	
	void ISTFT(float *buf, int bufSize, pkm::Mat &M_magnitudes, pkm::Mat &M_phases)
	{
		int padding = ceilf((float)bufSize/(float)fftSize) * fftSize - bufSize;
		int shift = padding / 2;
		float *padBuf;
		if (padding) 
		{
			//printf("Padding %d sample buffer with %d samples\n", bufSize, padding);
			padBufferSize = bufSize + padding;
			padBuf = (float *)malloc(padBufferSize*sizeof(float));
			vDSP_vclr(padBuf, 1, padBufferSize);
		}
		else {
			padBuf = buf;
			padBufferSize = bufSize;
		}
		
		pkm::Mat M_istft(padBufferSize, 1, padBuf, false);
		
		for(int i = 0; i < numWindows; i++)
		{
			float *buffer = padBuf + i*hopSize;
			float *magnitudes = M_magnitudes.row(i);
			float *phases = M_phases.row(i);
			
			FFT->inverse(0, buffer, magnitudes, phases);
		}

		//memcpy(buf, padBuf, sizeof(float)*bufSize);
		cblas_scopy(bufSize, padBuf + shift, 1, buf, 1);
		// release padded buffer
		if (padding) {
			free(padBuf);
		}
	}
	
	pkmFFT				*FFT;
	
private:
	
	
	int				sampleRate,
						numFFTs,
						fftSize,
						fftBins,
						hopSize,
						bufferSize,
						padBufferSize,
						windowSize,
						numWindows;
};