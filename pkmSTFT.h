/*
 *  pkmSTFT.h
 *
 *  STFT implementation making use of Apple's Accelerate Framework (pkmFFT)
 *
 *  Created by Parag K. Mital - http://pkmital.com 
 *  Contact: parag@pkmital.com
 *
 *  Copyright 2011 Parag K. Mital. All rights reserved.

 
 LICENSE:
 
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
 *  fft.ISTFT(sample_data, buffer_size, magnitude_matrix, phase_matrix);
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

	pkmSTFT(int size)
	{
		fftSize = size;
		numFFTs = 0;
		fftBins = fftSize/2;
		hopSize = fftSize/4;
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
			printf("Padding %d sample buffer with %d samples\n", bufSize, padding);
			padBufferSize = bufSize + padding;
			padBuf = (float *)malloc(sizeof(float)*padBufferSize);
			// set padding to 0
			//memset(&(padBuf[bufSize]), 0, sizeof(float)*padding);
			vDSP_vclr(padBuf + bufSize, 1, padding);
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
			printf("Padding %d sample buffer with %d samples\n", bufSize, padding);
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