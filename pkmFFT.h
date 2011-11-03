/*
 *  pkmFFT.h
 *
 *  Real FFT wraper for Apple's Accelerate Framework
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
 
 *
 *  Additional resources: 
 *      http://developer.apple.com/library/ios/#documentation/Accelerate/Reference/vDSPRef/Reference/reference.html
 *      http://developer.apple.com/library/ios/#documentation/Performance/Conceptual/vDSP_Programming_Guide/SampleCode/SampleCode.html
 *      http://stackoverflow.com/questions/3398753/using-the-apple-fft-and-accelerate-framework
 *      http://stackoverflow.com/questions/1964955/audio-file-fft-in-an-os-x-environment
 *     
 *
 *  This code is a very simple interface for Accelerate's fft/ifft code.
 *  It was built out of hacking Maximilian (Mick Grierson and Chris Kiefer) and
 *  the above mentioned resources for performing a windowed FFT which could
 *  be used underneath of an STFT implementation
 *
 *  Usage:
 *
 *  // be sure to either use malloc or __attribute__ ((aligned (16))
 *  float *sample_data = (float *) malloc (sizeof(float) * 4096);
 *  float *allocated_magnitude_buffer =  (float *) malloc (sizeof(float) * 2048);
 *  float *allocated_phase_buffer =  (float *) malloc (sizeof(float) * 2048);
 *
 *  pkmFFT *fft;
 *  fft = new pkmFFT(4096);
 *  fft.forward(0, sample_data, allocated_magnitude_buffer, allocated_phase_buffer);
 *  fft.inverse(0, sample_data, allocated_magnitude_buffer, allocated_phase_buffer);
 *  delete fft;
 *
 */
#pragma once

#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


class pkmFFT
{
public:

	pkmFFT(int size = 4096)
	{
		fftSize = size;					// sample size
		fftSizeOver2 = fftSize/2;		
		log2n = log2f(fftSize);			// bins
		log2nOver2 = log2n/2;
		
		in_real = (float *) malloc(fftSize * sizeof(float));
		out_real = (float *) malloc(fftSize * sizeof(float));		
		split_data.realp = (float *) malloc(fftSizeOver2 * sizeof(float));
		split_data.imagp = (float *) malloc(fftSizeOver2 * sizeof(float));
		
		windowSize = size;
		window = (float *) malloc(sizeof(float) * windowSize);
		memset(window, 0, sizeof(float) * windowSize);
		vDSP_hann_window(window, windowSize, vDSP_HANN_NORM);
		
		scale = 1.0f/(float)(4.0f*fftSize);
		
		// allocate the fft object once
		fftSetup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
		if (fftSetup == NULL || in_real == NULL || out_real == NULL || 
			split_data.realp == NULL || split_data.imagp == NULL || window == NULL) 
		{
			printf("\nFFT_Setup failed to allocate enough memory.\n");
		}
	}
	~pkmFFT()
	{
		free(in_real);
		free(out_real);
		free(split_data.realp);
		free(split_data.imagp);
		free(window);
		
		vDSP_destroy_fftsetup(fftSetup);
	}
	
	void forward(int start, 
				 float *buffer, 
				 float *magnitude, 
				 float *phase, 
                 bool doWindow = true)
	{	
        if (doWindow) {
            //multiply by window
            vDSP_vmul(buffer, 1, window, 1, in_real, 1, fftSize);
        }
        else {
            cblas_scopy(fftSize, buffer, 1, in_real, 1);
        }
        
        //convert to split complex format with evens in real and odds in imag
        vDSP_ctoz((COMPLEX *) in_real, 2, &split_data, 1, fftSizeOver2);
		
		//calc fft
		vDSP_fft_zrip(fftSetup, &split_data, 1, log2n, FFT_FORWARD);
		
		split_data.imagp[0] = 0.0;
		
		/*
		for (i = 0; i < fftSizeOver2; i++) 
		{
			//compute power 
			float power = split_data.realp[i]*split_data.realp[i] + 
							split_data.imagp[i]*split_data.imagp[i];
			
			//compute magnitude and phase
			magnitude[i] = sqrtf(power);
			phase[i] = atan2f(split_data.imagp[i], split_data.realp[i]);
		}*/
		
		vDSP_ztoc(&split_data, 1, (COMPLEX *) in_real, 2, fftSizeOver2);
		vDSP_polar(in_real, 2, out_real, 2, fftSizeOver2);
		cblas_scopy(fftSizeOver2, out_real, 2, magnitude, 1);
		cblas_scopy(fftSizeOver2, out_real+1, 2, phase, 1);
	}
	
	void inverse(int start, 
				 float *buffer,
				 float *magnitude,
				 float *phase, 
				 bool dowindow = true)
	{
		/*
		float	*real_p = split_data.realp, 
				*imag_p = split_data.imagp;
		for (i = 0; i < fftSizeOver2; i++) {
			*real_p++ = magnitude[i] * cosf(phase[i]);
			*imag_p++ = magnitude[i] * sinf(phase[i]);
		}
		*/
		
		cblas_scopy(fftSizeOver2, magnitude, 1, in_real, 2);
		cblas_scopy(fftSizeOver2, phase, 1, in_real+1, 2);
		vDSP_rect(in_real, 2, out_real, 2, fftSizeOver2);
		
		//convert to split complex format with evens in real and odds in imag
		vDSP_ctoz((COMPLEX *) out_real, 2, &split_data, 1, fftSizeOver2);
		
		vDSP_fft_zrip(fftSetup, &split_data, 1, log2n, FFT_INVERSE);
		vDSP_ztoc(&split_data, 1, (COMPLEX*) out_real, 2, fftSizeOver2);
		
		vDSP_vsmul(out_real, 1, &scale, out_real, 1, fftSize);
		
		// multiply by window w/ overlap-add
		if (dowindow) {
			float *p = buffer + start;
			for (i = 0; i < fftSize; i++) {
				*p++ += out_real[i] * window[i];
			}
		}
        else {
            cblas_scopy(fftSize, out_real, 1, buffer+start, 1);
        }

	}
	
	
	int					fftSize, 
						fftSizeOver2,
						log2n,
						log2nOver2,
						windowSize,
						i;	
	
private:
	
				
	
	float				*in_real, 
						*out_real,
						*window;
	
	float				scale;
	
    FFTSetup			fftSetup;
    COMPLEX_SPLIT		split_data;
	
	
};
