//=======================================================================
/** @file OnsetDetectionFunction.cpp
 *  @brief A class for calculating onset detection functions
 *  @author Adam Stark
 *  @copyright Copyright (C) 2008-2014  Queen Mary University of London
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
//=======================================================================

#include "OnsetDetectionFunction.h"

#include <math.h>

//=======================================================================
OnsetDetectionFunction::OnsetDetectionFunction(int hopSize_, int frameSize_)
    : OnsetDetectionFunction(hopSize_, frameSize_, ComplexSpectralDifferenceHWR,
                             HanningWindow) {}

//=======================================================================
OnsetDetectionFunction::OnsetDetectionFunction(int hopSize_, int frameSize_,
                                               int onsetDetectionFunctionType_,
                                               WindowType windowType_)
    : onsetDetectionFunctionType(ComplexSpectralDifferenceHWR) {
  // indicate that we have not initialised yet
  initialised = false;

  spectral_analysis_.Init(frameSize_, windowType_);

  // initialise with arguments to constructor
  initialise(hopSize_, frameSize_, onsetDetectionFunctionType_);
}

//=======================================================================
OnsetDetectionFunction::~OnsetDetectionFunction() {}

//=======================================================================
void OnsetDetectionFunction::initialise(int hopSize_, int frameSize_) {
  // use the already initialised onset detection function and window type and
  // pass the new frame and hop size to the main initialisation function
  initialise(hopSize_, frameSize_, onsetDetectionFunctionType);
}

//=======================================================================
void OnsetDetectionFunction::initialise(int hopSize_, int frameSize_,
                                        int onsetDetectionFunctionType_) {
  hopSize = hopSize_;      // set hopsize
  frameSize = frameSize_;  // set framesize

  onsetDetectionFunctionType =
      onsetDetectionFunctionType_;  // set detection function type

  // initialise buffers
  frame.resize(frameSize);
  window.resize(frameSize);
  magSpec.resize(frameSize);
  prevMagSpec.resize(frameSize);
  phase.resize(frameSize);
  prevPhase.resize(frameSize);
  prevPhase2.resize(frameSize);

  // initialise previous magnitude spectrum to zero
  for (int i = 0; i < frameSize; i++) {
    prevMagSpec[i] = 0.0;
    prevPhase[i] = 0.0;
    prevPhase2[i] = 0.0;
    frame[i] = 0.0;
  }

  prevEnergySum = 0.0;  // initialise previous energy sum value to zero
}

//=======================================================================
void OnsetDetectionFunction::setOnsetDetectionFunctionType(
    int onsetDetectionFunctionType_) {
  onsetDetectionFunctionType =
      onsetDetectionFunctionType_;  // set detection function type
}

//=======================================================================
double OnsetDetectionFunction::calculateOnsetDetectionFunctionSample(
    const std::vector<double>& buffer) {
  double odfSample;

 // shift audio samples back in frame by hop size
    for (int i =  0; i <frameSize - hopSize; ++i) {
      frame[i] = frame[i + hopSize];
    }

    // add new samples to frame from input buffer
    int j = 0;
    for (int i = frameSize - hopSize; i < frameSize; ++i, ++j) {
      frame[i] = buffer[j];
    }

  switch (onsetDetectionFunctionType) {
    case EnergyEnvelope: {
      // calculate energy envelope detection function sample
      odfSample = energyEnvelope();
      break;
    }
    case EnergyDifference: {
      // calculate half-wave rectified energy difference detection function
      // sample
      odfSample = energyDifference();
      break;
    }
    case SpectralDifference: {
      // calculate spectral difference detection function sample
      odfSample = spectralDifference();
      break;
    }
    case SpectralDifferenceHWR: {
      // calculate spectral difference detection function sample (half wave
      // rectified)
      odfSample = spectralDifferenceHWR();
      break;
    }
    case PhaseDeviation: {
      // calculate phase deviation detection function sample (half wave
      // rectified)
      odfSample = phaseDeviation();
      break;
    }
    case ComplexSpectralDifference: {
      // calcualte complex spectral difference detection function sample
      odfSample = complexSpectralDifference();
      break;
    }
    case ComplexSpectralDifferenceHWR: {
      // calcualte complex spectral difference detection function sample
      // (half-wave rectified)
      odfSample = complexSpectralDifferenceHWR();
      break;
    }
    case HighFrequencyContent: {
      // calculate high frequency content detection function sample
      odfSample = highFrequencyContent();
      break;
    }
    case HighFrequencySpectralDifference: {
      // calculate high frequency spectral difference detection function sample
      odfSample = highFrequencySpectralDifference();
      break;
    }
    case HighFrequencySpectralDifferenceHWR: {
      // calculate high frequency spectral difference detection function
      // (half-wave rectified)
      odfSample = highFrequencySpectralDifferenceHWR();
      break;
    }
    default: {
      odfSample = 1.0;
    }
  }

  return odfSample;
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Methods for Detection Functions
////////////////////////////////////

//=======================================================================
double OnsetDetectionFunction::energyEnvelope() {
  double sum;

  sum = 0;  // initialise sum

  // sum the squares of the samples
  for (int i = 0; i < frameSize; i++) {
    sum = sum + (frame[i] * frame[i]);
  }

  return sum;  // return sum
}

//=======================================================================
double OnsetDetectionFunction::energyDifference() {
  double sum;
  double sample;

  sum = 0;  // initialise sum

  // sum the squares of the samples
  for (int i = 0; i < frameSize; i++) {
    sum = sum + (frame[i] * frame[i]);
  }

  sample = sum - prevEnergySum;  // sample is first order difference in energy

  prevEnergySum = sum;  // store energy value for next calculation

  if (sample > 0) {
    return sample;  // return difference
  } else {
    return 0;
  }
}

//=======================================================================
double OnsetDetectionFunction::spectralDifference() {
  double diff;
  double sum;

  // perform the FFT
  spectral_analysis_.ComputeForwardFft(frame, &complex_spec_);

  // compute first (N/2)+1 mag values
  for (int i = 0; i < (frameSize / 2) + 1; i++) {
    magSpec[i] = sqrt(pow(complex_spec_[i].real(), 2) + pow(complex_spec_[i].imag(), 2));
  }
  // mag spec symmetric above (N/2)+1 so copy previous values
  for (int i = (frameSize / 2) + 1; i < frameSize; i++) {
    magSpec[i] = magSpec[frameSize - i];
  }

  sum = 0;  // initialise sum to zero

  for (int i = 0; i < frameSize; i++) {
    // calculate difference
    diff = magSpec[i] - prevMagSpec[i];

    // ensure all difference values are positive
    if (diff < 0) {
      diff = diff * -1;
    }

    // add difference to sum
    sum = sum + diff;

    // store magnitude spectrum bin for next detection function sample
    // calculation
    prevMagSpec[i] = magSpec[i];
  }

  return sum;
}

//=======================================================================
double OnsetDetectionFunction::spectralDifferenceHWR() {
  double diff;
  double sum;

  // perform the FFT
  spectral_analysis_.ComputeForwardFft(frame, &complex_spec_);

  // compute first (N/2)+1 mag values
  for (int i = 0; i < (frameSize / 2) + 1; i++) {
    magSpec[i] = sqrt(pow(complex_spec_[i].real(), 2) + pow(complex_spec_[i].imag(), 2));
  }
  // mag spec symmetric above (N/2)+1 so copy previous values
  for (int i = (frameSize / 2) + 1; i < frameSize; i++) {
    magSpec[i] = magSpec[frameSize - i];
  }

  sum = 0;  // initialise sum to zero

  for (int i = 0; i < frameSize; i++) {
    // calculate difference
    diff = magSpec[i] - prevMagSpec[i];

    // only add up positive differences
    if (diff > 0) {
      // add difference to sum
      sum = sum + diff;
    }

    // store magnitude spectrum bin for next detection function sample
    // calculation
    prevMagSpec[i] = magSpec[i];
  }

  return sum;
}

//=======================================================================
double OnsetDetectionFunction::phaseDeviation() {
  double dev, pdev;
  double sum;

  // perform the FFT
  spectral_analysis_.ComputeForwardFft(frame, &complex_spec_);

  sum = 0;  // initialise sum to zero

  // compute phase values from fft output and sum deviations
  for (int i = 0; i < frameSize; i++) {
    // calculate phase value
    phase[i] = atan2(complex_spec_[i].imag(), complex_spec_[i].real());

    // calculate magnitude value
    magSpec[i] = sqrt(pow(complex_spec_[i].real(), 2) + pow(complex_spec_[i].imag(), 2));

    // if bin is not just a low energy bin then examine phase deviation
    if (magSpec[i] > 0.1) {
      dev = phase[i] - (2 * prevPhase[i]) + prevPhase2[i];  // phase deviation
      pdev = princarg(dev);  // wrap into [-M_PI,M_PI] range

      // make all values positive
      if (pdev < 0) {
        pdev = pdev * -1;
      }

      // add to sum
      sum = sum + pdev;
    }

    // store values for next calculation
    prevPhase2[i] = prevPhase[i];
    prevPhase[i] = phase[i];
  }

  return sum;
}

//=======================================================================
double OnsetDetectionFunction::complexSpectralDifference() {
  double phaseDeviation;
  double sum;
  double csd;

  // perform the FFT
  spectral_analysis_.ComputeForwardFft(frame, &complex_spec_);

  sum = 0;  // initialise sum to zero

  // compute phase values from fft output and sum deviations
  for (int i = 0; i < frameSize; i++) {
    // calculate phase value
    phase[i] = atan2(complex_spec_[i].imag(), complex_spec_[i].real());

    // calculate magnitude value
    magSpec[i] = sqrt(pow(complex_spec_[i].real(), 2) + pow(complex_spec_[i].imag(), 2));

    // phase deviation
    phaseDeviation = phase[i] - (2 * prevPhase[i]) + prevPhase2[i];

    // calculate complex spectral difference for the current spectral bin
    csd = sqrt(pow(magSpec[i], 2) + pow(prevMagSpec[i], 2) -
               2 * magSpec[i] * prevMagSpec[i] * cos(phaseDeviation));

    // add to sum
    sum = sum + csd;

    // store values for next calculation
    prevPhase2[i] = prevPhase[i];
    prevPhase[i] = phase[i];
    prevMagSpec[i] = magSpec[i];
  }

  return sum;
}

//=======================================================================
double OnsetDetectionFunction::complexSpectralDifferenceHWR() {
  double phaseDeviation;
  double sum;
  double magnitudeDifference;
  double csd;

  // perform the FFT
  spectral_analysis_.ComputeForwardFft(frame, &complex_spec_);

  sum = 0;  // initialise sum to zero
  // compute phase values from fft output and sum deviations
  for (int i = 0; i < frameSize; i++) {
    // calculate phase value
    phase[i] = atan2(complex_spec_[i].imag(), complex_spec_[i].real());

    // calculate magnitude value
    magSpec[i] = sqrt(pow(complex_spec_[i].real(), 2) + pow(complex_spec_[i].imag(), 2));

    // phase deviation
    phaseDeviation = phase[i] - (2 * prevPhase[i]) + prevPhase2[i];

    // calculate magnitude difference (real part of Euclidean distance between
    // complex frames)
    magnitudeDifference = magSpec[i] - prevMagSpec[i];

    // if we have a positive change in magnitude, then include in sum, otherwise
    // ignore (half-wave rectification)
    if (magnitudeDifference > 0) {
      // calculate complex spectral difference for the current spectral bin
      csd = sqrt(pow(magSpec[i], 2) + pow(prevMagSpec[i], 2) -
                 2 * magSpec[i] * prevMagSpec[i] * cos(phaseDeviation));

      // add to sum
      sum = sum + csd;
    }

    // store values for next calculation
    prevPhase2[i] = prevPhase[i];
    prevPhase[i] = phase[i];
    prevMagSpec[i] = magSpec[i];
  }

  return sum;
}

//=======================================================================
double OnsetDetectionFunction::highFrequencyContent() {
  double sum;

  // perform the FFT
  spectral_analysis_.ComputeForwardFft(frame, &complex_spec_);

  sum = 0;  // initialise sum to zero

  // compute phase values from fft output and sum deviations
  for (int i = 0; i < frameSize; i++) {
    // calculate magnitude value
    magSpec[i] = sqrt(pow(complex_spec_[i].real(), 2) + pow(complex_spec_[i].imag(), 2));

    sum = sum + (magSpec[i] * ((double)(i + 1)));

    // store values for next calculation
    prevMagSpec[i] = magSpec[i];
  }

  return sum;
}

//=======================================================================
double OnsetDetectionFunction::highFrequencySpectralDifference() {
  double sum;
  double mag_diff;

  // perform the FFT
  spectral_analysis_.ComputeForwardFft(frame, &complex_spec_);

  sum = 0;  // initialise sum to zero

  // compute phase values from fft output and sum deviations
  for (int i = 0; i < frameSize; i++) {
    // calculate magnitude value
    magSpec[i] = sqrt(pow(complex_spec_[i].real(), 2) + pow(complex_spec_[i].imag(), 2));

    // calculate difference
    mag_diff = magSpec[i] - prevMagSpec[i];

    if (mag_diff < 0) {
      mag_diff = -mag_diff;
    }

    sum = sum + (mag_diff * ((double)(i + 1)));

    // store values for next calculation
    prevMagSpec[i] = magSpec[i];
  }

  return sum;
}

//=======================================================================
double OnsetDetectionFunction::highFrequencySpectralDifferenceHWR() {
  double sum;
  double mag_diff;

  // perform the FFT
  spectral_analysis_.ComputeForwardFft(frame, &complex_spec_);

  sum = 0;  // initialise sum to zero

  // compute phase values from fft output and sum deviations
  for (int i = 0; i < frameSize; i++) {
    // calculate magnitude value
    magSpec[i] = sqrt(pow(complex_spec_[i].real(), 2) + pow(complex_spec_[i].imag(), 2));

    // calculate difference
    mag_diff = magSpec[i] - prevMagSpec[i];

    if (mag_diff > 0) {
      sum = sum + (mag_diff * ((double)(i + 1)));
    }

    // store values for next calculation
    prevMagSpec[i] = magSpec[i];
  }

  return sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Other Handy Methods
/////////////////////////////////////////////

//=======================================================================
double OnsetDetectionFunction::princarg(double phaseVal) {
  // if phase value is less than or equal to -M_PI then add 2*M_PI
  while (phaseVal <= (-M_PI)) {
    phaseVal = phaseVal + (2 * M_PI);
  }

  // if phase value is larger than M_PI, then subtract 2*M_PI
  while (phaseVal > M_PI) {
    phaseVal = phaseVal - (2 * M_PI);
  }

  return phaseVal;
}
