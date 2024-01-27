// The entire point of using a filter to downsample is to antialias with subsample resolution in the output signal
// We need to decide how precise our filter is
class BlipBuf {
    static KERNEL_RESOLUTION = 64;
    static MAX_KERNEL_SIZE = 256;
    /** @type {Object.<object, Float64Array>} */
    static kernels = {};
    /**
     * @param {number} kernelSize
     * @param {number} bufSize
     * @param {boolean} normalize
     * @param {number} channels
     * @param {number} filterRatio
     * @param {boolean} minimumPhase
     */
    constructor(kernelSize, bufSize, normalize, channels, filterRatio, minimumPhase) {
        if (kernelSize > BlipBuf.MAX_KERNEL_SIZE) {
            throw 'kernelSize > BlipBuf.MAX_KERNEL_SIZE';
        }
        if (bufSize < kernelSize * 2) {
            throw 'bufSize needs to be greater than kernelSize * 2';
        }
        if (kernelSize % 4 !== 0) {
            throw 'kernelSize must be multiple of 16';
        }
        // Lanzcos kernel
        let key = kernelSize.toString() + normalize + filterRatio + minimumPhase;
        this.kernel = BlipBuf.kernels[key];
        if (this.kernel === undefined) {
            this.kernel = BlipBuf.genKernel(kernelSize, normalize, filterRatio, minimumPhase);
            console.log(this.kernel);
            BlipBuf.kernels[key] = this.kernel;
        }
        this.kernelSize = kernelSize;
        this.channels = channels;
        this.bufSize = bufSize;

        this.bufPos = 0;
        this.outputL = 0;
        this.outputR = 0;
        this.currentSampleInPos = 0;
        this.currentSampleOutPos = 0;
        this.channelValsL = new Float64Array(channels);
        this.channelValsR = new Float64Array(channels);
        this.channelT = new Float64Array(channels);
        this.channelRealSample = new Float64Array(channels);
        // This is a buffer of differences we are going to write bandlimited impulses to
        this.bufL = new Float64Array(this.bufSize);
        this.bufR = new Float64Array(this.bufSize);
    }

    // Discrete Fourier Transform
    /**
     * @param {number} n
     * @param {Float64Array} realTime
     * @param {Float64Array} imagTime
     * @param {Float64Array} realFreq
     * @param {Float64Array} imagFreq
     */
    static dft(n, realTime, imagTime, realFreq, imagFreq) {
        for (let k = 0; k < n; k++) {
            realFreq[k] = 0;
            imagFreq[k] = 0;
        }

        for (let k = 0; k < n; k++)
            for (let i = 0; i < n; i++) {
                let p = (2.0 * Math.PI * (k * i)) / n;
                let sr = Math.cos(p);
                let si = -Math.sin(p);
                realFreq[k] += (realTime[i] * sr) - (imagTime[i] * si);
                imagFreq[k] += (realTime[i] * si) + (imagTime[i] * sr);
            }
    }

    // Inverse Discrete Fourier Transform
    /**
     * @param {number} n
     * @param {Float64Array} realTime
     * @param {Float64Array} imagTime
     * @param {Float64Array} realFreq
     * @param {Float64Array} imagFreq
     */
    static inverseDft(n, realTime, imagTime, realFreq, imagFreq) {
        for (let k = 0; k < n; k++) {
            realTime[k] = 0;
            imagTime[k] = 0;
        }

        for (let k = 0; k < n; k++) {
            for (let i = 0; i < n; i++) {
                let p = (2.0 * Math.PI * (k * i)) / n;
                let sr = Math.cos(p);
                let si = -Math.sin(p);
                realTime[k] += (realFreq[i] * sr) + (imagFreq[i] * si);
                imagTime[k] += (realFreq[i] * si) - (imagFreq[i] * sr);
            }
            realTime[k] /= n;
            imagTime[k] /= n;
        }
    }

    // Complex Absolute Value
    /**
     * @param {number} x
     * @param {number} y
     */
    static cabs(x, y) {
        return Math.sqrt((x * x) + (y * y));
    }

    // Complex Exponential
    /**
     * @param {number} x
     * @param {number} y
     */
    static cexp(x, y) {
        let expx = Math.E ** x;
        return [expx * Math.cos(y), expx * Math.sin(y)];
    }

    // Compute Real Cepstrum Of Signal
    /**
     * @param {number} n
     * @param {Float64Array} signal
     * @param {Float64Array} realCepstrum
     */
    static realCepstrum(n, signal, realCepstrum) {
        // Compose Complex FFT Input
        let realTime = new Float64Array(n);
        let imagTime = new Float64Array(n);
        let realFreq = new Float64Array(n);
        let imagFreq = new Float64Array(n);

        for (let i = 0; i < n; i++) {
            realTime[i] = signal[i];
            imagTime[i] = 0;
        }

        // Perform DFT
        BlipBuf.dft(n, realTime, imagTime, realFreq, imagFreq);

        // Calculate Log Of Absolute Value
        for (let i = 0; i < n; i++) {
            realFreq[i] = Math.log(BlipBuf.cabs(realFreq[i], imagFreq[i]));
            imagFreq[i] = 0.0;
        }

        // Perform Inverse FFT
        BlipBuf.inverseDft(n, realTime, imagTime, realFreq, imagFreq);

        // Output Real Part Of FFT
        for (let i = 0; i < n; i++)
            realCepstrum[i] = realTime[i];
    }

    // Compute Minimum Phase Reconstruction Of Signal
    /**
    * @param {number} n
    * @param {Float64Array} realCepstrum
    * @param {Float64Array} minimumPhase
    */
    static minimumPhase(n, realCepstrum, minimumPhase) {
        let nd2 = n / 2;

        let realTime = new Float64Array(n);
        let imagTime = new Float64Array(n);
        let realFreq = new Float64Array(n);
        let imagFreq = new Float64Array(n);

        if ((n % 2) == 1) {
            realTime[0] = realCepstrum[0];
            for (let i = 1; i < nd2; i++)
                realTime[i] = 2.0 * realCepstrum[i];
            for (let i = nd2; i < n; i++)
                realTime[i] = 0.0;
        }
        else {
            realTime[0] = realCepstrum[0];
            for (let i = 1; i < nd2; i++)
                realTime[i] = 2.0 * realCepstrum[i];
            realTime[nd2] = realCepstrum[nd2];
            for (let i = nd2 + 1; i < n; i++)
                realTime[i] = 0.0;
        }

        for (let i = 0; i < n; i++)
            imagTime[i] = 0.0;

        BlipBuf.dft(n, realTime, imagTime, realFreq, imagFreq);

        for (let i = 0; i < n; i++) {
            let res = BlipBuf.cexp(realFreq[i], imagFreq[i]);
            realFreq[i] = res[0];
            imagFreq[i] = res[1];
        }

        BlipBuf.inverseDft(n, realTime, imagTime, realFreq, imagFreq);

        for (let i = 0; i < n; i++)
            minimumPhase[i] = realTime[i];
    }

    /**
    * @param {number} kernelSize
    * @param {boolean} normalize
    * @param {number} filterRatio
    * @param {boolean} minimumPhase
    * @returns {Float64Array}
    */
    static genKernel(kernelSize, normalize, filterRatio, minimumPhase) {
        let kernel = new Float64Array(kernelSize * BlipBuf.KERNEL_RESOLUTION);
        kernelSize = kernelSize;
        if ((kernelSize & (kernelSize - 1)) != 0) {
            throw "kernelSize not power of 2:" + kernelSize;
        }
        if (filterRatio <= 0 || filterRatio > Math.PI) {
            throw "invalid filterRatio, outside of (0, pi]";
        }

        // Generate the normalized Lanzcos kernel
        // Derived from Wikipedia https://en.wikipedia.org/wiki/Lanczos_resampling

        for (let i = 0; i < BlipBuf.KERNEL_RESOLUTION; i++) {
            let sum = 0;
            for (let j = 0; j < kernelSize; j++) {
                let x = j - kernelSize / 2;
                // Shift X coordinate right for subsample accuracy
                // We now have the X coordinates for an impulse bandlimited at the sample rate
                x += i / BlipBuf.KERNEL_RESOLUTION;
                // filterRatio determines the highest frequency that will be generated,
                // as a portion of the Nyquist frequency. e.g. at a sample rate of 48000 hz, the Nyquist frequency
                // will be at 24000 hz, so a filterRatio of 0.5 would set the highest frequency
                // generated at 12000 hz.
                x *= filterRatio * Math.PI;
                // Get the sinc, which represents a bandlimited impulse
                let sinc = Math.sin(x) / x;
                // A sinc function's domain is infinte, meaning 
                // convolving a signal with a true sinc function would take an infinite amount of time
                // To avoid creating a filter with infinite latency, we have to decide when to cut off
                // our sinc function. We can window (i.e. multiply) our true sinc function with a
                // horizontally stretched sinc function to create a windowed sinc function of our desired width. 
                let lanzcosWindow = Math.sin(x / kernelSize) / (x / kernelSize);
                // A hole exists in the sinc function at zero, special case it
                if (x == 0) {
                    kernel[i * kernelSize + j] = 1;
                }
                else {
                    // Apply our window here
                    kernel[i * kernelSize + j] = sinc * lanzcosWindow;
                }
                sum += kernel[i * kernelSize + j];

            }
            if (normalize && !minimumPhase) {
                for (let j = 0; j < kernelSize; j++) {
                    kernel[i * kernelSize + j] /= sum;
                }
            }
        }
        if (minimumPhase) {
            let fullSize = kernelSize * BlipBuf.KERNEL_RESOLUTION;

            let index = 0;
            let kMinPhase = new Float64Array(fullSize);
            for (let u = 0; u < kernelSize; u++) {
                for (let v = 0; v < BlipBuf.KERNEL_RESOLUTION; v++) {
                    let val = kernel[u + v * kernelSize];
                    kMinPhase[index++] = val;
                }
            }

            let tmp = new Float64Array(fullSize);
            BlipBuf.realCepstrum(fullSize, kMinPhase, tmp);
            BlipBuf.minimumPhase(fullSize, tmp, kMinPhase);

            index = 0;
            for (let u = 0; u < kernelSize; u++) {
                for (let v = 0; v < BlipBuf.KERNEL_RESOLUTION; v++) {
                    kernel[u + v * kernelSize] = kMinPhase[index++];
                }
            }

            if (normalize) {
                for (let i = 0; i < BlipBuf.KERNEL_RESOLUTION; i++) {
                    let sum = 0;
                    for (let j = 0; j < kernelSize; j++) {
                        sum += kernel[i * kernelSize + j];
                    }
                    for (let j = 0; j < kernelSize; j++) {
                        kernel[i * kernelSize + j] /= sum;
                    }
                }
                console.log("normalizing");
            }
        }
        return kernel;
    }
    reset() {
        // Flush out the difference buffer
        this.bufPos = 0;
        this.outputL = 0;
        this.outputR = 0;
        this.currentSampleInPos = 0;
        this.currentSampleOutPos = 0;
        for (let i = 0; i < this.bufSize; i++) {
            this.bufL[i] = 0;
            this.bufR[i] = 0;
        }
        for (let i = 0; i < this.channels; i++) {
            this.channelValsL[i] = 0;
            this.channelValsR[i] = 0;
            this.channelT[i] = 0;
            this.channelRealSample[i] = 0;
        }
    }
    // Sample is in terms of out samples
    /**
     * @param {number} channel
     * @param {number} t
     * @param {number} valL
     * @param {number} valR
     */
    setValue(channel, t, valL, valR) {
        if (t >= this.channelT[channel]) {
            this.channelT[channel] = t;
        }
        else {
            console.warn(`Channel ${channel}: Tried to set amplitude backward in time from ${this.channelT[channel]} to ${t}`);
        }

        let diffL = valL - this.channelValsL[channel];
        let diffR = valR - this.channelValsR[channel];

        let subsamplePos = (1 - (t % 1)) * (BlipBuf.KERNEL_RESOLUTION - 1) | 0;
        // Add our bandlimited impulse to the difference buffer
        let bufPos = this.bufPos + (t | 0) - this.currentSampleOutPos;
        if (bufPos > this.bufSize) {
            bufPos -= this.bufSize;
            if (bufPos > this.bufSize)
                throw `Overflowed buffer (${bufPos} > ${this.bufSize}) `;
        }
        for (let i = 0; i < this.kernelSize; i++) {
            let kVal = this.kernel[this.kernelSize * subsamplePos + i];
            this.bufL[bufPos] += kVal * diffL;
            this.bufR[bufPos] += kVal * diffR;
            if (++bufPos >= this.bufSize)
                bufPos = 0;
        }

        this.channelValsL[channel] = valL;
        this.channelValsR[channel] = valR;
    }
    /**
     * @param {number} channel
     * @param {number} t
     * @param {number} val
     */
    setValueL(channel, t, val) {
        if (t >= this.channelT[channel]) {
            this.channelT[channel] = t;
        }
        else {
            console.warn(`Channel ${channel}: Tried to set amplitude backward in time from ${this.channelT[channel]} to ${t}`);
        }

        let diff = val - this.channelValsL[channel];

        let subsamplePos = (1 - (t % 1)) * (BlipBuf.KERNEL_RESOLUTION - 1) | 0;
        // Add our bandlimited impulse to the difference buffer
        let bufPos = this.bufPos + (t | 0) - this.currentSampleOutPos;
        if (bufPos > this.bufSize) {
            bufPos -= this.bufSize;
            if (bufPos > this.bufSize)
                throw `Overflowed buffer (${bufPos} > ${this.bufSize}) `;
        }
        let offs = this.kernelSize * subsamplePos;
        let endOffs = offs + this.kernelSize;
        while (offs < endOffs) {
            // Unrolling this really, REALLY helps. 
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
            this.bufL[bufPos++] += this.kernel[offs++] * diff; if (bufPos >= this.bufSize) bufPos = 0;
        }

        this.channelValsL[channel] = val;
    }
    readOutSample() {
        // Integrate the difference buffer
        this.outputL += this.bufL[this.bufPos];
        this.outputR += this.bufR[this.bufPos];
        this.bufL[this.bufPos] = 0;
        this.bufR[this.bufPos] = 0;
        if (++this.bufPos >= this.bufSize)
            this.bufPos = 0;
        this.currentSampleOutPos++;
    }
    readOutSampleL() {
        // Integrate the difference buffer
        this.outputL += this.bufL[this.bufPos];
        this.bufL[this.bufPos] = 0;
        if (++this.bufPos >= this.bufSize)
            this.bufPos = 0;
        this.currentSampleOutPos++;
        return this.outputL;
    }
}

// Based off NAudio's BiQuadFilter, look there for comments
class BiquadFilter {
    // coefficients
    a0 = 0;
    a1 = 0;
    a2 = 0;
    a3 = 0;
    a4 = 0;

    // state
    x1; /** @type {Float64Array} */
    x2; /** @type {Float64Array} */
    y1; /** @type {Float64Array} */
    y2; /** @type {Float64Array} */

    numCascade;

    /**
     * @param {number} order
     */
    constructor(order, aa0 = 0, aa1 = 0, aa2 = 0, b0 = 0, b1 = 0, b2 = 0) {
        this.setCoefficients(aa0, aa1, aa2, b0, b1, b2);

        if ((order % 2) != 0) {
            throw 'Order not divisible by 2';
        }

        this.numCascade = order / 2;

        this.x1 = new Float64Array(this.numCascade);
        this.x2 = new Float64Array(this.numCascade);
        this.y1 = new Float64Array(this.numCascade);
        this.y2 = new Float64Array(this.numCascade);
    }

    resetState() {
        for (let i = 0; i < this.numCascade; i++) {
            this.x1[i] = 0;
            this.x2[i] = 0;
            this.y1[i] = 0;
            this.y2[i] = 0;
        }
    }

    /**
     * @param {number} inSample
     */
    transform(inSample) {
        for (let i = 0; i < this.numCascade; i++) {
            // compute result
            let result = this.a0 * inSample + this.a1 * this.x1[i] + this.a2 * this.x2[i] - this.a3 * this.y1[i] - this.a4 * this.y2[i];
            // if (isNaN(result)) throw "NaN in filter";
            // if (result == Infinity) throw "Infinity in filter";

            // shift x1 to x2, sample to x1 
            this.x2[i] = this.x1[i];
            this.x1[i] = inSample;

            // shift y1 to y2, result to y1 
            this.y2[i] = this.y1[i];
            this.y1[i] = result;

            inSample = result;
        }
        return inSample;
    }

    /**
     * @param {number} aa0
     * @param {number} aa1
     * @param {number} aa2
     * @param {number} b0
     * @param {number} b1
     * @param {number} b2
     */
    setCoefficients(aa0, aa1, aa2, b0, b1, b2) {
        if (isNaN(aa0)) throw "aa0 is NaN";
        if (isNaN(aa1)) throw "aa1 is NaN";
        if (isNaN(aa2)) throw "aa2 is NaN";
        if (isNaN(b0)) throw "b0 is NaN";
        if (isNaN(b1)) throw "b1 is NaN";
        if (isNaN(b2)) throw "b2 is NaN";

        this.a0 = b0 / aa0;
        this.a1 = b1 / aa0;
        this.a2 = b2 / aa0;
        this.a3 = aa1 / aa0;
        this.a4 = aa2 / aa0;
    }

    /**
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} q
     */
    setLowPassFilter(sampleRate, cutoffFrequency, q) {
        let w0 = 2 * Math.PI * cutoffFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let alpha = Math.sin(w0) / (2 * q);

        let b0 = (1 - cosw0) / 2;
        let b1 = 1 - cosw0;
        let b2 = (1 - cosw0) / 2;
        let aa0 = 1 + alpha;
        let aa1 = -2 * cosw0;
        let aa2 = 1 - alpha;
        this.setCoefficients(aa0, aa1, aa2, b0, b1, b2);
    }

    /**
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     * @param {number} dbGain
     */
    setPeakingEq(sampleRate, centreFrequency, q, dbGain) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);
        let a = Math.pow(10, dbGain / 40);

        let b0 = 1 + alpha * a;
        let b1 = -2 * cosw0;
        let b2 = 1 - alpha * a;
        let aa0 = 1 + alpha / a;
        let aa1 = -2 * cosw0;
        let aa2 = 1 - alpha / a;
        this.setCoefficients(aa0, aa1, aa2, b0, b1, b2);
    }

    /**
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} q
     */
    setHighPassFilter(sampleRate, cutoffFrequency, q) {
        let w0 = 2 * Math.PI * cutoffFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let alpha = Math.sin(w0) / (2 * q);

        let b0 = (1 + cosw0) / 2;
        let b1 = -(1 + cosw0);
        let b2 = (1 + cosw0) / 2;
        let aa0 = 1 + alpha;
        let aa1 = -2 * cosw0;
        let aa2 = 1 - alpha;
        this.setCoefficients(aa0, aa1, aa2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} q
     */
    static lowPassFilter(order, sampleRate, cutoffFrequency, q) {
        let filter = new BiquadFilter(order);
        filter.setLowPassFilter(sampleRate, cutoffFrequency, q);
        return filter;
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} q
     */
    static highPassFilter(order, sampleRate, cutoffFrequency, q) {
        let filter = new BiquadFilter(order);
        filter.setHighPassFilter(sampleRate, cutoffFrequency, q);
        return filter;
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     */
    static bandPassFilterConstantSkirtGain(order, sampleRate, centreFrequency, q) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);

        let b0 = sinw0 / 2;
        let b1 = 0;
        let b2 = -sinw0 / 2;
        let a0 = 1 + alpha;
        let a1 = -2 * cosw0;
        let a2 = 1 - alpha;
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     */
    static bandPassFilterConstantPeakGain(order, sampleRate, centreFrequency, q) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);

        let b0 = alpha;
        let b1 = 0;
        let b2 = -alpha;
        let a0 = 1 + alpha;
        let a1 = -2 * cosw0;
        let a2 = 1 - alpha;
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     */
    static notchFilter(order, sampleRate, centreFrequency, q) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);

        let b0 = 1;
        let b1 = -2 * cosw0;
        let b2 = 1;
        let a0 = 1 + alpha;
        let a1 = -2 * cosw0;
        let a2 = 1 - alpha;
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     */
    static allPassFilter(order, sampleRate, centreFrequency, q) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);

        let b0 = 1 - alpha;
        let b1 = -2 * cosw0;
        let b2 = 1 + alpha;
        let a0 = 1 + alpha;
        let a1 = -2 * cosw0;
        let a2 = 1 - alpha;
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     * @param {number} dbGain
     */
    static peakingEQ(order, sampleRate, centreFrequency, q, dbGain) {
        let filter = new BiquadFilter(order,);
        filter.setPeakingEq(sampleRate, centreFrequency, q, dbGain);
        return filter;
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} shelfSlope
     * @param {number} dbGain
     */
    static lowShelf(order, sampleRate, cutoffFrequency, shelfSlope, dbGain) {
        let w0 = 2 * Math.PI * cutoffFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let a = Math.pow(10, dbGain / 40);
        let alpha = sinw0 / 2 * Math.sqrt((a + 1 / a) * (1 / shelfSlope - 1) + 2);
        let temp = 2 * Math.sqrt(a) * alpha;

        let b0 = a * ((a + 1) - (a - 1) * cosw0 + temp);
        let b1 = 2 * a * ((a - 1) - (a + 1) * cosw0);
        let b2 = a * ((a + 1) - (a - 1) * cosw0 - temp);
        let a0 = (a + 1) + (a - 1) * cosw0 + temp;
        let a1 = -2 * ((a - 1) + (a + 1) * cosw0);
        let a2 = (a + 1) + (a - 1) * cosw0 - temp;
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} shelfSlope
     * @param {number} dbGain
     */
    static highShelf(order, sampleRate, cutoffFrequency, shelfSlope, dbGain) {
        let w0 = 2 * Math.PI * cutoffFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let a = Math.pow(10, dbGain / 40);
        let alpha = sinw0 / 2 * Math.sqrt((a + 1 / a) * (1 / shelfSlope - 1) + 2);
        let temp = 2 * Math.sqrt(a) * alpha;

        let b0 = a * ((a + 1) + (a - 1) * cosw0 + temp);
        let b1 = -2 * a * ((a - 1) + (a + 1) * cosw0);
        let b2 = a * ((a + 1) + (a - 1) * cosw0 - temp);
        let a0 = (a + 1) - (a - 1) * cosw0 + temp;
        let a1 = 2 * ((a - 1) - (a + 1) * cosw0);
        let a2 = (a + 1) - (a - 1) * cosw0 - temp;
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }
}


/**
 * @param {number} t
 * @param {number} dt
 */
function polyBlep(t, dt) {
    // 0 <= t < 1
    if (t < dt) {
        t /= dt;
        // 2 * (t - t^2/2 - 0.5)
        return t + t - t * t - 1.0;
    }
    // -1 < t < 0
    else if (t > 1.0 - dt) {
        t = (t - 1.0) / dt;
        // 2 * (t^2/2 + t + 0.5)
        return t * t + t + t + 1.0;
    }
    // 0 otherwise
    else {
        return 0.0;
    }
}

/**
 * @param {number} t
 * @param {number} dt
 */
function polyBlepTest(t, dt) {
    // 0 <= t < 1
    if (t < dt) {
        return 1;
    }
    // -1 < t < 0
    else if (t > 1.0 - dt) {
        return -1;
    }
    // 0 otherwise
    else {
        return 0.0;
    }
}

class SoundgoodizerFilterChannel {
    // Each biquad filter has a slope of 12db/oct so 2 biquads chained gets us 24db/oct
    /** @type {BiquadFilter[]} */ lowFilters = new Array(2);
    /** @type {BiquadFilter[]} */ midFilters = new Array(4);
    /** @type {BiquadFilter[]} */ highFilters = new Array(2);

    outLow = 0;
    outMid = 0;
    outHigh = 0;

    dbPerOct24 = false;

    /**
     * @param {boolean} dbPerOct24
     * @param {number} sampleRate
     * @param {number} lowHz
     * @param {number} highHz
     **/
    constructor(dbPerOct24, sampleRate, lowHz, highHz) {
        this.dbPerOct24 = dbPerOct24;

        // q = 1/sqrt(2) maximally flat "butterworth" filter
        let q = 1 / Math.SQRT2;
        this.lowFilters[0] = BiquadFilter.lowPassFilter(1, sampleRate, lowHz, q);
        this.lowFilters[1] = BiquadFilter.lowPassFilter(1, sampleRate, lowHz, q);

        this.midFilters[0] = BiquadFilter.highPassFilter(1, sampleRate, lowHz, q);
        this.midFilters[1] = BiquadFilter.lowPassFilter(1, sampleRate, highHz, q);
        this.midFilters[2] = BiquadFilter.highPassFilter(1, sampleRate, lowHz, q);
        this.midFilters[3] = BiquadFilter.lowPassFilter(1, sampleRate, highHz, q);

        this.highFilters[0] = BiquadFilter.highPassFilter(1, sampleRate, highHz, q);
        this.highFilters[1] = BiquadFilter.highPassFilter(1, sampleRate, highHz, q);
    }

    /**
     * @param {boolean} dbPerOct24
     * @param {number} sampleRate
     * @param {number} lowHz
     * @param {number} highHz
     * */
    changeFilterParams(dbPerOct24, sampleRate, lowHz, highHz) {
        this.dbPerOct24 = dbPerOct24;

        let q = 1 / Math.SQRT2;
        this.lowFilters[0].setLowPassFilter(sampleRate, lowHz, q);
        this.lowFilters[1].setLowPassFilter(sampleRate, lowHz, q);

        this.midFilters[0].setHighPassFilter(sampleRate, lowHz, q);
        this.midFilters[1].setLowPassFilter(sampleRate, highHz, q);
        this.midFilters[2].setHighPassFilter(sampleRate, lowHz, q);
        this.midFilters[3].setLowPassFilter(sampleRate, highHz, q);

        this.highFilters[0].setHighPassFilter(sampleRate, highHz, q);
        this.highFilters[1].setHighPassFilter(sampleRate, highHz, q);
    }

    /**
     * @param {number} inVal
     * */
    process(inVal) {
        this.outLow = inVal;
        this.outMid = inVal;
        this.outHigh = inVal;

        for (let i = 0; i < (this.dbPerOct24 ? 2 : 1); i++) {
            this.outLow = this.lowFilters[i].transform(this.outLow);
        }

        for (let i = 0; i < (this.dbPerOct24 ? 4 : 2); i++) {
            this.outMid = this.midFilters[i].transform(this.outMid);
        }

        for (let i = 0; i < (this.dbPerOct24 ? 2 : 1); i++) {
            this.outHigh = this.highFilters[i].transform(this.outHigh);
        }
    }
}